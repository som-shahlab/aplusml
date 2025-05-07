"""
Functions to parse APLUS config YAML files into Python objects for use in `sim.py`.
Contains a list of required and optional keys for each config section.
"""
from ruamel.yaml import YAML
from aplusml.config import Config
from typing import Any
import yaml

VALID_METADATA_KEYS = {
    'name' : 'optional', 
    'path_to_functions' : 'optional', 
    'path_to_properties' : 'optional',
    'properties_col_for_patient_id' : 'optional',
    'patient_sort_preference_property' : 'optional',
}
VALID_VARIABLE_KEYS = {
    'type' : 'optional',
}
VALID_VARIABLE_SCALAR_KEYS = {
    **VALID_VARIABLE_KEYS,
    'value' : 'required',
}
VALID_VARIABLE_RESOURCE_KEYS = {
    **VALID_VARIABLE_KEYS,
    'init_amount' : 'required',
    'max_amount' : 'required',
    'refill_amount' : 'required',
    'refill_duration' : 'required',
}
VALID_VARIABLE_PROPERTY_FILE_KEYS = {
    **VALID_VARIABLE_KEYS,
    'column' : 'required',
}
VALID_VARIABLE_PROPERTY_CONSTANT_KEYS = {
    **VALID_VARIABLE_KEYS,
    'value' : 'required',
}
VALID_VARIABLE_PROPERTY_DIST_KEYS = {
    **VALID_VARIABLE_KEYS,
    'distribution' : 'required',
    'mean' : 'optional',
    'std' : 'optional',
    'start' : 'optional',
    'end' : 'optional',
}
VALID_VARIABLE_SIMULATION_IDS = [
    'time_left_in_sim', 
    'time_already_in_sim',
    'sim_current_timestep',
]
VALID_STATE_KEYS = {
    'type' : 'optional',
    'label' : 'optional',
    'duration' : 'optional',
    'utilities' : 'optional',
    'transitions' : 'optional',
    'resource_deltas' : 'optional',
}
VALID_TRANSITION_KEYS = {
    'dest' : 'required',
    'label' : 'optional',
    'prob' : 'optional',
    'if' : 'optional',
    'duration' : 'optional',
    'utilities' : 'optional',
    'resource_deltas' : 'optional',
}
VALID_UTILITY_KEYS = {
    'value' : 'optional',
    'if' : 'optional',
    'unit' : 'optional',
}


def load_yaml(path_to_yaml: str) -> dict:
    """Loads YAML file into a Python dictionary.

    Args:
        path_to_yaml (str): Path to YAML file

    Returns:
        dict: Python dictionary
    """
    data = None
    with open(path_to_yaml, "r") as fd:
        try:
            yaml = YAML(typ="safe") # default, if not specfied, is 'rt' (round-trip)
            data = yaml.load(fd)
        except Exception as exc:
            print("ERROR loading YAML:", exc)
    return data

def is_keys_valid(yaml_entry: dict, 
                    yaml_entry_id: str, 
                    yaml_entry_type: str, 
                    valid_keys: dict[str],
                    is_check_required_keys: bool = True,
                    is_check_optional_keys: bool = True) -> bool:
    """Check that the keys in a given YAML entry are valid (i.e. all required keys are present, and no extraneous keys are present).
    
    Args:
        yaml_entry (dict): The YAML entry to check
        yaml_entry_id (str): The ID of the YAML entry
        yaml_entry_type (str): The type of YAML entry
        valid_keys (dict): A dictionary of valid keys for the YAML entry
        is_check_required_keys (bool): Whether to check that all required keys are present
        is_check_optional_keys (bool): Whether to check that no extraneous keys are present

    Returns:
        bool: True if the keys are valid, False otherwise
    """
    yaml_entry_keys = set(yaml_entry.keys())
    # 1. Make sure no extraneous keys
    if is_check_optional_keys:
        invalid_keys = yaml_entry_keys - set(valid_keys.keys())
        if len(invalid_keys) > 0:
            print(f"ERROR - Invalid keys ({invalid_keys}) for {yaml_entry_type} {yaml_entry_id}")
            return False
    # 2. Make sure all required keys are present
    if is_check_required_keys:
        required_keys = set([ key for key, val in valid_keys.items() if val == 'required' ])
        missing_keys = required_keys - yaml_entry_keys
        if len(missing_keys) > 0:
            print(f"ERROR - Missing keys ({missing_keys}) for {yaml_entry_type} {yaml_entry_id}")
            return False
    return True

def _replace_if_with_if_(obj: Any) -> Any:
    """Recursively replace 'if' keys with 'if_' in dicts and lists."""
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            new_key = "if_" if k == "if" else k
            new_obj[new_key] = _replace_if_with_if_(v)
        return new_obj
    elif isinstance(obj, list):
        return [_replace_if_with_if_(item) for item in obj]
    else:
        return obj

def is_valid_config_yaml(yaml: dict) -> bool:
    """Return TRUE if the provided YAML config is valid, FALSE otherwise."""
    #
    # Metadata
    metadata = yaml.get('metadata', {})
    if not is_keys_valid(metadata, 'metadata', 'metadata', VALID_METADATA_KEYS):
        return False

    #
    # Variables
    variables = yaml.get('variables', {})
    all_variable_ids = variables.keys()
    # Cast each variable to a dict from YAML parser
    # variables = { v_id: dict(v) if type(v).__name__ == 'CommentedMap' else v for v_id, v in variables.items() }
    for v_id, v in variables.items():
        assert type(v) == dict, f"ERROR - Variable '{v_id}' must be a dict"
        # Check keys
        type_ = v.get('type', 'scalar')
        if type_ == 'scalar' and not is_keys_valid(v, v_id, 'variable', VALID_VARIABLE_SCALAR_KEYS):
            return False
        elif type_ == 'resource' and not is_keys_valid(v, v_id,'variable', VALID_VARIABLE_RESOURCE_KEYS):
            return False
        elif type_ == 'property':
            if 'column' in v:
                ## If file...
                if 'distribution' in v or 'value' in v:
                    print(f"ERROR - Can only have one of ('column', 'distribution', 'value') keys for variable '{v_id}")
                    return False
                if not is_keys_valid(v, v_id,'variable', VALID_VARIABLE_PROPERTY_FILE_KEYS):
                    return False
            elif 'value' in v:
                ## If constant...
                if 'distribution' in v or 'column' in v:
                    print(f"ERROR - Can only have one of ('column', 'distribution', 'value') keys for variable '{v_id}")
                    return False
                if not is_keys_valid(v, v_id, 'variable', VALID_VARIABLE_PROPERTY_CONSTANT_KEYS):
                    return False
            else:
                ## If distribution...
                if not is_keys_valid(v, v_id, 'variable', VALID_VARIABLE_PROPERTY_DIST_KEYS):
                    return False
        elif type_ == 'simulation':
            if v_id not in VALID_VARIABLE_SIMULATION_IDS:
                print(f"ERROR - Invalid simulation variable name for variable '{v_id}'")
                return False
            if v_id == 'time_left_in_sim':
                # Require 'total_duration_in_sim' variable (otherwise can't calculate)
                if 'total_duration_in_sim' not in all_variable_ids:
                    print(f"ERROR - A variable with the ID 'total_duration_in_sim' is required to use the simulation variable 'time_left_in_sim'")
                    return False
    # Enforce uniqueness
    if len(all_variable_ids) != len(set(all_variable_ids)):
        print("ERROR - Cannot have a repeated variable ID")
        return False
    # Ensure 'patient_sort_preference_property' is an actual property
    patient_sort_preference_property = metadata.get('patient_sort_preference_property', {}).get('variable')
    if patient_sort_preference_property and len([ key for key, val in variables.items() if val.get('type') == 'property' and key == patient_sort_preference_property ]) != 1:
        if patient_sort_preference_property not in [ 'start_timestep', 'id']:
            print("ERROR - The 'variable' key in metadata's 'patient_sort_preference_property' must be the name of a variable with the type 'property' or must be an attribute of the 'Patient' class")
            return False
    #
    # States
    states = yaml.get('states', {})
    all_state_ids = states.keys()
    for s_id, s in states.items():
        assert type(s) == dict, f"ERROR - State '{s_id}' must be a dict"
        if not is_keys_valid(s, s_id, 'state', VALID_STATE_KEYS):
            return False
        # Ensure that all variables in resource_deltas are in the 'variables' section of the YAML
        resource_deltas = s.get('resource_deltas', {})
        if resource_deltas is None:
            print(f"ERROR - No resources specified under 'resource_deltas' for state '{s_id}'. Perhaps you left this blank accidentally?")
            return False
        for v_id in resource_deltas.keys():
            if v_id not in all_variable_ids:
                print(f"ERROR - The variable {v_id} is used in a state's 'resource_deltas', but isn't listed in the 'variables' section")
                return False
        # Utilities
        utilities = s.get('utilities', [])
        if utilities is None:
            print(f"ERROR - No resources specified under 'utilities' for state '{s_id}'. Perhaps you left this blank accidentally?")
            return False
        if isinstance(utilities, list):
            for u in utilities:
                if not is_keys_valid(u, s_id, 'utility', VALID_UTILITY_KEYS):
                    return False
        else:
            if utilities is None:
                print(f"ERROR - No value specified for 'utilities' for state '{s_id}'")
                return False
        # Transitions
        transitions = s.get('transitions', [])
        if transitions is None:
            print(f"ERROR - No resources specified under 'transitions' for state '{s_id}'. Perhaps you left this blank accidentally?")
            return False
        for t in transitions:
            assert type(t) == dict, f"ERROR - Transitions for state '{s_id}' must be dicts"
            if not is_keys_valid(t, s_id, 'transition', VALID_TRANSITION_KEYS):
                return False
            # Ensure that 'if' and 'prob' aren't intermixed
            if 'prob' in t and 'if' in t:
                print(f"ERROR - If you have both 'if' and 'prob' statements in the same transition, then all 'if' statements must precede the 'prob' statements for transition in state '{s_id}'")
                return False
            # Ensure that all variables in resource_deltas are in the 'variables' section of the YAML
            for v_id in t.get('resource_deltas', {}).keys():
                if v_id not in all_variable_ids:
                    print(f"ERROR - The variable {v_id} is used in a transition's 'resource_deltas', but isn't listed in the 'variables' section")
                    return False
            # Utilities
            utilities = t.get('utilities', [])
            if utilities is None:
                print(f"ERROR - No resources specified under 'utilities' for state '{s_id}' and transition '{t}'. Perhaps you left this blank accidentally?")
                return False
            if isinstance(utilities, list):
                for u in utilities:
                    if not is_keys_valid(u, s_id, 'utility', VALID_UTILITY_KEYS):
                        return False
            else:
                if utilities is None:
                    print(f"ERROR - No value specified for 'utilities' for transition in state '{s_id}'")
                    return False
        # Enforce correct # of transitions for start/intermediate/end/ states
        type_ = s.get('type', 'intermediate')
        if type_ == 'start' and len(transitions) == 0:
            print(f"ERROR - state '{s_id}' must have at 1+ transitions because it has type = 'start'")
            return False
        elif type_ == 'intermediate' and len(transitions) == 0:
            print(f"ERROR - state '{s_id}' must have at 1+ transitions because it has type = 'intermediate'")
            return False
        if type_ == 'end' and len(transitions) > 0:
            print(f"ERROR - state '{s_id}' must have exactly 0 transitions because it has type = 'end'")
            return False
            
    # Enforce uniqueness
    if len(all_state_ids) != len(set(all_state_ids)):
        print("ERROR - Cannot have a repeated state ID")
        return False
    return True

def parse_yaml_into_config(path_to_yaml: str) -> Config:
    """
    Parse a YAML file into a Config object, printing errors if parsing or validation fails.

    Args:
        path_to_yaml (str): Path to YAML file.

    Returns:
        Config: Config object.
    """
    try:
        with open(path_to_yaml, "r") as f:
            config_dict = yaml.safe_load(f)
        config_dict = _replace_if_with_if_(config_dict)
        return Config(**config_dict)
    except FileNotFoundError:
        print(f"Error: File not found: {path_to_yaml}")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    pass