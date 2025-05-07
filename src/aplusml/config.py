"""
Pydantic model for APLUS config YAML file.
"""
import os
from typing import Any, Dict, List, Literal, Optional, Union, get_args
import pandas as pd
from pydantic import BaseModel

VALID_DISTRIBUTION_TYPES = Literal["bernoulli", "exponential", "binomial", "normal", "poisson", "uniform"]
VALID_VARIABLE_TYPES = Literal["scalar", "resource", "property", "simulation"]
VALID_STATE_TYPES = Literal["start", "end", "intermediate"]
VALID_SIMULATION_VARIABLE_IDS = Literal["time_left_in_sim", "time_already_in_sim", "sim_current_timestep"]

class ConfigPatientSortPreferenceProperty(BaseModel):
    """
    Specification for patient sort preference property section of config YAML file.
    
    :param variable: Name of a property (must be defined in `variables`) that will be used to sort patients when prioritizing allocation of a finite resource
    :param is_ascending: True = ascending order, False = descending.
    """
    variable: str
    is_ascending: bool

class ConfigMetadata(BaseModel):
    """
    Specification for metadata section of config YAML file.
    
    :param name: Name of the simulation.
    :param path_to_properties: Path to CSV file where each row is a patient, each column is a property. Note: Only properties explicitly enumerated in the 'variables' section will be imported
    :param properties_col_for_patient_id: Column name in the CSV that contains unique patient IDs.
    :param patient_sort_preference_property: Property to sort patients by when prioritizing allocation of a finite resource.
    """
    name: Optional[str] = None
    path_to_properties: Optional[str] = None
    properties_col_for_patient_id: Optional[str] = None
    patient_sort_preference_property: Optional[ConfigPatientSortPreferenceProperty] = None

    def is_valid(self) -> bool:
        """Return TRUE if the ConfigMetadata is valid, FALSE otherwise."""
        if self.properties_col_for_patient_id is not None:
            if self.path_to_properties is None:
                print(f"ERROR - `path_to_properties` must be specified if `properties_col_for_patient_id` is specified")
                return False
        if self.path_to_properties is not None:
            if not os.path.exists(self.path_to_properties):
                print(f"ERROR - `path_to_properties` (path='{self.path_to_properties}') does not exist")
                return False
            df = pd.read_csv(self.path_to_properties)
            if self.properties_col_for_patient_id is not None and self.properties_col_for_patient_id not in df.columns:
                print(f"ERROR - `properties_col_for_patient_id` (col_name='{self.properties_col_for_patient_id}') not found in {self.path_to_properties}")
                return False
        return True

class ConfigVariable(BaseModel):
    """
    Specification for variable section of config YAML file.
    
    :param type: Type of variable. Must be one of: 'scalar', 'resource', 'property', 'simulation'.

    # If type == 'scalar'...
    #   This is a scalar value that is shared across all patients.
    #   It can be used to model things like the sensitivity of a screening test, the prevalence of a disease, etc.
    #   It can be specified by the following parameters...
            :param value: Scalar value. Must be a valid Python type. Use '!!set' tag for sets.

    # If type == 'resource'...
    #   This is a finite resource that is shared across all patients.
    #   It can be decremented, incremented, and reset by the simulation.
    #   It can be used to model things like hospital beds, lab capacity, etc.
    #   It can be specified by the following parameters...
            :param init_amount: Initial amount of the resource.
            :param max_amount: Maximum amount of resource allowed.
            :param refill_amount: Amount added per refill.
            :param refill_duration: Time interval between refills.

    # If type == 'property'...
    #   This is a property that is UNIQUE to each patient (i.e. each patient may have a different value for this property).
    #   It can be used to model things like the age of a patient, the gender of a patient, etc.
    ##  Either load from file...
            :param column: If loaded from a CSV file, specify the column name (e.g. 'y' or 'y_hat_dl'). Each row of the CSV will be a patient, and the value of this property for each patient will be the value of the column in the CSV file.
    ##  Or specify a constant value...
            :param value: Scalar value. Must be a valid Python type. Use '!!set' tag for sets.
    ##  Or randomly sample from a distribution...
            :param distribution: If randomly sampled.
            :param mean: Mean value for distribution.
            :param std: Standard deviation.
            :param start: Minimum value.
            :param end: Maximum value.

    # If type == 'simulation', then `value` must be one of the following...
            :param time_left_in_sim: The number of timesteps remaining in the simulation.
            :param time_already_in_sim: The number of timesteps that have passed in the simulation.
            :param sim_current_timestep: The current timestep of the simulation.
    """
    type: VALID_VARIABLE_TYPES = "scalar"
    # Scalar value
    value: Optional[Union[int, float, bool, str, list, dict, set]] = None
    # Resource value
    init_amount: Optional[int] = None
    max_amount: Optional[int] = None
    refill_amount: Optional[int] = None
    refill_duration: Optional[int] = None
    # Property value
    column: Optional[str] = None
    # Simulation value
    distribution: Optional[VALID_DISTRIBUTION_TYPES] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    start: Optional[float] = None
    end: Optional[float] = None

    def is_valid(self, id: str) -> bool:
        """Return TRUE if the ConfigVariable is valid, FALSE otherwise."""
        if self.type == 'scalar':
            # This is a scalar value that is shared across all patients.
            # It can be used to model things like the sensitivity of a screening test, the prevalence of a disease, etc.
            if self.value is None:
                print(f"ERROR - `value` must be specified if `type` is 'scalar' for variable '{id}'")
                return False
        elif self.type == 'resource':
            # This is a finite resource that is shared across all patients. 
            # It can be decremented, incremented, and reset by the simulation. 
            # It can be used to model things like hospital beds, lab capacity, etc.
            if self.init_amount is None:
                print(f"ERROR - `init_amount` must be specified if `type` is 'resource' for variable '{id}'")
                return False
            if self.max_amount is None:
                print(f"ERROR - `max_amount` must be specified if `type` is 'resource' for variable '{id}'")
                return False
            if self.refill_amount is None:
                print(f"ERROR - `refill_amount` must be specified if `type` is 'resource' for variable '{id}'")
                return False
            if self.refill_duration is None:
                print(f"ERROR - `refill_duration` must be specified if `type` is 'resource' for variable '{id}'")
                return False
        elif self.type == 'property':
            # This is a property that is UNIQUE to each patient (i.e. each patient may have a different value for this property).
            # It can be used to model things like the age of a patient, the gender of a patient, etc.
            if len([ key for key in [ self.column, self.value, self.distribution ] if key is not None ]) > 1:
                print(f"ERROR - Can only have one of ('column', 'value', 'distribution') keys for variable '{id}'")
                return False
        elif self.type == 'simulation':
            # This is a simulation variable that is shared across all patients.
            # It can be used to model things like the time left in the simulation, the time already in the simulation, etc.
            # It must be one of the pre-defined simulation variable IDs
            if id not in get_args(VALID_SIMULATION_VARIABLE_IDS):
                print(f"ERROR - Invalid simulation variable name for variable '{id}'. Must be one of: {VALID_SIMULATION_VARIABLE_IDS}")
                return False
        return True

class ConfigUtility(BaseModel):
    """
    Specification for utility section of config YAML file.
    
    :param value: If str, it's evaluated as a Python expression.
    :param if_: A Python expression. If it evaluates to TRUE, then the `value` for this utility is set to this `value`. Note: These 'if' statements are not mutually exclusive (i.e. if multiple conditions within the same State evaluate to TRUE, then they will simply be summed together)
    :param unit: Measurement unit. Default: ''.
    """
    value: Optional[Union[int, float, str]] = None
    if_: Optional[Union[str, bool]] = None
    unit: str = ''
    
    def __repr__(self):
        return f"Utility(value={self.value}, if_={self.if_}, unit={self.unit})"

    def is_valid(self, state_id: str) -> bool:
        """Return TRUE if the ConfigUtility is valid, FALSE otherwise."""
        return True

class ConfigTransition(BaseModel):
    """
    Specification for transition section of config YAML file.
    
    :param dest: ID of the destination state.
    :param label: Human-readable label for the transition. Default: "".
    :param if_: A Python expression. If it evaluates to TRUE, then the transition is taken.
    :param prob: Probability of the transition.
    :param duration: Number of timesteps to wait before transitions are evaluated. Default: 0
    :param utilities: If str, float, or bool, it's evaluated as a Python expression. Default: [].
    :param resource_deltas: Changes to resource levels from taking this transition. Default: {}. [key] = name of a resource defined in `variables`. [value] = how much to change each resource level AS SOON AS this transition is taken
    """
    dest: str
    label: Optional[str] = ""
    # Transition conditions
    ## Can either have...
    ## - All transitions have an 'if' condition (where if the last transition doesn't have an 'if', it defaults to always TRUE)
    ## - All transitions have a 'prob' condition (where if the last transition doesn't have a 'prob', it defaults to = 1 - (sum of other probs))
    ## - The first set of transitions have an 'if' condition, but the second set have a 'prob'
    if_: Optional[Union[str, bool]] = None
    prob: Optional[Union[str, float, int]] = None
    duration: int = 0
    utilities: Union[str, int, float, bool, List[ConfigUtility]] = []
    resource_deltas: Dict[str, float] = {}
    
    def is_valid(self, state_id: str) -> bool:
        """Return TRUE if the ConfigTransition is valid, FALSE otherwise."""
        return True

class ConfigState(BaseModel):
    """
    Specification for state section of config YAML file.
    
    :param type: Whether the state is a start, end, or intermediate state within the workflow. Default: "intermediate".
    :param label: Human-readable label for the state. Default: value of `key`.
    :param transitions: List of possible state transitions.
    :param duration: Number of timesteps to wait before transitions are evaluated. Default: 0
    :param utilities: If str, float, or bool, it's evaluated as a Python expression. Default: [].
    :param resource_deltas: Changes to resource levels from entering this state. Default: {}. [key] = name of a resource defined in `variables`. [value] = how much to change each resource level AS SOON AS this state is hit
    """
    type: VALID_STATE_TYPES = "intermediate"
    label: Optional[str] = None
    transitions: List[ConfigTransition] = []
    duration: int = 0
    utilities: Union[str, int, float, bool, List[ConfigUtility]] = []
    resource_deltas: Dict[str, float] = {}
    
    def is_valid(self, id: str) -> bool:
        """Return TRUE if the ConfigState is valid, FALSE otherwise."""
        if self.type not in get_args(VALID_STATE_TYPES):
            print(f"ERROR - Invalid state type. Must be one of: {VALID_STATE_TYPES}")
            return False
        return True

class Config(BaseModel):
    """Specification for config YAML file."""
    metadata: ConfigMetadata
    variables: Dict[str, ConfigVariable] = {} # [key] = variable id, [value] = variable value
    states: Dict[str, ConfigState] = {} # [key] = state id, [value] = state value
    
    def is_valid(self) -> bool:
        """Return TRUE if the Config is valid, FALSE otherwise."""
        #
        # Metadata
        metadata = self.metadata
        if not isinstance(metadata, ConfigMetadata):
            print(f"ERROR - Metadata must be of type `ConfigMetadata`, but is of type {type(metadata)}")
            return False
        if not metadata.is_valid():
            print(f"ERROR - Metadata is invalid. Metadata: {metadata}")
            return False

        #
        # Variables
        variables = self.variables
        all_variable_ids: List[str] = list(variables.keys())
        # Cast each variable to a dict from YAML parser
        for v_id, v in variables.items():
            # Check type
            if not isinstance(v, ConfigVariable):
                print(f"ERROR - Variable '{v_id}' must be of type `ConfigVariable`, but is of type {type(v)}")
                return False
            # Check internal validity
            if not v.is_valid(v_id):
                print(f"ERROR - Variable '{v_id}' is invalid")
                return False
            # Check variable names
            if v.type == 'simulation':
                if v_id == 'time_left_in_sim':
                    # Require 'total_duration_in_sim' variable (otherwise can't calculate)
                    if 'total_duration_in_sim' not in all_variable_ids:
                        print(f"ERROR - A variable with the ID 'total_duration_in_sim' is required to use the simulation variable 'time_left_in_sim'")
                        return False
        # Enforce unique variable IDs
        if len(all_variable_ids) != len(set(all_variable_ids)):
            print(f"ERROR - Cannot have a repeated variable ID. Instead, found: {all_variable_ids}")
            return False
        
        # Ensure 'patient_sort_preference_property' is an actual property
        patient_sort_preference_property: Optional[Dict[str, Any]] = metadata.patient_sort_preference_property
        if patient_sort_preference_property and len([ key for key, val in variables.items() if val.type == 'property' and key == patient_sort_preference_property.get('variable') ]) != 1:
            if patient_sort_preference_property.get('variable') not in [ 'start_timestep', 'id']:
                print("ERROR - The 'variable' key in metadata's 'patient_sort_preference_property' must be the name of a variable with the type 'property' or must be an attribute of the 'Patient' class")
                return False
        
        #
        # States
        states = self.states
        all_state_ids: List[str] = list(states.keys())
        for s_id, s in states.items():
            if not isinstance(s, ConfigState):
                print(f"ERROR - State '{s_id}' must be of type `ConfigState`, but is of type {type(s)}")
                return False
            # Check internal validity
            if not s.is_valid(s_id):
                print(f"ERROR - State '{s_id}' is invalid")
                return False
            # Ensure that all variables in resource_deltas are in the 'variables' section of the YAML
            resource_deltas: Dict[str, float] = s.resource_deltas
            for v_id in resource_deltas.keys():
                if v_id not in all_variable_ids:
                    print(f"ERROR - The variable {v_id} is used in a state's 'resource_deltas', but isn't listed in the 'variables' section")
                    return False
            # Utilities
            utilities: Union[str, float, bool, List[ConfigUtility]] = s.utilities
            if isinstance(utilities, list):
                for u in utilities:
                    if not u.is_valid(s_id):
                        print(f"ERROR - Utility '{u}' is invalid")
                        return False
            # Transitions
            transitions: List[ConfigTransition] = s.transitions
            for t in transitions:
                if not isinstance(t, ConfigTransition):
                    print(f"ERROR - Transition '{t}' must be of type `ConfigTransition`, but is of type {type(t)}")
                    return False
                if not t.is_valid(s_id):
                    print(f"ERROR - Transition '{t}' is invalid")
                    return False
                # Ensure that all variables in resource_deltas are in the 'variables' section of the YAML
                for v_id in t.resource_deltas.keys():
                    if v_id not in all_variable_ids:
                        print(f"ERROR - The variable {v_id} is used in a transition's 'resource_deltas', but isn't listed in the 'variables' section")
                        return False
                # Utilities
                utilities: Union[str, float, bool, List[ConfigUtility]] = t.utilities
                if isinstance(utilities, list):
                    for u in utilities:
                        if not u.is_valid(s_id):
                            print(f"ERROR - Utility '{u}' is invalid")
                            return False
            # Enforce correct # of transitions for start/intermediate/end/ states
            type_: str = s.type
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
            print(f"ERROR - Cannot have a repeated state ID, but found: {all_state_ids}")
            return False
        return True

if __name__ == "__main__":
    config = Config(
        metadata=ConfigMetadata(
            name="Hello World Workflow",
            path_to_properties="patient_properties.csv",
            properties_col_for_patient_id="patient_id",
        ),
        variables={
            "patient_property_1": ConfigVariable(type="property", column="patient_property_1"),
            "patient_property_2": ConfigVariable(type="property", column="patient_property_2"),
        },
        states={
            "start": ConfigState(type="start"),
            "end": ConfigState(type="end"),
        },
    )