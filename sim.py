import collections
import io
import random
from types import CodeType
from typing import Any, Callable, Tuple, Union
import numpy as np
import pandas as pd
import ast
import pickle
import pydot
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Utility(object):
    def __init__(self,
                 value: str,
                 unit: str = '',
                 _if: str = None):
        self.value: str = value
        self.unit: str = unit
        self._if: str = _if
        self._if_compiled: CodeType = compile(_if, '<string>', 'eval', optimize=2) if type(_if) == str else None
        self.value_compiled: CodeType = compile(value, '<string>', 'eval', optimize=2) if type(value) == str else None

    def __setattr__(self, name, value):
        # Update compiled versions of if/value
        if name == '_if':
            super().__setattr__('_if_compiled', compile(value, '<string>', 'eval', optimize=2) if type(value) == str else None)
        if name == 'value':
            super().__setattr__('value_compiled', compile(value, '<string>', 'eval', optimize=2) if type(value) == str else None)
        super().__setattr__(name, value)
    
    def is_conditional_if(self):
        return self._if is not None
    
    def __repr__(self):
        return str({
            'value' : self.value,
            'unit' : self.unit,
            '_if' : self._if,
        })
    
    def serialize(self):
        return {
            'value' : self.value,
            'unit' : self.unit,
            '_if' : self._if,
        }

class Transition(object):
    def __init__(self, 
                 dest: str,
                 label: str,
                 duration: int,
                 utilities: list[Utility],
                 resource_deltas: dict,
                 _if: Union[str, bool] = None,
                 prob: Union[str, float] = None):
        self.dest: str = dest
        self.label: str = label
        self.duration: int = duration
        self.utilities: list[Utility] = utilities
        self.resource_deltas: dict = resource_deltas
        self._if: Union[str, bool] = _if # NOTE: This is referred to as 'if' outside of this object
        self.prob: Union[str, float] = prob
        self._if_compiled: CodeType = compile(_if, '<string>', 'eval', optimize=2) if type(_if) == str else None
        self.prob_compiled: CodeType = compile(prob, '<string>', 'eval', optimize=2) if type(prob) == str else None
        
    def __setattr__(self, name, value):
        # Update compiled versions of if/prob
        if name == '_if':
            super().__setattr__('_if_compiled', compile(value, '<string>', 'eval', optimize=2) if type(value) == str else None)
        if name == 'prob':
            super().__setattr__('prob_compiled', compile(value, '<string>', 'eval', optimize=2) if type(value) == str else None)
        super().__setattr__(name, value)

    def is_conditional_prob(self):
        return self.prob is not None
    def is_conditional_if(self):
        return self._if is not None
    
    def get_variables_in_conditional(self) -> list[str]:
        expression = ''
        # Determine where to find conditional in Transition
        if self.is_conditional_prob():
            expression = self.prob
        elif self.is_conditional_if():
            expression = self._if
        else:
            # If there is not a conditional, then there can't be any variables involved
            return []
        # If the conditional is not a string (i.e. is a float or bool), then there can't be any variables involved
        if type(expression) != str:
            return []
        # Parse conditional expression for variables
        parsed_expression = ast.parse(expression)
        parsed_variable_ids: list[str] = []
        for node in ast.walk(parsed_expression):
            if type(node) is ast.Name:
                parsed_variable_ids.append(node.id)
        return parsed_variable_ids

    def print(self):
        return f"=> {self.dest} ({self.label})"

    def __repr__(self):
        return str({
            'dest' : self.dest,
            'label' : self.label,
            'duration' : self.duration,
            'utilities' : self.utilities,
            'if' : self._if,
            'prob' : self.prob,
        })
    
    def serialize(self):
        return {
            'dest' : self.dest,
            'label' : self.label,
            'duration' : self.duration,
            'utilities' : [ u.serialize() for u in self.utilities ],
            'resource_deltas' : self.resource_deltas,
            '_if' : self._if,
            'prob' : self.prob,
        }

class State(object):
    def __init__(self, id: str,
                 label: str,
                 type: str,
                 duration: int,
                 utilities: list[Utility],
                 transitions: list[Transition],
                 resource_deltas: dict):
        self.id: str = id
        self.label: str = label
        self.type: str = type
        self.duration: int = duration
        self.utilities: list[Utility] = utilities
        self.transitions: list[Transition] = transitions
        self.resource_deltas: dict = resource_deltas

    def print(self):
        return f"{self.id} | {self.label}"

    def __repr__(self):
        return str({
            'id' : self.id,
            'label' : self.label,
            'type' : self.type,
            'duration' : self.duration,
            'utilities' : self.utilities,
            'transitions' : [ x.print() for x in self.transitions ],
        })

    def serialize(self):
        return {
            'id' : self.id,
            'label' : self.label,
            'type' : self.type,
            'duration' : self.duration,
            'utilities' : [ x.serialize() for x in self.utilities ],
            'transitions' : [ x.serialize() for x in self.transitions ],
            'resource_deltas' : self.resource_deltas,
        }

class History(object):
    def __init__(self, 
                 current_timestep: int,
                 state_id: str,
                 transition_idx: int, # Transition == state.transitions[idx]
                 state_utility_idxs: list[int], # Utilities == state.utilities[idxs]
                 transition_utility_idxs: list[int], # Utilities == state.transitions[idx].utilities[idxs]
                 state_utility_vals: list[float], # Evaluated Utility Values == evaluate_utility_value(state.utilities[idxs].value)
                 transition_utility_vals: list[float], # EvaluatedUtility Values == evaluate_utility_value(state.transitions[idx].utilities[idxs].value)
                 sim_variables: dict):
        self.current_timestep: int = current_timestep
        self.state_id: str = state_id
        self.transition_idx: Union[int, None] = transition_idx
        self.state_utility_idxs: list[int] = state_utility_idxs
        self.transition_utility_idxs: list[int] = transition_utility_idxs
        self.state_utility_vals: list[float] = state_utility_vals
        self.transition_utility_vals: list[float] = transition_utility_vals
        self.sim_variables: dict = sim_variables
    def __repr__(self):
        return str({
            'current_timestep' : self.current_timestep,
            'state_id' : self.state_id,
            'transition_idx' : self.transition_idx,
            'state_utility_idxs' : self.state_utility_idxs,
            'transition_utility_idxs' : self.transition_utility_idxs,
            'state_utility_vals' : self.state_utility_vals,
            'transition_utility_vals' : self.transition_utility_vals,
        })
    
class Patient(object):
    def __init__(self, 
                 id: str, 
                 start_timestep: int,
                 properties: dict = None):
        self.id: str = id
        self.start_timestep: int = int(start_timestep) # Start time for this patient (i.e. admitted date)
        self.properties: dict = properties if properties is not None else {} # Patient specific properties, i.e. "y_hat" or "y" or "los"
        self.history: list[History]= [] # Track history of (state, transition, utility)
        self.current_state: str = None # ID of current state
    
    def print_state_history(self, is_show_timesteps: bool = False):
        if is_show_timesteps:
            return " > ".join([ f"({h.current_timestep}) {h.state_id}" for h in self.history])
        else:
            return " > ".join([ h.state_id for h in self.history])

    def get_sum_utilities(self, simulation) -> dict:
        sums = collections.defaultdict(float) # [key] = unit, [value] = sum of that unit's utility across entire Patient's history
        for h in self.history:
            # State utilities
            state = simulation.states[h.state_id]
            for i, idx in enumerate(h.state_utility_idxs):
                u = state.utilities[idx]
                sums[u.unit] += h.state_utility_vals[i]
            # Transition utilities (if transition exists)
            if h.transition_idx is not None:
                transition = state.transitions[h.transition_idx]
                for i, idx in enumerate(h.transition_utility_idxs):
                    u = transition.utilities[idx]
                    sums[u.unit] += h.transition_utility_vals[i]
        return dict(sums)

    def __repr__(self):
        return str({
            'id' : self.id,
            'start_timestep' : self.start_timestep,
            'properties' : self.properties,
            'history' : self.history,
            'current_state' : self.current_state,
        })

class Simulation(object):
    def __init__(self):
        self.metadata = {}
        self.variables: dict[dict] = {} # [key] = id, [value] = dict
        self.states: dict[State] = {} # [key] = id, [value] = State
        self.current_timestep: int = None
    
    def evaluate_variables(self, patient: Patient) -> dict:
        variable_to_value = {}
        for v_id, v in self.variables.items():
            if v['type'] == 'scalar':
                variable_to_value[v_id] = v['value']
            elif v['type'] == 'resource':
                variable_to_value[v_id] = v['level']
            elif v['type'] == 'property':
                variable_to_value[v_id] = patient.properties[v_id]
            elif v['type'] == 'simulation':
                if v_id == 'time_left_in_sim':
                    # If patient LOS = 1, t = 0, patient start = 0, then 'time_left_in_sim' = 1
                    variable_to_value[v_id] = max(0, patient.start_timestep + patient.properties['total_duration_in_sim'] - self.current_timestep)
                elif v_id == 'time_already_in_sim':
                    # If t = 1, patient start = 0, then 'time_already_in_sim' = 1
                    variable_to_value[v_id] = max(0, self.current_timestep - patient.start_timestep)
                elif v_id == 'sim_current_timestep':
                    variable_to_value[v_id] = self.current_timestep
            elif v['type'] == 'function':
                # TODO
                variable_to_value[v_id] = v_id()
            else:
                print("ERROR - Invalid 'type' for variable:", v_id)
        assert len(variable_to_value) == len(self.variables)
        return variable_to_value

    def evaluate_expression(self, patient: Patient, 
                            expression: Any,
                            variables: dict,
                            expression_compiled: CodeType = None) -> Any:
        if type(expression) in [bool, int, float]:
            return expression
        elif type(expression) != str:
            # NOTE: This is necessary b/c eval() won't accept a bool
            return None
        # Evaluate expression
        try:
            if expression_compiled:
                return eval(expression_compiled, {}, variables)
            else:
                return eval(expression, {}, variables)
        except NameError as e:
            print(f"ERROR - Missing variable ({e}) for expression: {expression}")
            return None
        
    def evaluate_transition_if(self, patient: Patient, 
                               transition: Transition,
                               variables: dict) -> bool:
        if not transition.is_conditional_if():
            # If there is no 'if', return TRUE
            return True
        return self.evaluate_expression(patient, transition._if, variables, transition._if_compiled)

    def evaluate_transition_prob(self, patient: Patient, 
                                 transition: Transition,
                                 variables: dict) -> bool:
        if not transition.is_conditional_prob():
            # If there is no 'prob', return None (default will be set to: 1 - (sum of other probs))
            return None
        return self.evaluate_expression(patient, transition.prob, variables, transition.prob_compiled)

    def evaluate_utility_if(self, patient: Patient, 
                            utility: Utility,
                            variables: dict) -> bool:
        if not utility.is_conditional_if():
            # If there is no 'if', return TRUE
            return True
        return self.evaluate_expression(patient, utility._if, variables, utility._if_compiled)

    def evaluate_utility_value(self, patient: Patient, 
                               utility: Utility,
                               variables: dict) -> Any:
        return self.evaluate_expression(patient, utility.value, variables, utility.value_compiled)
            
    def select_transition(self, patient: Patient, 
                          transitions: list[Transition],
                          variables: dict) -> int:
        """From a set of transitions, returns the idx of the one to take
            First, this evaluates all conditional transitions, i.e. 'if' statements
            If all 'if' are FALSE, then evaluate all probabilistic transitions, i.e. 'prob' statements
                Thus, the probabilistic transitions are conditional on the 'if' statements all being FALSE
        """    
        # Conditional transition
        start_idx_for_probs = 0
        for t_idx, t in enumerate(transitions):
            if t.is_conditional_prob():
                start_idx_for_probs = t_idx
                break
            value = self.evaluate_transition_if(patient, t, variables)
            if value:
                # NOTE: Must just be `value` to evaluate Truthiness -- i.e. can't have "value is True" here,
                # otherwise code like "X is True" will return false when eval() returns an np.bool_(True)
                # b/c "is" compares X to the Python True, otherwise returns false
                # See: https://stackoverflow.com/questions/27276610/boolean-identity-true-vs-is-true
                return t_idx
            elif value is None:
                return None
        # Probabilistic transition
        prob_transitions = transitions[start_idx_for_probs:]
        probs = [ self.evaluate_transition_prob(patient, t, variables) for t in prob_transitions ]
        if probs.count(None) > 1:
            # If there are multiple Nones, throw error
            print(f"ERROR - Multiple transitions are missing a 'prob' value: {transitions}")
            return None
        elif probs.count(None) == 1:
            # If there is one None, set it = 1 - (sum of other probs)
            probs[probs.index(None)] = 1 - sum(filter(None, probs))
        else:
            # If there is no catch-all (i.e. no `None`)...
            if sum(probs) != 1:
                # ...And probs don't sum to 1, throw error
                print(f"ERROR - Values for 'prob' don't sum to 1: {transitions}")
                return None
        t_idx = start_idx_for_probs + random.choices(range(len(prob_transitions)), weights = probs)[0]
        return t_idx

    def init_variables(self, variables: list[dict]):
        """Modifies 'variables' in place

        Args:
            variables (list[dict]): From YAML
        """        
        # Resource initial amounts
        for v_id, v in variables.items():
            if v['type'] == 'resource':
                variables[v_id]['level'] = v['init_amount']
                variables[v_id]['last_refill_timestep'] = 0

    def init_run(self, random_seed: int = 0):
        # Track simulation timesteps
        self.current_timestep: int = 0 # Current simulation timestep
        # Initialize variables
        self.init_variables(self.variables)
        # Random seed
        np.random.seed(random_seed)
        random.seed(random_seed)
    
    def is_valid_patients(self, patients: list[Patient]) -> Tuple[bool, str]:
        """Returns true if 'patients' are valid

        Args:
            patients (list[Patient]): NOTE: These aren't imported from YAML, but must be programmatically generated

        Returns:
            bool: TRUE if ecah Patient obj meets validation criteria
            str: Error message
        """
        is_unique_ids = len(set([ p.id for p in patients ])) == len(patients)
        if not is_unique_ids:
            return False, f"Patients must all have unique IDs"
        return True, ""

    def init_patients(self, patients: list[Patient]):
        """Modifies 'patients' in place

        Args:
            patients (list[Patient]): NOTE: These aren't imported from YAML, but must be programmatically generated
        """        
        for p in patients:
            # Current state = 'start'
            p.current_state = 'start'
            # History
            p.history = []

    def log(self, string: str):
        if self.is_print_log:
            print(f"{self.current_timestep} | {string}")

    def run(self, 
            all_patients: list[Patient],
            max_timesteps: int = None,
            random_seed: int = 0,
            is_print_log: bool = False) -> list[Patient]:
        """This takes about 3 seconds to run for 15,000 patients, 10 seconds to run for 50,000 patients

        Args:
            all_patients (list[Patient]): Contains all patients to be simulated, across all admit days. 
                NOTE: This modifies all_patients in place
                The only attributes that are modified are `history` and `current_state`
            max_timesteps (int, optional): End simulation after running this timestep (i.e. max_timesteps = 0, then immediately break; if max_timesteps=2, then run t=0, t=1 then break)
                NOTE: If break b/c of max_timesteps, then: simulation.current_timestep = max_timesteps - 1
            random_seed (int, optional): Used for 'init_run()'. Defaults to 0.
            is_print_log (bool, optional): If FALSE, then don't print anything to console. Defaults to FALSE.
        """
        self.is_print_log = is_print_log

        # Track patients admitted after current value of 'self.current_timestep'
        admitted_patients: list[Patient] = []
        # Track patients waiting some # of timesteps
        paused_patients: dict[int] = {} # [key] = patient ID
        # Track patients who were just unpaused
        unpaused_patients: dict[bool] = {} # [key] = patient ID
        # Track patients who hit an end state
        finished_patients: dict[bool] = {} # [key] = patient ID
        
        # Reset simulation to initialize for run
        self.init_patients(all_patients)
        self.init_run(random_seed)
        
        # Validate input
        is_valid_patients, msg = self.is_valid_patients(all_patients)
        assert is_valid_patients, f"ERROR - Patient are invalid - {msg}"
        
        # 
        # Sort all_patients by `start_timestep`
        # 
        all_patients = sorted(all_patients, key = lambda x : x.start_timestep)
        ## Track which patient is the latest cutoff for admittance
        current_timestep_to_all_patients_idx = {}
        for p_idx, p in enumerate(all_patients):
            current_timestep_to_all_patients_idx[p.start_timestep] = p_idx
        most_recent_current_timestep_to_all_patients_idx = current_timestep_to_all_patients_idx.get(0, 0)
        ## Track which patient is the earliest cutoff for unfinished patients
        earliest_unfinished_patient_tuple: Tuple[int, int] = (0, 0) # Tuple (X,Y) where X = idx of patient with 'start_timestep' = Y, where X is the smallest idx in 'all_patients' which corresponds to an unfinished patient

        # Progress all patients until all are finished or hit `max_timestamp`...
        while True:
            #
            # Check if we should end this simulation
            #
            if max_timesteps is not None and self.current_timestep >= max_timesteps:
                self.current_timestep -= 1 # NOTE: Don't change this
                self.log(f"Max timestep exceeded @ {self.current_timestep}")
                break
            if len(all_patients) <= len(finished_patients):
                self.current_timestep -= 1 # NOTE: Don't change this
                self.log(f"All patients are finished @ {self.current_timestep}")
                break
            self.log(f"Timestep: {self.current_timestep}")
            
            #
            # Replenish resources
            #
            for _, v in self.variables.items():
                if v.get('type', 'scalar') == 'resource':
                    if self.current_timestep - v['last_refill_timestep'] >= v['refill_duration']:
                        # Refill resource
                        v['last_refill_timestep'] = self.current_timestep
                        v['level'] += v['refill_amount']
                        # Cap resource at 'max_amount'
                        v['level'] = min(v['max_amount'], v['level'])
                    assert v['level'] <= v['max_amount'], f"ERROR - Variable '{v}' value for 'level' exceeded 'max_amount' during REFILL"
            #
            # Admit new patients
            #
            if self.current_timestep in current_timestep_to_all_patients_idx:
                most_recent_current_timestep_to_all_patients_idx = current_timestep_to_all_patients_idx[self.current_timestep]
            admitted_patients_start_idx: int = earliest_unfinished_patient_tuple[0] # NOTE: Need to save this info here for usage in 'while: True' loop
            admitted_patients_end_idx: int = most_recent_current_timestep_to_all_patients_idx + 1
            admitted_patients: list[Patient] = all_patients[admitted_patients_start_idx:admitted_patients_end_idx]
            ## Set 'earliest_unfinished_patient_tuple' to best possible case (i.e. assume all 'admitted_patients' finish)
            ## We'll progressively decrease this value to "worse" cases (i.e. earlier patients in 'admitted_patients') as we loop through them below
            earliest_unfinished_patient_tuple: Tuple[int,int] = (admitted_patients_end_idx, all_patients[min(admitted_patients_end_idx, len(all_patients) - 1)].start_timestep)
            #
            # Process admitted patients
            #
            # Within a timestep, ordering of patients is arbitrary, so process them
            # in order of our constraint preference 
            # (NOTE: this avoids the need to do separate processing on individual constraints)
            patient_sort_preference_variable: str = self.metadata['patient_sort_preference_property'].get('variable') if self.metadata['patient_sort_preference_property'] else None
            patient_sort_preference_is_ascending: str = self.metadata['patient_sort_preference_property'].get('is_ascending') if self.metadata['patient_sort_preference_property'] else None
            sort_patient_by_preference(admitted_patients, property_to_sort_by=patient_sort_preference_variable, is_ascending=patient_sort_preference_is_ascending)
            # print(self.current_timestep, "|", len(admitted_patients), admitted_patients_start_idx, ':', admitted_patients_end_idx, [ p.id for p in admitted_patients ])
            for p_idx, p in enumerate(admitted_patients):
                while True:
                    # Progress patient until hit pause or finish...
                    if p.id in finished_patients:
                        break
                    if p.id in paused_patients:
                        # Note that this patient is unfinished
                        earliest_unfinished_patient_tuple = (p_idx + admitted_patients_start_idx, p.start_timestep) if p_idx + admitted_patients_start_idx < earliest_unfinished_patient_tuple[0] else earliest_unfinished_patient_tuple
                        break
                    # Go through 'current_state'...
                    current_state: State = self.states[p.current_state]
                    transition: Transition = None
                    # Track if we need to "wait X timesteps" AS SOON AS state is hit (unless we've already waited, i.e. patient is in 'unpaused_patients')
                    if current_state.duration > 0:
                        if p.id in unpaused_patients:
                            # We HAVE already waited the requisite timesteps for 'current_state', so continue with rest of iteration
                            del unpaused_patients[p.id]
                        else:
                            # We HAVE NOT already waited the requisite timesteps for 'current_state' (i.e. this is the first time we're hitting this state)
                            paused_patients[p.id] = current_state.duration
                            continue
                    # Evaluate variables
                    variables = self.evaluate_variables(p)
                    # Select TRANSITION / Update if patient is 'finished' with workflow (i.e. has reached an end state)
                    transition_idx: int = None
                    if current_state.type == 'end':
                        # If this is an END state, add patient to 'finished_patients'
                        finished_patients[p.id] = 1
                    else:
                        # Select transition
                        transition_idx = self.select_transition(p, current_state.transitions, variables)
                        assert transition_idx is not None, f"ERROR - No transition conditional is TRUE for patient '{p.id}' given transitions: {current_state.transitions}"
                        transition = current_state.transitions[transition_idx]
                    # Determine STATE / TRANSITION utilities
                    state_utility_idxs, transition_utility_idxs = [], []
                    state_utility_vals, transition_utility_vals = [], []
                    for utils_idx, utils in enumerate([ current_state.utilities, transition.utilities if transition else [] ]):
                        for u_idx, u in enumerate(utils):
                            # Check that 'if' statement is TRUE (if present)
                            if not self.evaluate_utility_if(p, u, variables):
                                continue
                            # Evaluate value
                            u_val = self.evaluate_utility_value(p, u, variables)
                            if utils_idx == 0: 
                                state_utility_idxs.append(u_idx)
                                state_utility_vals.append(u_val)
                            else: 
                                transition_utility_idxs.append(u_idx)
                                transition_utility_vals.append(u_val)
                    # Record history
                    p.history.append(History(self.current_timestep,
                                             current_state.id,
                                             transition_idx,
                                             state_utility_idxs,
                                             transition_utility_idxs,
                                             state_utility_vals,
                                             transition_utility_vals,
                                             variables,
                                            )
                                    )
                    # Take transition
                    next_state: str = None
                    if transition:
                        # Move patient to 'next_state'
                        assert transition.dest in self.states, f"ERROR - Transition dest '{transition.dest}' not in 'states' section of YAML"
                        next_state = transition.dest
                        # Track if we need to "wait X timesteps" AFTER we take this transition (NOTE: we don't need to do an 'unpaused_patients' check, like we do for state, b/c it's impossible for this transition to have already been taken)
                        if transition.duration > 0:
                            assert p.id not in paused_patients
                            paused_patients[p.id] = transition.duration
                    p.current_state = next_state
                    # Decrement variables used in this STATE or TRANSITION
                    resource_deltas: dict = { **current_state.resource_deltas, **(transition.resource_deltas if transition else {}) }
                    for v_id, delta in resource_deltas.items():
                        # Add 'delta' to resource
                        assert v_id in self.variables, f"ERROR - Variable '{v_id}' is not in the 'variables' section of the YAML, as it is used in the 'resource_deltas' of a state or transition"
                        self.variables[v_id]['level'] += delta
                        # Cap resource at 'max_amount'
                        self.variables[v_id]['level'] = min(self.variables[v_id]['max_amount'], self.variables[v_id]['level'])
                        assert self.variables[v_id]['level'] <= self.variables[v_id]['max_amount'], f"ERROR - Variable '{v_id}' value for 'level' exceeded 'max_amount' "
                    self.log(f"=> {transition.dest if transition else 'N/A'}")
            #
            # By this point, all patients are either finished or paused
            # Now, we take a timestep forward
            #
            
            # For all paused patients, advance them one timestep...
            for p_id in list(paused_patients.keys()):
                time_left = paused_patients[p_id] - 1
                if time_left <= 0:
                    del paused_patients[p_id]
                    unpaused_patients[p_id] = 1
                else:
                    paused_patients[p_id] = time_left
            self.current_timestep += 1
    
    def get_all_utility_units(self) -> list[str]:
        """Returns a list containing all the unit names for utilities (across both states and transitions)
            i.e. if a simulation tracks "QALY" and "USD", this returns the list ["QALY", "USD"]
        """
        units = []
        for s in self.states.values():
            for u in s.utilities:
                units.append(u.unit)
            for t in s.transitions:
                for u in t.utilities:
                    units.append(u.unit)
        return units
    
    def draw_workflow_diagram(self, path_to_file: str = None, figsize: Tuple[int, int] = (20, 20)):
        """Visualize (states, transitions) as a diagram using pydot
        """
        dot_graph = pydot.Dot(graph_type='digraph')

        colors = [
            'blue',
            'red',
            'green',
            'purple',
            'brown',
            'olive',
            'cyan',
            'darkblue',
            'darkslategray',
            'orange',
            'maroon',
            'darkcyan',
        ]
        for idx, state in enumerate(self.states.values()):
            color = colors[idx % len(colors)]
            # Generate edges
            for t in state.transitions:
                # Label
                if t.is_conditional_prob():
                    label = 'Probability = ' + str(t.prob)
                elif t.is_conditional_if():
                    label = 'If: ' + str(t._if)
                elif len(state.transitions) == 1:
                    label = 'Always'
                else: 
                    label = 'Otherwise'
                edge = pydot.Edge(state.id, t.dest)
                edge.set_label(label)
                edge.set_color(color)
                edge.set_fontcolor(color)
                dot_graph.add_edge(edge)
            # Generate node
            label = state.label + (
                '\n' + ('+' if state.duration > 0 else '') + str(state.duration) + ' timesteps'
                if state.duration > 0 else ''
            )
            shape = 'cds' if state.type == 'start' else 'doublecircle' if state.type == 'end' else 'ellipse'
            node = pydot.Node(state.id, label=label)
            node.set_shape(shape)
            node.set_color(color)
            dot_graph.add_node(node)
        
        if path_to_file:
            dot_graph.write_png(path_to_file + '.png')
        else:
            # Source: https://stackoverflow.com/questions/4596962/display-graph-without-saving-using-pydot
            png_str = dot_graph.create_png()
            sio = io.BytesIO()
            sio.write(png_str)
            sio.seek(0)
            img = mpimg.imread(sio)
            plt.figure(figsize = figsize)
            plt.axis('off')
            plt.imshow(img, aspect='equal')

def sort_patient_by_preference(patients: list[Patient], 
                               property_to_sort_by: str = None, 
                               is_ascending: bool = True):
    """Sorts 'patients' in place by whatever patient property is specified in 'property_to_sort_by'
        NOTE: This is done in place for speed
    """
    if property_to_sort_by:
        patients.sort(key=lambda x : x.properties[property_to_sort_by], reverse = not is_ascending)

def create_patients_for_simulation(simulation: Simulation, 
                                   patients: list[Patient],
                                   func_match_patient_to_property_column: Callable = None,
                                   random_seed: int = 0) -> list[Patient]:
    patients = pickle.loads(pickle.dumps(patients))
    patients = sorted(patients, key = lambda x: x.id)
    properties = [ (id, v) for id, v in simulation.variables.items() if v.get('type', 'scalar') == 'property' ]
    # CSV for file-defined properties
    path_to_properties = simulation.metadata.get('path_to_properties', None)
    properties_col_for_patient_id = simulation.metadata.get('properties_col_for_patient_id', None)
    if path_to_properties:
        # Read CSV containing patient properties
        _df = pd.read_csv(path_to_properties)
        if properties_col_for_patient_id:
            # Check that column corresponding to the Patient ID actually exists in CSV
            if properties_col_for_patient_id not in _df.columns:
                print(f"ERROR - The value for `properties_col_for_patient_id` ({properties_col_for_patient_id}) must be a column name in the file {path_to_properties}")
                return None
            # Sort patients by ID
            _df = _df.sort_values(properties_col_for_patient_id)
        # If we want to randomly sample patients from the CSV (instead of using their ID), 
        # then do this sampling deterministically by tracking `map_pid_to_random_df_idx`
        np.random.seed(random_seed)
        random_idxs = np.random.randint(0, _df.shape[0], size = len(patients))
        map_pid_to_random_df_idx = { p.id: random_idxs[idx] for idx, p in enumerate(patients) }
    #
    # Add properties to each Patient
    np.random.seed(random_seed)
    for (v_id, v) in properties:
        if 'value' in v:
            # Set to constant
            for p in patients:
                p.properties[v_id] = v['value']
        elif 'column' in v:
            # Load from 'path_to_properties' file
            if not path_to_properties:
                print(f"ERROR - If you specify a 'column' variable, you need to specify a 'path_to_properties' value in the 'metadata' section")
                return None
            if v['column'] not in _df:
                print(f"ERROR - 'column' {v['column']} is not contained in the file pointed to by 'path_to_properties'")
                return None
            if properties_col_for_patient_id:
                ## NOTE: This may seem like an unnecessary special case of the functionality offered by `func_match_patient_to_property_column`,
                ## But its a necessary performance optimization that actually helps speed up the program a lot
                sorted_properties = _df[v['column']].values
                for p_idx, p in enumerate(patients):
                    p.properties[v_id] = sorted_properties[p_idx]
            else:
                if func_match_patient_to_property_column is None and properties_col_for_patient_id is None:
                    print(f"ERROR - You need to either specify a `func_match_patient_to_property_column` when calling this function, or the `properties_col_for_patient_id` metadata property in your YAML file (otherwise we have no idea how to match patients to rows in the file)")
                    return None
                for p in patients:
                    p.properties[v_id] = func_match_patient_to_property_column(p.id, map_pid_to_random_df_idx[p.id], _df, v['column'])
        elif 'distribution' in v:
            # Distribution
            if v['distribution'] == 'bernoulli':
                assert 'mean' in v, f"ERROR - Bernoulli variable '{v_id}' missing 'mean' property"
                values = np.random.binomial(1, v['mean'], size = len(patients))
            elif v['distribution'] == 'exponential':
                assert 'mean' in v, f"ERROR - Exponential variable '{v_id}' missing 'mean' property"
                values = np.random.exponential(v['mean'], size = len(patients))
            elif v['distribution'] == 'binomial':
                assert 'mean' in v and 'n' in v, f"ERROR - Binomial variable '{v_id}' missing 'mean' or 'n' property"
                values = np.random.binomial(v['n'], v['mean'], size = len(patients))
            elif v['distribution'] == 'normal':
                assert 'mean' in v and 'std' in v, f"ERROR - Normal variable '{v_id}' missing 'mean' or 'std' property"
                values = np.random.normal(v['mean'], v['std'], size = len(patients))
            elif v['distribution'] == 'poisson':
                assert 'mean' in v, f"ERROR - Poisson variable '{v_id}' missing 'mean' property"
                values = np.random.poisson(v['mean'], size = len(patients))
            elif v['distribution'] == 'uniform':
                assert 'start' in v and 'end' in v, f"ERROR - Uniform variable '{v_id}' missing 'start' or 'end' property"
                values = np.random.uniform(v['start'], v['end'], size = len(patients))
            else:
                print(f"ERROR - Unrecognized 'distribution' in variable '{v_id}'")
                return None
            for idx, p in enumerate(patients):
                p.properties[v_id] = values[idx]
        else:
            print(f"ERROR - Unrecognized properties for variable '{v_id}'")
            return None
    return patients

def get_unit_utility_baselines(patients: list[Patient],
                               utilities: dict,
                               y_true_column_name: str = 'ground_truth') -> dict[float]:
    """Get average utility per patient under baseline settings (Treat All, Treat None, Treat Perfect)

    Args:
        patients (list[Patient]): List of Patient objects
        utilities (dict): Utility values for TP/FP/FN/TN
        y_true_column_name (str, optional): Name of patient property that corresponds to the ground truth label. Defaults to 'ground_truth'.
            i.e. For a given patient `p`, the function will use `p.properties[y_true_column_name]` as the ground truth label for that patient

    Returns:
        dict[float]: Three keys (all|none|perfect), each corresponding to their average utility per patient
    """    
    positives = len([ p for p in patients if p.properties[y_true_column_name] == 1 ])
    negatives = len([ p for p in patients if p.properties[y_true_column_name] == 0 ])
    """Treat all (i.e. TP -> TP; FP -> FP, FN -> TP, TN -> FP)
    """
    treat_all = 0
    treat_all += positives * utilities['tp']
    treat_all += negatives * utilities['fp']
    """Treat none (i.e. TP -> FN; FP -> TN, FN -> FN, TN -> TN)
    """
    treat_none = 0
    treat_none += positives * utilities['fn']
    treat_none += negatives * utilities['tn']
    """Treat perfect (i.e. TP -> TP; FP -> TN, FN -> TP, TN -> TN)
    """
    treat_perfect = 0
    treat_perfect += positives * utilities['tp']
    treat_perfect += negatives * utilities['tn']
    return {
        'all' : treat_all / len(patients),
        'none' : treat_none / len(patients),
        'perfect' : treat_perfect / len(patients),
    }

def log_patients(simulation: Simulation, patients: list[Patient]):
    for p in patients:
        print(f"{p.id} (t_0 = {p.start_timestep})")
        print('\t', p.properties)
        print('\t', p.print_state_history())
        print('\t', p.get_sum_utilities(simulation))

if __name__ == "__main__":
    pass