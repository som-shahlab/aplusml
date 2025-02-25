"""
Models for the APLUS simulation framework.
"""
import collections
from types import CodeType
from typing import Union, Dict, List
import ast

class Utility(object):
    """
    A utility is a value that is associated with being in a state or undergoing a transition.
    """

    def __init__(self,
                 value: str,
                 unit: str = '',
                 _if: str = None):
        """
        Args:
            value (str): The value of the utility. Example: '100000'
            unit (str, optional): The unit of the utility. Defaults to ''. Example: 'USD', 'days', 'kg', 'cm', etc.
            _if (str, optional): The condition for the utility, specified as a Python expression. Defaults to None. Example: 'y_hat > 0.5'
        """
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
    
    def is_conditional_if(self) -> bool:
        """
        Returns True if the utility is conditional on a Python expression.
        """
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
    """
    A transition between states.
    """

    def __init__(self, 
                 dest: str,
                 label: str,
                 duration: int,
                 utilities: List[Utility],
                 resource_deltas: Dict[str, float],
                 _if: Union[str, bool] = None,
                 prob: Union[str, float] = None):
        """
        Args:
            dest (str): The destination state.
            label (str): The label of the transition.
            duration (int): The duration of the transition. Example: 1
            utilities (List[Utility]): The utilities of the transition. Example: [Utility(value='100000', unit='USD')]
            resource_deltas (Dict[str, float]): The resource deltas of the transition. Example: {'MRI': -1}
            _if (Union[str, bool], optional): The condition for the transition, specified as a Python expression. Defaults to None. Example: 'y_hat > 0.5'
            prob (Union[str, float], optional): The probability of the transition. Defaults to None. Example: 0.5
        """
        self.dest: str = dest
        self.label: str = label
        self.duration: int = duration
        self.utilities: List[Utility] = utilities
        self.resource_deltas: Dict[str, float] = resource_deltas
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

    def is_conditional_prob(self) -> bool:
        """
        Returns True if the transition is probabilistic.
        """
        return self.prob is not None

    def is_conditional_if(self) -> bool:
        """
        Returns True if the transition is conditional on a Python expression.
        """
        return self._if is not None
    
    def get_variables_in_conditional(self) -> List[str]:
        """
        Returns a list of variables involved in the conditional expression.
        """
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
        parsed_variable_ids: List[str] = []
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
    """
    A state in the workflow.
    """

    def __init__(self, 
                 id: str,
                 label: str,
                 type: str,
                 duration: int,
                 utilities: List[Utility],
                 transitions: List[Transition],
                 resource_deltas: Dict[str, float]):
        """
        Args:
            id (str): The ID of the state.
            label (str): The label of the state.
            type (str): The type of the state. Must be one of: 'start', 'intermediate', 'end'
            duration (int): The duration of the state. Example: 1
            utilities (List[Utility]): The utilities of the state. Example: [Utility(value='100000', unit='USD')]
            transitions (List[Transition]): The transitions of the state. Example: [Transition(dest='state2', label='To State 2', duration=1, utilities=[Utility(value='100000', unit='USD')], resource_deltas={'MRI': -1})]
            resource_deltas (Dict[str, float]): The resource deltas of the state. Example: {'MRI': -1}
        """
        self.id: str = id
        self.label: str = label
        self.type: str = type
        self.duration: int = duration
        self.utilities: List[Utility] = utilities
        self.transitions: List[Transition] = transitions
        self.resource_deltas:  Dict[str, float] = resource_deltas

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
    """
    The history of a patient's states and transitions.
    """

    def __init__(self, 
                 current_timestep: int,
                 state_id: str,
                 transition_idx: int, # Transition == state.transitions[idx]
                 state_utility_idxs: List[int], # Utilities == state.utilities[idxs]
                 transition_utility_idxs: List[int], # Utilities == state.transitions[idx].utilities[idxs]
                 state_utility_vals: List[float], # Evaluated Utility Values == evaluate_utility_value(state.utilities[idxs].value)
                 transition_utility_vals: List[float], # EvaluatedUtility Values == evaluate_utility_value(state.transitions[idx].utilities[idxs].value)
                 sim_variables: Dict):
        """
        Args: 
            current_timestep (int): The current timestep. Example: 1
            state_id (str): The ID of the current state. Example: 'state1'
            transition_idx (int): The index of the current transition. Example: 0
            state_utility_idxs (List[int]): The indices of the utilities of the current state. Example: [0]
            transition_utility_idxs (List[int]): The indices of the utilities of the current transition. Example: [0]
            state_utility_vals (List[float]): The evaluated utility values of the current state. Example: [100000]
            transition_utility_vals (List[float]): The evaluated utility values of the current transition. Example: [100000]
            sim_variables (Dict): The variables of the simulation. Example: {'y_hat': 0.5}
        """
        self.current_timestep: int = current_timestep
        self.state_id: str = state_id
        self.transition_idx: Union[int, None] = transition_idx
        self.state_utility_idxs: List[int] = state_utility_idxs
        self.transition_utility_idxs: List[int] = transition_utility_idxs
        self.state_utility_vals: List[float] = state_utility_vals
        self.transition_utility_vals: List[float] = transition_utility_vals
        self.sim_variables: Dict = sim_variables

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
    """
    A patient in the simulation.
    """

    def __init__(self, 
                 id: str, 
                 start_timestep: int,
                 properties: dict = None):
        """
        Args:
            id (str): The ID of the patient. Example: 'patient1'
            start_timestep (int): The start timestep of the patient. Example: 1
            properties (dict, optional): The properties of the patient. Example: {'y_hat': 0.5}
        """
        self.id: str = id
        self.start_timestep: int = int(start_timestep) # Start time for this patient (i.e. admitted date)
        self.properties: dict = properties if properties is not None else {} # Patient specific properties, i.e. "y_hat" or "y" or "los"
        self.history: List[History]= [] # Track history of (state, transition, utility)
        self.current_state: str = None # ID of current state
    
    def print_state_history(self, is_show_timesteps: bool = False):
        if is_show_timesteps:
            return " > ".join([ f"({h.current_timestep}) {h.state_id}" for h in self.history])
        else:
            return " > ".join([ h.state_id for h in self.history])

    def get_sum_utilities(self, simulation: 'aplusml.sim.Simulation') -> Dict[str, float]:
        """
        Returns a dictionary of the sum of the utilities of the patient's history.

        Args:
            simulation (Simulation): The simulation.

        Returns:
            Dict[str, float]: A dictionary of the sum of the utilities of the patient's history. Example: {'USD': 100000}
        """
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