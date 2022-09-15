import sys
sys.path.append("..")
import sim
import parse

################################################
# Goal: Test 'run' function of Simulation object
# on basic setup with utilities
# start -> send_nurse -> no_acp
################################################
PATH_TO_YAML = 'test6.yaml'

# Parse simulation
yaml = parse.load_yaml(PATH_TO_YAML)
simulation = parse.create_simulation_from_yaml(yaml)

patients = [
    sim.Patient(0,0, properties={
        'property' : 0,
    }),
    sim.Patient(1,1, properties={
        'property' : 1,
    }),
    sim.Patient(2,2, properties={
        'property' : 2,
    }),
]

def check_state_utility_match(simulation: sim.Simulation, history: sim.History, utility_idx: int, value: float, unit: str = ''):
    utility = simulation.states[history.state_id].utilities[history.state_utility_idxs[utility_idx]]
    assert utility.value == value and utility.unit == unit

def check_transition_utility_match(simulation: sim.Simulation, history: sim.History, utility_idx: int, value: float, unit: str = ''):
    state = simulation.states[history.state_id]
    utility = state.transitions[history.transition_idx].utilities[history.transition_utility_idxs[utility_idx]]
    assert utility.value == value and utility.unit == unit

#
# Test state utilities
#
utility_val1 = -1
utility_val2 = -2
simulation.init_run()
simulation.run(patients)
# patient starting at t = 0
## STATE = 'start'
check_state_utility_match(simulation, patients[0].history[0], 0, utility_val1, '')
check_transition_utility_match(simulation, patients[0].history[0], 0, 11, 'B')
check_transition_utility_match(simulation, patients[0].history[0], 1, 3, '')
assert len(patients[0].history[0].state_utility_idxs) == 1
assert len(patients[0].history[0].transition_utility_idxs) == 2
## STATE = 'send_nurse'
check_state_utility_match(simulation, patients[0].history[1], 0, 5, 'B')
check_state_utility_match(simulation, patients[0].history[1], 1, utility_val1 + utility_val2, 'B')
check_state_utility_match(simulation, patients[0].history[1], 2, 9, 'C')
check_transition_utility_match(simulation, patients[0].history[1], 0, 6, '')
assert len(patients[0].history[1].state_utility_idxs) == 3
assert len(patients[0].history[1].transition_utility_idxs) == 1
## STATE = 'no_acp'
check_state_utility_match(simulation, patients[0].history[2], 0, 8, 'B')
check_state_utility_match(simulation, patients[0].history[2], 1, 7, 'A')
check_state_utility_match(simulation, patients[0].history[2], 2, 12, '')
assert len(patients[0].history[2].state_utility_idxs) == 3
assert len(patients[0].history[2].transition_utility_idxs) == 0

# patient starting at t = 1
## STATE = 'start'
check_state_utility_match(simulation, patients[1].history[0], 0, utility_val1, '')
check_transition_utility_match(simulation, patients[1].history[0], 0, 2, 'A')
check_transition_utility_match(simulation, patients[1].history[0], 1, 11, 'B')
check_transition_utility_match(simulation, patients[1].history[0], 2, 3, '')
assert len(patients[1].history[0].state_utility_idxs) == 1
assert len(patients[1].history[0].transition_utility_idxs) == 3
## STATE = 'send_nurse'
check_state_utility_match(simulation, patients[1].history[1], 0, 5, 'B')
check_state_utility_match(simulation, patients[1].history[1], 1, utility_val1 + utility_val2, 'B')
check_transition_utility_match(simulation, patients[1].history[1], 0, 6, '')
assert len(patients[1].history[1].state_utility_idxs) == 2
assert len(patients[1].history[1].transition_utility_idxs) == 1
## STATE = 'no_acp'
check_state_utility_match(simulation, patients[1].history[2], 0, 8, 'B')
check_state_utility_match(simulation, patients[1].history[2], 1, 7, 'A')
check_state_utility_match(simulation, patients[1].history[2], 2, 12, '')
assert len(patients[1].history[2].state_utility_idxs) == 3
assert len(patients[1].history[2].transition_utility_idxs) == 0

# patient starting at t = 2
## STATE = 'start'
check_state_utility_match(simulation, patients[2].history[0], 0, utility_val1, '')
check_transition_utility_match(simulation, patients[2].history[0], 0, 2, 'A')
check_transition_utility_match(simulation, patients[2].history[0], 1, 11, 'B')
check_transition_utility_match(simulation, patients[2].history[0], 2, 3, '')
assert len(patients[2].history[0].state_utility_idxs) == 1
assert len(patients[2].history[0].transition_utility_idxs) == 3
## STATE = 'send_nurse'
check_state_utility_match(simulation, patients[2].history[1], 0, 5, 'B')
check_state_utility_match(simulation, patients[2].history[1], 1, utility_val1 + utility_val2, 'B')
check_state_utility_match(simulation, patients[2].history[1], 2, 9, 'C')
check_transition_utility_match(simulation, patients[2].history[1], 0, 6, '')
assert len(patients[2].history[1].state_utility_idxs) == 3
assert len(patients[2].history[1].transition_utility_idxs) == 1
## STATE = 'no_acp'
check_state_utility_match(simulation, patients[2].history[2], 0, 8, 'B')
check_state_utility_match(simulation, patients[2].history[2], 1, 7, 'A')
check_state_utility_match(simulation, patients[2].history[2], 2, 12, '')
assert len(patients[2].history[2].state_utility_idxs) == 3
assert len(patients[2].history[2].transition_utility_idxs) == 0

print("SUCCESSFULLY PASSED TEST 6")
