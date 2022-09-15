import collections
import sys
from utils import check_history, check_probabilistic_outcomes_are_within_margin_of_error
sys.path.append("..")
import sim
import parse
import pickle

################################################
# Goal: Test 'run' function of Simulation object
# on setup with conditionals (no timesteps or utilities)
################################################
PATH_TO_YAML = 'test5.yaml'

# Parse simulation
yaml = parse.load_yaml(PATH_TO_YAML)
simulation = parse.create_simulation_from_yaml(yaml)

all_patients = [
    sim.Patient(0,0, properties={
        'property' : 0,
        'property2' : 0,
        'total_duration_in_sim' : 4,
    }),
    sim.Patient(1,1, properties={
        'property' : 0,
        'property2' : 1,
        'total_duration_in_sim' : 4,
    }),
    sim.Patient(2,2, properties={
        'property' : 1,
        'property2' : 2,
        'total_duration_in_sim' : 4,
    }),
]

#
# Test 'run()'
#
patient_runs = []
for i in range(2000):
    simulation.run(all_patients[:1], random_seed=i)
    patient_runs.append(pickle.loads(pickle.dumps(all_patients[0])))
## Test 'prob' conditional
counts = collections.defaultdict(int)
for p in patient_runs:
    outcome = p.history[1].state_id
    counts[outcome] += 1
probabilities = {
    'bin1' : 0.2,
    'bin2' : 0.3,
    'bin3' : 0.1,
    'bin4' : 0.4,
}
check_probabilistic_outcomes_are_within_margin_of_error(counts,
                                                        probabilities,
                                                        0.05)
## Test 'if' TRUE
for p in patient_runs:
    for idx, h in enumerate(p.history):
        if h.state_id == 'test_if_true':
            assert p.history[idx + 1].state_id == 'test_capacity'
## Test 'if' + 'property' (multiple vars)
for i in [ .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, ]:
    patients = [
        sim.Patient(0,0, properties={
            'property' : i,
            'property2' : i,
            'total_duration_in_sim' : 4,
        }),
    ]
    simulation.run(patients, random_seed=0)
    if i < 0.5:
        assert patients[0].history[5].state_id == 'property_threshold1'
    else:
        assert patients[0].history[5].state_id == 'property_threshold2'
## Test 'if' + 'simulation' (multiple vars)
for los in range(0, 10):
    patients = [
        sim.Patient(0,0, properties={
            'property' : 1,
            'property2' : 1,
            'total_duration_in_sim' : los,
        }),
    ]
    simulation.run(patients, random_seed=0)
    if los > 8 or (los - 1) < 2:
        assert patients[0].history[7].state_id == 'sim1'
    else:
        assert patients[0].history[7].state_id == 'sim2'

## Test 'resource'
simulation.init_run()
assert simulation.variables['capacity']['level'] == simulation.variables['capacity']['init_amount']
simulation.run(all_patients, random_seed=0)
assert simulation.variables['capacity']['level'] == simulation.variables['capacity']['init_amount'] - 2
## Test 'if' + 'resource'
for p in all_patients:
    assert 'no_capacity' not in [ x.state_id for x in p.history ]
## Test 'if' + 'resource' + 'property' (multiple vars)
assert all_patients[0].history[-4].state_id == 's1'
assert all_patients[1].history[-4].state_id == 's1'
assert all_patients[2].history[-4].state_id == 's2'

## full run
simulation.run(all_patients, random_seed=0)
assert simulation.current_timestep == 3
## patient started at t = 0
check_history(simulation, 
              all_patients[0], 
              current_state=None,
              history=['start', 
                       'bin4', 
                       'test_if_true',
                       'test_capacity',
                       'test_property',
                       'property_threshold1',
                       'sim_time',
                       'sim2',
                       'outpatient_rescue',
                       's1',
                       'if_and_prob',
                       'a2',
                       'complete_acp',
                       ])
## patient started at t = 1
check_history(simulation, 
              all_patients[1], 
              current_state=None,
              history=['start', 
                       'bin4', 
                       'test_if_true',
                       'test_capacity',
                       'test_property',
                       'property_threshold1',
                       'sim_time',
                       'sim2',
                       'outpatient_rescue',
                       's1',
                       'if_and_prob',
                       'a3',
                       'complete_acp',
                       ])
## patient started at t = 2
check_history(simulation, 
              all_patients[2], 
              current_state=None,
              history=['start', 
                       'bin2', 
                       'test_if_true',
                       'test_capacity',
                       'test_property',
                       'property_threshold2',
                       'sim_time',
                       'sim2',
                       'outpatient_rescue',
                       's2',
                       'if_and_prob',
                       'a4',
                       'complete_acp',
                       ])
# randomness
for i in range(2000):
    simulation.run(all_patients, random_seed=0)
    assert all_patients[0].history[1].state_id == 'bin4'
    assert all_patients[1].history[1].state_id == 'bin4'
    assert all_patients[2].history[1].state_id == 'bin2'

print("SUCCESSFULLY PASSED TEST 5")