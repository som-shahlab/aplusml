import sys
from utils import check_history
sys.path.append("..")
import sim
import parse

################################################
# Goal: Test 'run' function of Simulation object
# on basic 3 state setup with timesteps (no utilities or conditionals):
# start -> send_nurse (+1) -> no_acp (+2)
################################################
PATH_TO_YAML = 'test4.yaml'

# Parse simulation
yaml = parse.load_yaml(PATH_TO_YAML)
simulation = parse.create_simulation_from_yaml(yaml)

patients = [
    sim.Patient(0,0, properties={
    }),
    sim.Patient(1,1, properties={
    }),
    sim.Patient(2,2, properties={
    }),
]
#
# Test 'run()'
#
## 'max_timestep' break condition
simulation.run(patients, 2)
assert simulation.current_timestep == 1
## patient started at t = 0
check_history(simulation,
              patients[0], 
              current_state='no_acp',
              history=['start', 'send_nurse'])
## patient started at t = 1
assert patients[1].current_state == 'send_nurse'
check_history(simulation,
              patients[1], 
              current_state='send_nurse',
              history=['start', ])
## patient started at t = 2
check_history(simulation,
              patients[2], 
              current_state='start',
              history=[])

## 'max_timestep' break condition
simulation.run(patients, 4)
assert simulation.current_timestep == 3
## patient started at t = 0
assert patients[0].current_state == None
check_history(simulation,
              patients[0], 
              current_state=None,
              history=['start', 'send_nurse', 'no_acp'])
## patient started at t = 1
assert patients[1].current_state == 'no_acp'
check_history(simulation,
              patients[1], 
              current_state='no_acp',
              history=['start', 'send_nurse',])
## patient started at t = 2
assert patients[2].current_state == 'no_acp'
check_history(simulation,
              patients[2], 
              current_state='no_acp',
              history=['start', 'send_nurse', ])

## full run
simulation.run(patients)
assert simulation.current_timestep == 5
## patient started at t = 0
check_history(simulation,
              patients[0], 
              current_state=None,
              history=['start', 'send_nurse', 'no_acp'])
## patient started at t = 1
check_history(simulation,
              patients[1], 
              current_state=None,
              history=['start', 'send_nurse', 'no_acp'])
## patient started at t = 2
check_history(simulation,
              patients[2], 
              current_state=None,
              history=['start', 'send_nurse', 'no_acp'])

print("SUCCESSFULLY PASSED TEST 4")