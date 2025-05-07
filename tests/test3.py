from utils import check_history
import aplusml

################################################
# Goal: Test 'run' function of Simulation object
# on basic 3 state setup (no conditionals, no timesteps, no utilities):
# start -> send_nurse -> no_acp
################################################
PATH_TO_YAML = 'test3.yaml'

# Parse simulation
simulation = aplusml.Simulation.create_from_yaml(PATH_TO_YAML)

patients = [
    aplusml.Patient(0,0, properties={
    }),
    aplusml.Patient(1,0, properties={
    }),
    aplusml.Patient(2,1, properties={
    }),
]

#
# Test 'run()'
#
## 'max_timestep' break condition
patients = simulation.run(patients, 1)
assert simulation.current_timestep == 0
## first patient started at t = 0
check_history(simulation,
              patients[0], 
              current_state=None,
              history=['start', 'send_nurse', 'no_acp'])
## second patient started at t = 0
check_history(simulation,
              patients[1], 
              current_state=None,
              history=['start', 'send_nurse', 'no_acp'])
## patient started at t = 1 won't make any progress through states
check_history(simulation,
              patients[2], 
              current_state='start',
              history=[])



## 'max_timestep' break condition
patients = simulation.run(patients, 2)
assert simulation.current_timestep == 1
## first patient started at t = 0
check_history(simulation,
              patients[0], 
              current_state=None,
              history=['start', 'send_nurse', 'no_acp'])
## second patient started at t = 0
check_history(simulation,
              patients[1], 
              current_state=None,
              history=['start', 'send_nurse', 'no_acp'])
## patient started at t = 1
check_history(simulation,
              patients[2], 
              current_state=None,
              history=['start', 'send_nurse', 'no_acp'])

print("SUCCESSFULLY PASSED TEST 3")
