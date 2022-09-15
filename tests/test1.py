import sys
sys.path.append("..")
import parse

################################################
# Goal: Test 'parse_yaml'
################################################
PATH_TO_YAML = 'test1.yaml'

# Parse simulation
yaml = parse.load_yaml(PATH_TO_YAML)
simulation = parse.create_simulation_from_yaml(yaml)

# Metadata
assert simulation.metadata['name'] == 'Test 1'
assert simulation.metadata['path_to_functions'] == 'functions.py'
assert simulation.metadata['path_to_properties'] == 'properties.csv'

# Variables
## Scalar - float
v = simulation.variables['float']
assert v['type'] == 'scalar' \
        and isinstance(v['value'], float) \
        and v['value'] == 0.5
## Scalar - int
v = simulation.variables['int']
assert v['type'] == 'scalar' \
        and isinstance(v['value'], int) \
        and v['value'] == 5
## Scalar - str
v = simulation.variables['str']
assert v['type'] == 'scalar' \
        and isinstance(v['value'], str) \
        and v['value'] == 'example string'
## Scalar - bool
v = simulation.variables['bool_True']
assert v['type'] == 'scalar' \
        and isinstance(v['value'], bool) \
        and v['value'] == True
v = simulation.variables['bool_False']
assert v['type'] == 'scalar' \
        and isinstance(v['value'], bool) \
        and v['value'] == False
## Scalar - dict
v = simulation.variables['dict']
assert v['type'] == 'scalar' \
        and isinstance(v['value'], dict) \
        and v['value']['a'] == 1 \
        and v['value']['b'] == 2 \
        and v['value']['c'] == 3 \
        and v['value']['d']['dd'] == 4 \
        and v['value']['d']['ee'][0] == 1 \
        and v['value']['d']['ee'][1] == 2
## Scalar - list
v = simulation.variables['list']
assert v['type'] == 'scalar' \
        and isinstance(v['value'], list) \
        and v['value'][0] == 10 \
        and v['value'][1] == 9 \
        and v['value'][2] == 8 \
        and v['value'][3][0] == 7 \
        and v['value'][3][1] == 6
## Scalar - set
v = simulation.variables['set']
assert v['type'] == 'scalar' \
        and isinstance(v['value'], set) \
        and 1 in v['value'] \
        and 3 in v['value'] \
        and 5 in v['value'] \
        and 7 in v['value']
## Scalar - tuple
# v = simulation.variables['tuple']
# assert v['type'] == 'scalar' \
#         and isinstance(v['value'], tuple) \
#         and v['value'][0] == 5 \
#         and v['value'][1] == 4 \
#         and v['value'][2] == 3
## Resource
v = simulation.variables['capacity']
assert (v['type'] == 'resource' 
        and v['init_amount'] == 0 
        and v['max_amount'] == 10 
        and v['refill_amount'] == 1 
        and v['refill_duration'] == 1)
## Property (dist)
v = simulation.variables['property_dist']
assert (v['type'] == 'property'
        and v['distribution'] == 'normal'
        and v['mean'] == 1
        and v['std'] == 1)
## Property (file)
v = simulation.variables['property_file']
assert (v['type'] == 'property'
        and v['column'] == 'y_hat')
## Simulation
v = simulation.variables['time_left_in_sim']
assert (v['type'] == 'simulation')

# States
s = simulation.states['start']
assert (s.label == 'Patient Admit'
        and s.type == 'start'
        and s.duration == 1
        and s.utilities[0].value == -1)
assert (s.transitions[0].dest == 'send_nurse'
        and s.transitions[0].label == 'Send Nurse')
s = simulation.states['send_nurse']
assert (s.transitions[0].dest == 'no_acp'
        and s.transitions[0]._if == 'nurse_capacity < 1'
        and s.transitions[1].dest == 'complete_acp')
s = simulation.states['no_acp']
assert (s.type == 'end'
        and s.utilities[0].value == -10
        and s.utilities[0]._if == False
        and s.utilities[1].value == -3
        and s.utilities[1]._if == True)
s = simulation.states['complete_acp']
assert (s.type == 'end'
        and s.utilities[0].value == -30
        and s.utilities[0]._if == True
        and s.utilities[1].value == -15
        and s.utilities[1]._if == False)
        
print("SUCCESSFULLY PASSED TEST 1")