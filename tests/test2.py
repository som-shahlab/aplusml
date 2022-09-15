import sys
sys.path.append("..")
import sim
import parse

################################################
# Goal: Test 'init_*' functions 
# and 'evaluate_variables' 
# of Simulation object
################################################
PATH_TO_YAML = 'test2.yaml'

# Parse simulation
yaml = parse.load_yaml(PATH_TO_YAML)
simulation = parse.create_simulation_from_yaml(yaml)

patients = [
    sim.Patient(0,0, properties={
        'property_file' : 1,
        'property_dist' : 0.3,
        'total_duration_in_sim' : 3,
    }),
    sim.Patient(1,1, properties={
        'property_file' : 0,
        'property_dist' : 0.4,
        'total_duration_in_sim' : 4,
    }),
    sim.Patient(2,1, properties={
        'property_file' : 1,
        'property_dist' : 0.8,
        'total_duration_in_sim' : 2,
    }),
]

simulation.init_run()
assert simulation.current_timestep == 0
for v in simulation.variables.values():
    if v['type'] == 'resource':
        assert v['level'] == v['init_amount']

simulation.init_patients(patients)
for p in patients:
    assert p.current_state == 'start'

#
# Test 'evaluate_variables()'
#
# Variables types: scalar, resource, property
variable_to_value = simulation.evaluate_variables(patients[0])
assert variable_to_value['probability'] == 0.5
assert variable_to_value['capacity'] == 2
assert variable_to_value['property_file'] == 1
assert variable_to_value['property_dist'] == 0.3
assert variable_to_value['total_duration_in_sim'] == 3
assert variable_to_value['time_left_in_sim'] == 3
variable_to_value = simulation.evaluate_variables(patients[1])
assert variable_to_value['probability'] == 0.5
assert variable_to_value['capacity'] == 2
assert variable_to_value['property_file'] == 0
assert variable_to_value['property_dist'] == 0.4
assert variable_to_value['total_duration_in_sim'] == 4
assert variable_to_value['time_left_in_sim'] == 5

# Variables types: simulation
simulation.current_timestep = 2
variable_to_value = simulation.evaluate_variables(patients[0])
assert variable_to_value['time_left_in_sim'] == 1
assert variable_to_value['time_already_in_sim'] == 2
variable_to_value = simulation.evaluate_variables(patients[1])
assert variable_to_value['time_left_in_sim'] == 3
assert variable_to_value['time_already_in_sim'] == 1

print("SUCCESSFULLY PASSED TEST 2")
