import aplusml
from aplusml.parse import parse_yaml_into_config

################################################
# Goal: Test YAML => config parsing
################################################
PATH_TO_YAML = 'test7.yaml'

# Parse simulation
config = parse_yaml_into_config(PATH_TO_YAML)

# Metadata
assert config.metadata.name == "Test 7"
assert config.metadata.path_to_properties == None
assert config.metadata.properties_col_for_patient_id == None
assert config.metadata.patient_sort_preference_property == None
# Variables
assert config.variables['property'].type == 'property'
assert config.variables['property'].column == 'los'
assert config.variables['utility_val1'].value == -1
assert config.variables['utility_val2'].value == -2
# States
assert config.states['no_acp'].type == 'end'
assert config.states['no_acp'].label == 'Don\'t deliver ACP'
assert config.states['no_acp'].duration == 2
assert config.states['no_acp'].utilities[0].value == 8
assert config.states['no_acp'].utilities[0].unit == 'B'
assert config.states['no_acp'].utilities[1].value == 7
assert config.states['no_acp'].utilities[1].unit == 'A'
assert config.states['no_acp'].utilities[2].value == 12
assert config.states['no_acp'].utilities[2].if_ == True

assert config.is_valid(), "Config is not valid"

simulation_config = aplusml.Simulation.create_from_config(config)
simulation_yaml = aplusml.Simulation.create_from_yaml(PATH_TO_YAML)

assert simulation_config.metadata == simulation_yaml.metadata
assert simulation_config.variables == simulation_yaml.variables
assert simulation_config.states == simulation_yaml.states

print("SUCCESSFULLY PASSED TEST 7")
