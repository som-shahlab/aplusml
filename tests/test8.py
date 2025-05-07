import aplusml
from aplusml.parse import parse_yaml_into_config

################################################
# Goal: Test YAML == Config == Simulation
################################################

# Parse simulation
for path in [ 'test1.yaml', 'test2.yaml', 'test3.yaml', 'test4.yaml', 'test5.yaml', 'test6.yaml', 'test7.yaml' ]:
    print(f"Testing {path}")
    config = parse_yaml_into_config(path)
    simulation_config = aplusml.Simulation.create_from_config(config)
    simulation_yaml = aplusml.Simulation.create_from_yaml(path)
    assert simulation_config.metadata == simulation_yaml.metadata, f"Error with {path}\n\nconfig.metadata\n{simulation_config.metadata}\n\nyaml.metadata\n{simulation_yaml.metadata}"
    assert simulation_config.variables == simulation_yaml.variables, f"Error with {path}\n\nconfig.variables\n{simulation_config.variables}\n\nyaml.variables\n{simulation_yaml.variables}"
    assert simulation_config.states == simulation_yaml.states, f"Error with {path}\n\nconfig.states\n{simulation_config.states}\n\nyaml.states\n{simulation_yaml.states}"

print("SUCCESSFULLY PASSED TEST 8")
