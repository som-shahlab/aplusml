import aplusml
from aplusml.config import Config, ConfigMetadata, ConfigVariable, ConfigState, ConfigTransition, ConfigUtility
import numpy as np
import pandas as pd
from typing import List, Dict
import collections

# Create config
config = Config(
    metadata = ConfigMetadata(
        name = 'Hello World Workflow',
        path_to_properties = None,
    ),
    variables = {
        'patient_property_1': ConfigVariable(
            type = 'property',
            value = None, # Note: Will get overwritten later
        ),
        'patient_property_2': ConfigVariable(
            type = 'property',
            value = None, # Note: Will get overwritten later
        ),
        'some_constant': ConfigVariable(
            type = 'scalar',
            value = 10,
        ),
    },
    states = {
        'start': ConfigState(
            type = 'start',
            label = 'Start',
            transitions = [
                ConfigTransition(
                    dest = 'state_2',
                ),
            ],
        ),
        'state_2': ConfigState(
            type = 'intermediate',
            label = 'Switch',
            transitions = [
                ConfigTransition(dest = 'state_3', if_ = 'patient_property_1 > patient_property_2'),
                ConfigTransition(dest = 'state_4', if_ = 'patient_property_1 <= patient_property_2'),
            ],
        ),
        'state_3': ConfigState(
            type = 'end',
            label = 'Good End',
            utilities = [
                ConfigUtility(
                    value = 100,
                    unit = 'USD',
                ),
            ],
        ),
        'state_4': ConfigState(
            type = 'end',
            label = 'Bad End',
            utilities = [
                ConfigUtility(
                    value = 0,
                    unit = 'USD',
                ),
            ],
        ),
    },
)

# Create simulation.
simulation: aplusml.Simulation = aplusml.Simulation.create_from_config(config)

# Set random seed for reproducibility
np.random.seed(0)

# Create Patients from CSV
df = pd.DataFrame(
    data = {
        'patient_id': [1, 2, 3, 4, 5, 6],
        'patient_property_1': [1, 1, 6, 6, 2, 2],
        'patient_property_2': [2, 5, 5, 1, 1, 8],
        'start_timestep': [0, 1, 2, 0, 1, 0],
    }
)
patients: List[aplusml.Patient] = [
    aplusml.Patient(
        id = row['patient_id'],
        start_timestep=row['start_timestep'],
        properties = {
            'patient_property_1': row['patient_property_1'],
            'patient_property_2': row['patient_property_2'],
        }
    )
    for _, row in df.iterrows()
]

# Initialize patients for simulation. 
# NOTE: Do not overwrite existing properties b/c we just manually set them above!
patients: List[aplusml.Patient] = simulation.create_patients_for_simulation(patients, 
                                                                            is_overwrite_existing_properties=False, # ! IMPORTANT
                                                                            random_seed = 0)

# Run simulation
patients = simulation.run(patients)

# Sum up the utilities achieved across all patient histories
sum_utilities: Dict[str, float] = collections.defaultdict(float)
for p in patients:
        _u: dict = p.get_sum_utilities(simulation)
        for key, val in _u.items():
                sum_utilities[key] += val

# Print workflow diagram
simulation.draw_workflow_diagram('workflow.png', is_display=False)

# Print results
print('Utilities:', sum_utilities)
# > Utilities: {'USD': 300.0}

# Print patient histories
for p in patients:
        print('Patient', p.id, 'History:', p.repr_state_history())
# > Patient 1 History: start > state_2 > state_4
# > Patient 2 History: start > state_2 > state_3
# > Patient 3 History: start > state_2 > state_4
# > Patient 4 History: start > state_2 > state_4
# > Patient 5 History: start > state_2 > state_3
# > Patient 6 History: start > state_2 > state_3