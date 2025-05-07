import aplusml
import numpy as np
import pandas as pd
from typing import List, Dict
import collections

PATH_TO_CONFIG_YAML: str = "config.yaml"
PATH_TO_PATIENT_PROPERTIES: str = "patient_properties.csv"

# Create simulation. Loads workflow + simulation parameters from YAML file, patient properties from CSV file.
# If `path_to_patient_properties` is None, then the default `metadata.path_to_properties` key in the YAML file is used. Here, we explicitly override it.
simulation: aplusml.Simulation = aplusml.Simulation.create_from_yaml(PATH_TO_CONFIG_YAML, PATH_TO_PATIENT_PROPERTIES)

# Set random seed for reproducibility
np.random.seed(0)

# Create Patients from CSV
df = pd.read_csv(PATH_TO_PATIENT_PROPERTIES)
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
patients: List[aplusml.Patient] = simulation.create_patients_for_simulation(patients, random_seed = 0)

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

# Print patient histories
for p in patients:
    print('Patient', p.id, 'History:', p.repr_state_history())