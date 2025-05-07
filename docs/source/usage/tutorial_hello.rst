Tutorial (Hello World)
================

.. toctree::
   :maxdepth: 2
   :caption: Tutorial (Hello World)

A minimal "Hello World" using APLUS. Please see `this folder in the repository <https://github.com/som-shahlab/aplusml/tree/main/tutorials/hello_world>`_ for downloadable code.

1. Self-Contained Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The code below shows the simplest way to run APLUS. It is entirely self-contained using Python for configuration and no file dependencies.

.. code-block:: python
  :caption: `script.py`
  :linenos:

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


2. Example with YAML config + Patient Properties CSV file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For more complex configs, you might want to specify them as YAML + CSV files instead of Python objects. Here's an example specifying the same workflow as above.

.. code-block:: yaml
  :caption: `config.yaml`
  :linenos:

   metadata:
      name: "Hello World Workflow"
      path_to_properties: "patient_properties.csv"
      properties_col_for_patient_id: "patient_id"
    
  variables:
    patient_property_1:
      type: "property"
      column: "patient_property_1"
    patient_property_2:
      type: "property"
      column: "patient_property_2"
    some_constant:
      type: "scalar"
      value: 10
    
  states:
    start:
      label: "Start"
      type: "start"
      transitions:
        - dest: "state_2"
    state_2:
      label: "Switch"
      type: "intermediate"
      transitions:
        - dest: "state_3"
          if: patient_property_1 > patient_property_2
        - dest: "state_4"
          if: patient_property_1 <= patient_property_2
    state_3:
      label: "Good End"
      type: "end"
      utilities:
        - value: 100
          unit: "USD"
    state_4:
      label: "Bad End"
      type: "end"
      utilities:
        - value: 0
          unit: "USD"


.. code-block:: csv
  :caption: `patient_properties.csv`

   patient_id,patient_property_1,patient_property_2,start_timestep
   1,1,2,0
   2,1,5,1
   3,6,5,2
   4,6,1,0
   5,2,1,1
   6,2,8,0

.. code-block:: python
  :caption: `script_with_yaml.py`
  :linenos:

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
