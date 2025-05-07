Quick Start
============

.. toctree::
   :maxdepth: 2
   :caption: Quick Start

Get up-and-running with APLUS in <5 minutes with this walkthrough!

1\. Install the APLUS package:

.. code-block:: bash

   pip install aplusml


2\. Create your workflow configuration:

.. code-block:: python
  :caption: `script.py`
  :linenos:

  import aplusml

  # Create Config object. You must specify three things:
  # 1. Metadata -- basic information about the simulation
  # 2. Variables -- variables used in the simulation
  # 3. States -- states in the workflow that your patients will progress through
  config = aplusml.config.Config(
    metadata = aplusml.config.ConfigMetadata(
      name = 'My Simulation',
    ),
    variables = {
      'some_constant': aplusml.config.ConfigVariable(
        type = 'scalar',
        value = -12.03,
      ),
      'some_resource': aplusml.config.ConfigVariable(
        type = 'resource',
        init_amount = 0,
        max_amount = 10,
        refill_amount = 3,
        refill_duration = 1,
      ),
    },
    states = {
      'start' : aplusml.config.ConfigState(
        type = 'start',
        transitions = [
          aplusml.config.ConfigTransition(dest = 'end_1', prob = 0.5),
          aplusml.config.ConfigTransition(dest = 'end_2', prob = 0.5),
        ],
      ),
      'end_1' : aplusml.config.ConfigState(
        type = 'end',
        utilities = {
          'qalys': aplusml.config.ConfigUtility(
            value = 1,
          ),
        },
      ),
      'end_2' : aplusml.config.ConfigState(
        type = 'end',
        utilities = {
          'qalys': aplusml.config.ConfigUtility(
            value = 2,
          ),
        },
      ),
    },
  )


3\. Create a :class:`~aplusml.sim.Simulation` object.

.. code-block:: python
  :caption: `script.py`
  :linenos:
  :lineno-start: 50

  simulation: aplusml.Simulation = aplusml.Simulation.create_from_config(config)

4\. Visualize the workflow via a **graphviz** diagram:

.. code-block:: python
  :caption: `script.py`
  :linenos:
  :lineno-start: 51

  # Draws workflow diagram. This will save the diagram to './output.png' and print it to your terminal.
  simulation.draw_workflow_diagram(figsize=(30,30), path_to_file='./output.png', is_display=True)

5\. Create a list of patients to simulate. We simulate the flow of patients as a Poisson process with the following parameters:

  * :math:`p` patients start our workflow each day, where :math:`p \sim \text{Poisson}(\lambda=35)`
  * :math:`N` total days will be simulated, where :math:`N=500`

.. code-block:: python
  :caption: `script.py`
  :linenos:
  :lineno-start: 53

  import numpy as np

  # Set random seed for reproducibility
  np.random.seed(0)

  # Simulate number of patients per day
  n_admits_per_day = np.random.poisson(lam=35, size=500)

  # Create empty Patient objects (with proper start timesteps according to our Poisson distribution)
  patients: List[aplusml.Patient] = []
  for timestep, n_admits in enumerate(n_admits_per_day):
      for x in range(n_admits):
          patients.append(aplusml.Patient(
              len(patients), # Unique ID
              timestep, # Start timestep
          ))

  # Function which matches a patient to a row in the CSV file.
  func_match_patient_to_property_column = lambda p_id, random_idx, df, col: df.iloc[random_idx][col]

  # Initialize patients for simulation. 
  # This creates a deep copy of each object in the `patients` array using pickle, sorts them by ID, and then initializes their properties
  patients: List[aplusml.Patient] = simulation.create_patients_for_simulation(patients, 
                                                                            func_match_patient_to_property_column,
                                                                            random_seed = 0)

6\. Run the simulation:

.. code-block:: python
  :caption: `script.py`
  :linenos:
  :lineno-start: 78
  
  # Run simulation
  patients = simulation.run(patients)


7\. Calculate the total utility achieved by the workflow by summing all patients' achieved utilities:

.. code-block:: python
  :caption: `script.py`
  :linenos:
  :lineno-start: 80
  
  # Sum up the utilities achieved across all patient histories
  sum_utilities: Dict[str, float] = collections.defaultdict(float)
  for p in patients:
      _u: dict = p.get_sum_utilities(simulation)
      for key, val in _u.items():
          sum_utilities[key] += val

  # Print results
  print(sum_utilities)

Putting it all together, we have the following APLUS script:

.. code-block:: python
  :caption: `script.py`
  :linenos:

  import aplusml
  import numpy as np
  from typing import List, Dict
  import collections

  ############################################################
  # 1. Initialize simulation
  ############################################################
  
  # Create Config object. You must specify three things:
  # 1. Metadata -- basic information about the simulation
  # 2. Variables -- variables used in the simulation
  # 3. States -- states in the workflow that your patients will progress through
  config = aplusml.config.Config(
    metadata = aplusml.config.ConfigMetadata(
      name = 'My Simulation',
    ),
    variables = {
      'some_constant': aplusml.config.ConfigVariable(
        type = 'scalar',
        value = -12.03,
      ),
      'some_resource': aplusml.config.ConfigVariable(
        type = 'resource',
        init_amount = 0,
        max_amount = 10,
        refill_amount = 3,
        refill_duration = 1,
      ),
    },
    states = {
      'start' : aplusml.config.ConfigState(
        type = 'start',
        transitions = [
          aplusml.config.ConfigTransition(dest = 'end_1', prob = 0.5),
          aplusml.config.ConfigTransition(dest = 'end_2', prob = 0.5),
        ],
      ),
      'end_1' : aplusml.config.ConfigState(
          type = 'end',
          utilities = [
              aplusml.config.ConfigUtility(
                  value = .5,
                  unit = 'qaly',
              ),
          ],
      ),
      'end_2' : aplusml.config.ConfigState(
          type = 'end',
          utilities = [
              aplusml.config.ConfigUtility(
                  value = 2,
                  unit = 'qaly',
              ),
          ],
      ),
    },
  )

  # Create Simulation object
  simulation: aplusml.Simulation = aplusml.Simulation.create_from_config(config)

  # Set random seed for reproducibility
  np.random.seed(0)

  ############################################################
  # 2. Initialize patients
  ############################################################

  # Simulate number of patients per day
  n_admits_per_day = np.random.poisson(lam=35, size=500)

  # Create empty Patient objects (with proper start timesteps according to our Poisson distribution)
  patients: List[aplusml.Patient] = []
  for timestep, n_admits in enumerate(n_admits_per_day):
      for x in range(n_admits):
          patients.append(aplusml.Patient(
              len(patients), # Unique ID
              timestep, # Start timestep
          ))

  # Function which matches a patient to a row in the CSV file.
  func_match_patient_to_property_column = lambda p_id, random_idx, df, col: df.iloc[random_idx][col]

  # Initialize patients for simulation. 
  # This creates a deep copy of each object in the `patients` array using pickle, sorts them by ID, and then initializes their properties
  patients: List[aplusml.Patient] = simulation.create_patients_for_simulation(patients, 
                                                                            func_match_patient_to_property_column,
                                                                            random_seed = 0)
  
  ############################################################
  # 3. Run simulation
  ############################################################

  patients = simulation.run(patients)

  ############################################################
  # 4. Analyze results
  ############################################################

  # Sum up the utilities achieved across all patient histories
  sum_utilities: Dict[str, float] = collections.defaultdict(float)
  for p in patients:
      _u: dict = p.get_sum_utilities(simulation)
      for key, val in _u.items():
          sum_utilities[key] += val

  # Print results
  print(sum_utilities)