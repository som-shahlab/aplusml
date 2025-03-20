Quick Start
============

.. toctree::
   :maxdepth: 2
   :caption: Quick Start

Code Example
------------

1\. Install the APLUS package:

.. code-block:: bash

   pip install aplusml


2\. Create your workflow YAML file and patient properties CSV file following the specifications in :doc:`config`.


3\. Create a simulation object by loading your workflow YAML file and patient properties CSV file.

.. code-block:: python

  import aplusml

  PATH_TO_CONFIG_YAML: str = "path/to/config.yaml"
  PATH_TO_PATIENT_PROPERTIES: str = "path/to/patient_properties.csv"

  # Create simulation. Loads workflow + simulation parameters from YAML file, patient properties from CSV file.
  # If `path_to_patient_properties` is None, then the default `metadata.path_to_properties` key in the YAML file is used. Here, we explicitly override it.
  simulation: aplusml.Simulation = aplusml.load_simulation(PATH_TO_CONFIG_YAML, PATH_TO_PATIENT_PROPERTIES)

4\. Visualize the workflow via a **graphviz** diagram:

.. code-block:: python

  # Draws workflow diagram. This will save the diagram to './output.png' and print it to your terminal.
  simulation.draw_workflow_diagram(figsize=(30,30), path_to_file='./output.png', is_display=True)

5\. Create a list of patients to simulate. We simulate the flow of patients as a Poisson process with the following parameters:

  * :math:`p` patients start our workflow each day, where :math:`p \sim \text{Poisson}(\lambda=35)`
  * :math:`N` total days will be simulated, where :math:`N=500`

.. code-block:: python

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
  
  # Runsimulation
  simulation.run(patients)


7\. Sum up the utilities achieved across all patient histories to ascertain the total utility achieved by the workflow:

.. code-block:: python
  
  # Sum up the utilities achieved across all patient histories
  sum_utilities: Dict[str, float] = collections.defaultdict(float)
  for p in patients:
      _u: dict = p.get_sum_utilities(simulation)
      for key, val in _u.items():
          sum_utilities[key] += val

  # Print results
  print(sum_utilities)


Useful Concepts
----------------

Transitions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Patients move between states through **transitions**. There are three types of transitions:

  * **Deterministic** – A single, predefined transition per state.  
  * **Probabilistic** – Transitions occur with specified probabilities (e.g., 30% "high-risk", 70% "low-risk").  
  * **Conditional** – Transitions based on Boolean expressions evaluating patient-level or system-level conditions (e.g., a gene therapy succeeding only for patients with a specific mutation).  

Utilities and Performance Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
APLUS supports tracking **multiple utility measures** to evaluate workflow effectiveness:

  * **Time-based** (e.g., patient length-of-stay)  
  * **Clinical** (e.g., patient outcomes)  
  * **Financial** (e.g., cost impact)  
  * **Resource-related** (e.g., staff utilization)  

Utilities can be **conditioned on expressions**, allowing individual-level utility analyses based on patient properties [29]. These values must be derived from literature, expert consultation, or financial modeling [32].  

Temporal and Resource Modeling
""""""""""""""""""""""""""""""""

**Time Duration:** Each state and transition can have an associated **time duration**, defining the number of simulation time steps before progression. Example:  

  * A hospital workflow where post-surgery patients stay in a "rest" state before daily evaluations.  

**Resource Constraints:** Transitions can alter **system-level resources** via **resource deltas**, simulating real-world capacity limits. Example:  

  * A transition directing a patient to an MRI scan may decrease **MRI capacity by 1**.  

