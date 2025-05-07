Concept Guide
================

.. toctree::
   :maxdepth: 2
   :caption: Concept Guide

This page describes the core models used by APLUS -- namely, :class:`~aplusml.sim.Simulation`, :class:`~aplusml.models.Patient`, :class:`~aplusml.models.Transition`, :class:`~aplusml.models.State`, and :class:`~aplusml.models.Utility`.

:class:`~aplusml.sim.Simulation`
------------------------------------

For full documentation, see :class:`~aplusml.sim.Simulation`.

The :class:`~aplusml.sim.Simulation` class handles all the details of running a simulation. There are four basic methods to know:

1. **Initialize Simulation** -- :meth:`~aplusml.sim.Simulation.create_from_yaml` or :meth:`~aplusml.sim.Simulation.create_from_config`
2. **Initialize Patients** -- :meth:`~aplusml.sim.Simulation.create_patients_for_simulation`
3. **Run Simulation** -- :meth:`~aplusml.sim.Simulation.run`
4. **Visualize Workflow** -- :meth:`~aplusml.sim.Simulation.draw_workflow_diagram`


1. **Initialize Simulation** -- :meth:`~aplusml.sim.Simulation.create_from_yaml` or :meth:`~aplusml.sim.Simulation.create_from_config` -- Create the :class:`~aplusml.sim.Simulation` object from a YAML configuration file and a CSV file containing patient properties.

.. code-block:: python
  :linenos:

  import aplusml
  simulation = aplusml.Simulation.create_from_yaml(PATH_TO_CONFIG_YAML, PATH_TO_PATIENT_PROPERTIES)
  # or
  simulation = aplusml.Simulation.create_from_config(config)

From the config, the :class:`~aplusml.sim.Simulation` loads the following attributes:

    - :attr:`metadata` (Dict): Configuration metadata for the simulation. A dictionary of strings.
    - :attr:`variables` (Dict[str, Dict]): Variables used in the simulation. A dictionary of variables, keyed by ID.
    - :attr:`states` (Dict[str, State]): States in the workflow. A dictionary of states, keyed by ID.

2. **Initialize Patients** -- :meth:`~aplusml.sim.Simulation.create_patients_for_simulation` -- Create a list of :class:`Patient` objects.

.. code-block:: python
  :linenos:

  import aplusml

  # Create dummy patients
  patients: List[aplusml.Patient] = [
    aplusml.Patient(0,
        0, # Unique ID
        0, # Start timestep
    ),
    aplusml.Patient(1,
        1, # Unique ID
        0, # Start timestep
    ),
    aplusml.Patient(2,
        2, # Unique ID
        1, # Start timestep
    ),
  ]

  # Initialize patients for simulation. 
  # This creates a deep copy of each object in the `patients` array using pickle, sorts them by ID, and then initializes their properties
  patients: List[aplusml.Patient] = simulation.create_patients_for_simulation(patients, random_seed=0)


3. **Run Simulation** -- :meth:`~aplusml.sim.Simulation.run` -- Run a list of patients through the workflow.

.. code-block:: python
  :linenos:

  # Run the simulation
  patients = simulation.run(patients)

During the simulation, the following attributes will be automatically tracked:

    - :attr:`variable_history` (Dict[List[Tuple[int, Any]]]): History of variable values over time. A dictionary of lists of tuples, where each tuple contains a timestep and a dictionary of variable values.
    - :attr:`current_timestep` (int): Current simulation timestep. 


4. **Visualize Workflow** -- :meth:`~aplusml.sim.Simulation.draw_workflow_diagram` -- Display the workflow as a graphviz diagram.

.. code-block:: python
  :linenos:

  # Draws workflow diagram. This will save the diagram to './output.png' and print it to your terminal.
  simulation.draw_workflow_diagram(figsize=(30,30), path_to_file='./output.png', is_display=True)


:class:`~aplusml.models.Patient`
------------------------------------

For full documentation, see :class:`~aplusml.models.Patient`.

This class represents a single patient in the simulation. It contains the following attributes:

  - :attr:`id` (str): The unique ID of the patient.
  - :attr:`start_timestep` (int): The timestep at which the patient enters the workflow.
  - :attr:`properties` (Dict[str, Any]): A dictionary of properties for the patient.
  - :attr:`history` (List[History]): A list of history objects, which track the patient's history through the workflow.
  - :attr:`current_state` (str): The ID of the current state that the patient is in.

The most useful method is the :meth:`get_sum_utilities` method, which returns the sum of the utilities achieved by the patient over the course of the entire simulation.
Each unique utility unit (e.g. "QALYs", "$", etc.) will get its own key in the returned dictionary.

.. code-block:: python
  :linenos:

  # Get the sum of the utilities achieved by the patient over the course of the entire simulation
  sum_utilities: Dict[str, float] = patient.get_sum_utilities(simulation)
  print(sum_utilities)

The most useful attribute is the :attr:`history` attribute. It contains a list of :class:`~aplusml.models.History` objects. This allows you to trace the patient's entire journey through the workflow.

.. code-block:: python
  :linenos:

  # Print the patient's history
  print(patient.history)


:class:`~aplusml.models.Transition`
------------------------------------

For full documentation, see :class:`~aplusml.models.Transition`.

Patients move between states through **transitions**. There are three types of transitions:

  * **Deterministic** – A single, predefined transition per state.  
  * **Probabilistic** – Transitions occur with specified probabilities (e.g., 30% "high-risk", 70% "low-risk").  
  * **Conditional** – Transitions based on Boolean expressions evaluating patient-level or system-level conditions (e.g., a gene therapy succeeding only for patients with a specific mutation).  

You do not need to interact with this class directly.

:class:`~aplusml.models.State`
------------------------------------

For full documentation, see :class:`~aplusml.models.State`.

States are the building blocks of the workflow. Each state has a unique ID, a label, a type, a duration, and a list of transitions.

You do not need to interact with this class directly.

:class:`~aplusml.models.Utility`
------------------------------------

For full documentation, see :class:`~aplusml.models.Utility`.

APLUS supports tracking multiple utility measures to evaluate workflow effectiveness:

  * **Time-based** (e.g., patient length-of-stay)  
  * **Clinical** (e.g., patient outcomes)  
  * **Financial** (e.g., cost impact)  
  * **Resource-related** (e.g., staff utilization)  

Utilities can be conditioned on expressions, allowing individual-level utility analyses based on patient properties.

You do not need to interact with this class directly.

Temporal and Resource Modeling
-----------------------------------

**Time Duration:** Each state and transition can have an associated **time duration**, defining the number of simulation time steps before progression. Example:  

  * A hospital workflow where post-surgery patients stay in a "rest" state before daily evaluations.  

**Resource Constraints:** Transitions can alter **system-level resources** via **resource deltas**, simulating real-world capacity limits. Example:  

  * A transition directing a patient to an MRI scan may decrease **MRI capacity by 1**.  

