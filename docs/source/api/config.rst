Configuration
==================

.. toctree::
   :maxdepth: 2
   :caption: Configuration

APLUS simulations can be specified in two ways -- via the :class:`~aplusml.config.Config` object or via a **(1) Workflow YAML** and **(2) Patient Properties CSV**.

1. Pythonic Config object
---------------------------------------------

Documentation for the :class:`~aplusml.config.Config` object is available :doc:`/api/api`

.. code-block:: python
  :caption: `script.py`
  :linenos:

  import aplusml
  
  # Create Config object
  config = aplusml.config.Config(
    metadata = aplusml.config.ConfigMetadata(
      name = 'My Simulation',
    ),
    variables = {
      'variable_id': aplusml.config.ConfigVariable(
        type = 'scalar',
        value = 1,
      ),
    },
    states = {
      'start' : aplusml.config.ConfigState(
        type = 'start',
        transitions = [
          aplusml.config.ConfigTransition(
            dest = 'end',
          ),
        ],
      ),
      'end' : aplusml.config.ConfigState(
        type = 'end',
      ),
    },
  )

  # Create Simulation object
  simulation = aplusml.Simulation.create_from_config(config)

2. Workflow YAML + Patient Properties CSV
---------------------------------------------

For improved readability, an APLUS simulation can also be specified via a **(1) Workflow YAML** and **(2) Patient Properties CSV**.

1. **Workflow YAML** specifies the states, transitions, utilities, resources, and other parameters of your workflow.
2. **(Optional) Patient Properties CSV** contains individual-level properties for each patient that will be run through your workflow.

This is equivalent to the Pythonic Config object (i.e. it gets converted to a `:class:aplusml.config.Config` object under the hood).

.. code-block:: python
  :caption: `script.py`
  :linenos:

  import aplusml

  PATH_TO_CONFIG_YAML: str = 'config.yaml'
  PATH_TO_PATIENT_PROPERTIES: str = 'patient_properties.csv'

  # Create Simulation object
  simulation = aplusml.Simulation.create_from_yaml(PATH_TO_CONFIG_YAML, PATH_TO_PATIENT_PROPERTIES)


🔧 Workflow YAML
^^^^^^^^^^^^^^^^^^^^^^^^^^

* *Format:* YAML
* *Purpose:* Specifies the states, transitions, utilities, resources, and other parameters of your workflow
* *Schema:* Listed below

The YAML must be formatted as follows:

.. code-block:: yaml
  :caption: `config.yaml`
  :linenos:

  metadata:
    name: Optional[str]  # Name of the simulation
    path_to_properties: Optional[str]  # Path to CSV file where each row is a patient, each column is a property. Note: Only properties explicitly enumerated in the 'variables' section will be imported
    properties_col_for_patient_id: Optional[str]  # Column name in the CSV that contains unique patient IDs.
    patient_sort_preference_property: Optional[dict]  # Determines priority order for resource allocation.
      variable: str  # Name of a property (must be defined in `variables`) that will be used to sort patients when prioritizing allocation of a finite resource
      is_ascending: bool  # True = ascending order, False = descending.

  variables:  # Dictionary of variables, each with a unique key.
    [key]: # `key` is the ID of the variable. It must be unique.
      type: Enum(scalar, resource, property, simulation)  # Type of variable. Default: "scalar".
      
      # If type == 'scalar'
      #   This is a scalar value that is shared across all patients.
      #   It can be used to model things like the sensitivity of a screening test, the prevalence of a disease, etc.
      value: Union[int, float, bool, str, list, dict, set]  # Scalar value. Use '!!set' tag for sets.

      # If type == 'resource'
      #   This is a finite resource that is shared across all patients. 
      #   It can be decremented, incremented, and reset by the simulation. 
      #   It can be used to model things like hospital beds, lab capacity, etc.
      init_amount: int  # Initial amount of the resource.
      max_amount: int  # Maximum amount of resource allowed.
      refill_amount: int  # Amount added per refill.
      refill_duration: int  # Time interval between refills.

      # If type == 'property'
      #   This is a property that is UNIQUE to each patient (i.e. each patient may have a different value for this property).
      #   It can be used to model things like the age of a patient, the gender of a patient, etc.
      ## Either load from file...
      column: str  # If loaded from a CSV file, specify the column name (e.g. 'y' or 'y_hat_dl'). Each row of the CSV will be a patient, and the value of this property for each patient will be the value of the column in the CSV file.
      ## or constant...
      value: Any  # If constant, specify value.
      ## or randomly sample...
      distribution: Enum(bernoulli, exponential, binomial, normal, poisson, uniform)  # If randomly sampled.
      mean: Optional[float]  # Mean value for distribution.
      std: Optional[float]  # Standard deviation.
      start: Optional[float]  # Minimum value.
      end: Optional[float]  # Maximum value.

  states:  # Dictionary of states, each with a unique key.
    [key]: str # `key` is the ID of the state. It must be unique.
      label: Optional[str]  # Human-readable label for the state. Default: value of `key`.
      type: Optional[Enum(start, end, intermediate)]  # Whether the state is a start, end, or intermediate state within the workflow. Default: "intermediate".
      duration: Optional[float]  # Number of timesteps to wait before transitions are evaluated. Default: 0.0. 
      
      utilities: Optional[Union[str, float, bool, list[dict]]]  # If string, it's evaluated as a Python expression. Default: 0.0.
        # If list[dict], then multiple utilities can be defined. Note: These 'if' statements are not mutually exclusive (i.e. if multiple conditions evaluate to TRUE, then they will simply be summed together)
        - value: Optional[Union[float, str]]  # If string, it's evaluated as a Python expression.
          if: Optional[Union[str, bool]]  # A Python expression. If it evaluates to TRUE, then the `value` for this utility is set to this `value`.
          unit: Optional[str]  # Measurement unit.

      resource_deltas: Optional[Dict[str, float]]  # Changes to resource levels from entering this state. Default: {}.
        [key]: str # `key` is the name of a resource defined in `variables`.
          [value] # Required. How much to change each resource level AS SOON AS this state is hit
      
      transitions: Optional[List[Dict]]  # List of possible state transitions.
        - dest: str  # Required. ID of the destination state.
          label: Optional[str] # Default: "".

          # Transition conditions
          ## Can either have...
          ## - All transitions have an 'if' condition (where if the last transition doesn't have an 'if', it defaults to always TRUE)
          ## - All transitions have a 'prob' condition (where if the last transition doesn't have a 'prob', it defaults to = 1 - (sum of other probs))
          ## - The first set of transitions have an 'if' condition, but the second set have a 'prob'
          if: Optional[Union[str, bool]]  # Conditional expression. If string, it's evaluated as a Python expression. Must come before 'prob' (if mixed). Must always be at least one TRUE condition across all transitions for this state (unless mixed with 'prob'). `if` conditionals will be evaluated in order and break on first TRUE. If the last `if` doesn't have a condition, then it defaults to TRUE.
          prob: Optional[float]  # Probability of transition. If string, it's evaluated as a Python expression. Must come after 'if' (if mixed). If mixed, then 'prob' is conditional probability given all 'if' are FALSE. Must sum to 1 across all 'prob' transitions for this state. If the last `prob` doesn't have a condition, then it defaults to = 1 - (sum of other probs).

          duration: Optional[float]  # Number of timesteps to wait before transitions are evaluated. Default: 0.0.

          utilities: Optional[Union[str, float, bool, list[dict]]]  # Same specification as the utilities for states.

          resource_deltas: Optional[Dict[str, float]] # Changes to resource levels from taking this transition. Default: {}.
            [key]: str # `key` is the name of a resource defined in `variables`.
              [value] # How much to change each resource level AS SOON AS this transition is taken

**NOTE:** APLUS makes the following simulation-level variables available to the user. To use them, you must include them in the ``variables`` section of the YAML file:

- ``sim_current_timestep``: The current timestep of the simulation.
- ``time_left_in_sim``: The number of timesteps remaining in the simulation.
- ``time_already_in_sim``: The number of timesteps that have passed in the simulation.

e.g.

.. code-block:: yaml
  :caption: `config.yaml`

  variables:
    sim_current_timestep:
      type: simulation
    time_left_in_sim:
      type: simulation
    time_already_in_sim:
      type: simulation


📄 Patient Properties CSV
^^^^^^^^^^^^^^^^^^^^^^^^^^

* *Format:* CSV
* *Purpose:* Contains individual-level properties for each patient that will be run through your workflow
* *Schema:* Listed below

The CSV must be formatted as follows:

Each row is a unique patient, and each column is a property. Properties can be anything -- e.g. demographic information, model predictions, clinical measurements, etc.

Example properties include:

* ``id``: The unique identifier for the patient.
* ``age``: The age of the patient.
* ``gender``: The gender of the patient.
* ``y``: The ground truth label for the patient.
* ``y_hat``: The predicted label for the patient.
* ``random_resource_priority``: A random number that we will use to sort patients when prioritizing them to access a finite resource.

Here is an excerpt from an example properties CSV file:

.. code-block:: text
  :caption: `patient_properties.csv`

   id,y,y_hat_dl,y_hat_rf,y_hat_lr,abi_test_pred,random_resource_priority
   1,0,0.95,0.9,0.85,0.9,0.1
   2,1,0.85,0.8,0.75,0.8,0.2
   3,0,0.75,0.7,0.65,0.7,0.3