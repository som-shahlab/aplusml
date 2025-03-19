Configuration
==================

.. toctree::
   :maxdepth: 2
   :caption: Configuration

APLUS takes two main inputs:

1. **Configuration YAML**: Specifies your workflow + simulation parameters
2. **Patient Properties CSV**: Specifies individual-level properties for each patient that will be run through your workflow.


🔧 Configuration YAML
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This configuration file specifies the states, transitions, utilities, resources, and other parameters of your workflow.

The schema of this YAML configuration file is as follows:

.. code-block:: yaml
  
  metadata:
    name: str  # (optional) Name of the simulation
    path_to_properties: str  # (optional) Path to CSV file where each row is a patient, each column is a property. Note: Only properties explicitly enumerated in the 'variables' section will be imported
    properties_col_for_patient_id: str  # (optional) Column name in the CSV that contains unique patient IDs.
    
    patient_sort_preference_property:  # (optional) Determines priority order for resource allocation.
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
      mean: float  # (optional) Mean value for distribution.
      std: float  # (optional) Standard deviation.
      start: float  # (optional) Minimum value.
      end: float  # (optional) Maximum value.

  states:  # Dictionary of states, each with a unique key.
    [key]: str # `key` is the ID of the state. It must be unique.
      label: str  # (optional) Human-readable label for the state. Default: value of `key`.
      type: Enum(start, end, intermediate)  # (optional) Whether the state is a start, end, or intermediate state within the workflow. Default: "intermediate".
      duration: float  # (optional) Number of timesteps to wait before transitions are evaluated. Default: 0.0. 
      
      utilities: Union[str, float, bool, list[dict]]  # (optional) If string, it's evaluated as a Python expression. Default: 0.0.
        # If list[dict], then multiple utilities can be defined. Note: These 'if' statements are not mutually exclusive (i.e. if multiple conditions evaluate to TRUE, then they will simply be summed together)
        - value: Union[float, str]  # (optional) If string, it's evaluated as a Python expression.
          if: str  # (optional) A Python expression. If it evaluates to TRUE, then the `value` for this utility is set to this `value`.
          unit: str  # (optional) Measurement unit.

      resource_deltas: dict[str, float]  # (optional) Changes to resource levels from entering this state. Default: {}.
        [key]: str # (optional) `key` is the name of a resource defined in `variables`.
          [value] # (optional) How much to change each resource level AS SOON AS this state is hit
      
      transitions: list[dict]  # (optional) List of possible state transitions.
        - dest: str  # Required. ID of the destination state.
          label: str  # (optional) Default: "".

          # Transition conditions
          ## Can either have...
          ## - All transitions have an 'if' condition (where if the last transition doesn't have an 'if', it defaults to always TRUE)
          ## - All transitions have a 'prob' condition (where if the last transition doesn't have a 'prob', it defaults to = 1 - (sum of other probs))
          ## - The first set of transitions have an 'if' condition, but the second set have a 'prob'
          if: Union[bool, str]  # (optional) Conditional expression. If string, it's evaluated as a Python expression. Must come before 'prob' (if mixed). Must always be at least one TRUE condition across all transitions for this state (unless mixed with 'prob'). `if` conditionals will be evaluated in order and break on first TRUE. If the last `if` doesn't have a condition, then it defaults to TRUE.
          prob: float  # (optional) Probability of transition. If string, it's evaluated as a Python expression. Must come after 'if' (if mixed). If mixed, then 'prob' is conditional probability given all 'if' are FALSE. Must sum to 1 across all 'prob' transitions for this state. If the last `prob` doesn't have a condition, then it defaults to = 1 - (sum of other probs).

          duration: float  # (optional) Number of timesteps to wait before transitions are evaluated. Default: 0.0.

          utilities: Union[str, float, bool, list[dict]]  # (optional) Same specification as the utilities for states.

          resource_deltas: dict[str, float] # (optional) Changes to resource levels from taking this transition. Default: {}.
            [key]: str # (optional) `key` is the name of a resource defined in `variables`.
              [value] # (optional) How much to change each resource level AS SOON AS this transition is taken

📄 Patient Properties CSV
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This CSV file contains individual-level properties for each patient that will be run through your workflow.

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

   id,y,y_hat_dl,y_hat_rf,y_hat_lr,abi_test_pred,random_resource_priority
   1,0,0.95,0.9,0.85,0.9,0.1
   2,1,0.85,0.8,0.75,0.8,0.2
   3,0,0.75,0.7,0.65,0.7,0.3