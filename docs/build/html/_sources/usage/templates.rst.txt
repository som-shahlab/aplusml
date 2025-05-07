Config YAML Templates
======================

.. toctree::
   :maxdepth: 2
   :caption: Config YAML Templates

This page contains some minimal YAML templates for various configurations.

No Patient Properties CSV
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This config is for the case where **no patient properties CSV** is provided.

.. code-block:: yaml
  :caption: `config.yaml`
  :linenos:

  metadata:
    name: "My simulation"

  variables:
    var_constant:
      type: scalar
      value: 10
    sim_property__los:
      # Every patient has a different random LOS, which is sampled from a normal distribution with mean 5 and std 1.
      type: property
      distribution: normal
      mean: 5
      std: 1

  states:
    state_1:
      label: "State 1"
      type: start
      transitions:
        - dest: state_2
    state_2:
      label: "State 2"
      type: intermediate
      transitions:
        - dest: state_3
    state_3:
      label: "State 3"
      type: end


Resource-Driven Config
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This config has a **resource** that changes over time.

.. code-block:: yaml
  :caption: `config.yaml`
  :linenos:

  metadata:
    name: "My simulation"

  variables:
    resource__open_beds:
      # Starts at 3, refills by 2 every 1 timesteps, maxes out at 10.
      type: resource
      init_amount: 3
      max_amount: 10
      refill_amount: 2
      refill_duration: 1

  states:
    state_1:
      label: "State 1"
      type: start
      transitions:
        - dest: state_2
    state_2:
      label: "State 2"
      type: intermediate
      transitions:
        # If there are open beds, then the patient can be admitted...
        - dest: state_admitted
          if: resource__open_beds > 0
          resource_deltas:
            # If patient takes this transition to state 3, then 1 bed is used.
            resource__open_beds: -1
        # ...otherwise, the patient is refused.
        - dest: state_refused
    state_admitted:
      label: "Admitted State"
      type: end
    state_refused:
      label: "Refused State"
      type: end


Patient Properties CSV
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This config has a **patient properties CSV** that must be provided at the path ``path/to/patient_properties.csv``.

The CSV must have a column for the patient ID (named ``id``), and then one column for each property referenced in the ``variables`` section (e.g. ``y`` and ``y_hat``).

.. code-block:: yaml
  :caption: `config.yaml`
  :linenos:

  metadata:
    name: "My simulation"
    path_to_properties: "path/to/patient_properties.csv"
    properties_col_for_patient_id: "id" # This property is LOADED from the CSV file

  variables:
    csv_property__y:
      type: property
      # This property is LOADED from the CSV file b/c the 'column' attribute is set.
      column: "y"
    csv_property__y_hat:
      type: property
      # This property is LOADED from the CSV file b/c the 'column' attribute is set.
      column: "y_hat"
    sim_property__los:
      type: property
      # This property is NOT LOADED from the CSV file b/c the 'column' attribute is not set.
      # Instead, the value is randomly sampled by the simulation.
      distribution: normal
      mean: 5
      std: 1

  states:
    state_1:
      label: "State 1"
      type: start
      transitions:
        - dest: state_2
    state_2:
      label: "State 2"
      type: intermediate
      transitions:
        # If model prediction is correct, then the patient is successfully treated.
        - dest: state_success
          if: csv_property__y == csv_property__y_hat
        # ...otherwise, the model failed.
        - dest: state_failure
    state_success:
      label: "Success State"
      type: end
    state_failure:
      label: "Failure State"
      type: end