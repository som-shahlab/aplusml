Tutorial (HCM)
================

.. toctree::
   :maxdepth: 2
   :caption: Tutorial (HCM)

In this section, we use APLUS to evaluate a Hypertrophic Cardiomyopathy (HCM) workflow.

The full ``.ipynb`` notebook for this tutorial can be found `at this link <https://github.com/som-shahlab/aplusml/blob/main/tutorials/hcm.ipynb>`_.

🏥 Clinical Motivation
----------------------

The goal of this project is to evaluate the clinical, resource utilization, financial, and ethical impact of possible deployment of a Hypertrophic Cardiomyopathy (HCM) ML detection model.

🤖 ML Models
-------------

TODO

👩‍⚕️ Workflows
----------------

We were given four possible HCM workflows to consider.

First, we considered the **current state** of HCM care, where patients are referred to the specialist by a primary care physician.
.. image:: ../_static/hcm_current_state.png
   :width: 700
   :alt: Current HCM workflow

Next, we considered an **AI-guided** workflow, where the patient's EHR is reviewed by an AI to determine if they are a candidate for HCM workup.

.. image:: ../_static/hcm_ai.png
   :width: 700
   :alt: AI-guided HCM workflow

We also considered two baseline workflows. The first is the **optimistic** workflow, where all patients are referred to the specialist without any sort of screening or capacity constraints.

.. image:: ../_static/hcm_optimistic.png
   :width: 700
   :alt: Optimistic HCM workflow

The second is the **random** workflow, where patient's are randomly referred to the specialist.

.. image:: ../_static/hcm_random.png
   :width: 700

🔧 Creating the APLUS Config
----------------------------

Let's now create our config files, one for each workflow. 

We'll start with the **optimistic** workflow, then the **random** workflow, then the **current state** workflow, and finally the **AI-guided** workflow.

For each workflow, we'll start by defining all of the steps in the workflow as states in our YAML file. We'll then add any variables needed for the workflow, then show the full config file at the bottom.

1. Optimistic Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^

For reference, here is the **optimistic** workflow that we're trying to replicate in APLUS:

.. image:: ../_static/hcm_optimistic.png
   :width: 700
   :alt: Optimistic HCM workflow

We'll start by defining all of the steps in the workflow as states in our YAML file.

Our first state is called "All screenable patients" (in green). Let's add this state to our YAML file.

.. code-block:: yaml

  states:
    start:
      type: start
    label: "Start"
    transitions:
      - dest: all_screenable_patients
    all_screenable_patients:
      label: "All screenable patients"
      transitions:
        - dest: visit_hcm_clinic

The "All screenable patients" state will immediately send all patients to the HCM clinic. These patients are then split into two groups: Diagnosed and Not Diagnosed. 

We'll assume that this split is done perfectly by the HCM clinic, i.e. all patients with HCM (as marked by a patient-level property `has_hcm`) are diagnosed and all patients without HCM are undiagnosed.

.. code-block:: yaml

  states:
    visit_hcm_clinic:
      # ... existing states ...
      label: "Patients worked up at HCM clinic"
      transitions:
        - dest: diagnosed
          if: has_hcm
        - dest: undiagnosed
    diagnosed:
      label: "Diagnosed"
      transitions:
        - dest: untreated
    undiagnosed:
      label: "Undiagnosed"
      transitions:
        - dest: untreated

Finally, we create end states for each of the four possible outcomes (True Positive, False Positive, True Negative, False Negative) and assign utility values to each.

.. code-block:: yaml

  states:
    untreated:
      label: "Untreated"
      utility: 0
    true_positive:
      label: "True Positive"
      utility: 1
    false_positive:
      label: "False Positive"
      utility: -1
    true_negative:
      label: "True Negative"
      utility: 1
    false_negative:
      label: "False Negative"
      utility: -1

Let's finish this config by adding relevant variables to the config.

.. code-block:: yaml

  variables:
    # Patient properties
    has_hcm:
      type: property
      column: has_hcm # Boolean property that is true if the patient has HCM
    # Fixed constants
    hcm_prevalence:
      value: 1/350 # Prevalence of HCM in the population ('p' in the diagram)

Our final **optimistic** config is shown below.

.. code-block:: yaml

  metadata:
    name: "HCM (Optimistic)"
    path_to_properties: "input/hcm/properties.csv"

  variables:
    # Patient properties
    has_hcm:
      type: property
      column: has_hcm # Boolean property that is true if the patient has HCM
    # Fixed constants
    hcm_prevalence:
      value: 1/350 # Prevalence of HCM in the population ('p' in the diagram)

  states:
    start:
      type: start
      label: "Start"
      transitions:
        - dest: all_screenable_patients
    all_screenable_patients:
      label: "All screenable patients"
      transitions:
        - dest: visit_hcm_clinic
    visit_hcm_clinic:
      label: "Patients worked up at HCM clinic"
      transitions:
        - dest: diagnosed
          if: has_hcm
        - dest: undiagnosed
    diagnosed:
      label: "Diagnosed"
      transitions:
        - dest: untreated
    undiagnosed:
      label: "Undiagnosed"
      transitions:
        - dest: untreated
    untreated:
      label: "Untreated"
      utility: 0
    true_positive:
      label: "True Positive"
      utility: 1
    false_positive:
      label: "False Positive"
      utility: -1
    true_negative:
      label: "True Negative"
      utility: 1
    false_negative:
      label: "False Negative"
      utility: -1

2. Random Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^

For reference, here is the **random** workflow that we're trying to replicate in APLUS:

.. image:: ../_static/hcm_random.png
   :width: 700
   :alt: Random HCM workflow

Many of the states are the same as in the optimistic workflow, so we'll only show the new states below:

.. code-block:: yaml

  states:
    # ...overwritten states from optimistic workflow ...
    start:
      label: "Start"
      transitions:
        - dest: screen_patients
    # ... existing states from optimistic workflow ...
    # ... new states ...
    screen_patients:
      label: "Screen patients"
      transitions:
        - dest: visit_hcm_clinic
          if: patient_screen_idx <= n_patients_screened
        - dest: undiagnosed

And we'll add one new variable to the config to represent the capacity constraint on the number of patients screened.

.. code-block:: yaml

  variables:
    # ... existing variables ...
    # ... new variables ...
    patient_screen_idx:
      type: property
      column: patient_screen_idx # Random index of the patient; used for determining which patients get screened given a capacity constraint. All patients with ``patient_random_idx <= n_patients_screened`` will be screened.
    n_patients_screened:
      value: 1_000 # Capacity constraint on the number of patients screened ('C' in the diagram)

Putting it all together, we get the following config:

.. code-block:: yaml

  metadata:

  metadata:
    name: "HCM (Random)"
    path_to_properties: "input/hcm/properties.csv"

  variables:
    # Patient properties
    has_hcm:
      type: property
      column: has_hcm # Boolean property that is true if the patient has HCM
    patient_screen_idx:
      type: property
      column: patient_screen_idx # Random index of the patient; used for determining which patients get screened given a capacity constraint. All patients with ``patient_random_idx <= n_patients_screened`` will be screened.
    # Fixed constants
    hcm_prevalence:
      value: 1/350 # Prevalence of HCM in the population ('p' in the diagram)
    n_patients_screened:
      value: 1_000 # Capacity constraint on the number of patients screened ('C' in the diagram)

  states:
    start:
      type: start
      label: "Start"
      transitions:
        - dest: screen_patients
    screen_patients:
      label: "Screen patients"
      transitions:
        - dest: visit_hcm_clinic
          if: patient_screen_idx <= n_patients_screened
        - dest: undiagnosed
    visit_hcm_clinic:
      label: "Patients worked up at HCM clinic"
      transitions:
        - dest: diagnosed
          if: has_hcm
        - dest: undiagnosed
    diagnosed:
      label: "Diagnosed"
      transitions:
        - dest: untreated
    undiagnosed:
      label: "Undiagnosed"
      transitions:
        - dest: untreated
    untreated:
      label: "Untreated"
      utility: 0
    true_positive:
      label: "True Positive"
      utility: 1
    false_positive:
      label: "False Positive"
      utility: -1
    true_negative:
      label: "True Negative"
      utility: 1
    false_negative:
      label: "False Negative"
      utility: -1

3. Current State Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^

We're now going to try to replicate the **current state** workflow (shown below) in APLUS:

.. image:: ../_static/hcm_current_state.png
   :width: 700
   :alt: Current state HCM workflow

Again, we will reuse most of the states from the **random** workflow, but will add a few states as shown below:

.. code-block:: yaml

  states:
    # ...overwritten states from random workflow ...
    start:
      label: "Start"
      transitions:
        - dest: prescreen_patients
    # ... existing states from random workflow ...
    # ... new states ...
    prescreen_patients: # Clinical pre-screening
      label: "Clinically screened patients"
      transitions:
        - dest: screen_patients
          if: (has_hcm and (clinical_result <=  clinical_sensitivity)) or (not has_hcm and (clinical_result >= clinical_specificity))
        - dest: undiagnosed

And add a variable to represent the clinical screening sensitivity and specificity.

.. code-block:: yaml

  variables:
    # ... existing variables ...
    # ... new variables ...
    clinical_result:
      type: property
      distribution: uniform
      start: 0
      end: 1
    clinical_sensitivity:
      value: 0.95 # Sensitivity of clinical screening ('s' in the diagram)
    clinical_specificity:
      value: 0.95 # Specificity of clinical screening ('r' in the diagram)

4. AI-guided Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, we're going to replicate the **AI-guided** workflow (shown below) in APLUS:

.. image:: ../_static/hcm_ai.png
   :width: 700
   :alt: AI-guided HCM workflow

We'll reuse the states from the **random** workflow, but will need to add several new steps for the ECG and Echo screening:

.. code-block:: yaml

  states:
    # ...overwritten states from random workflow ...
    start:
      label: "Start"
      transitions:
        - dest: ecg_screening
    # ... existing states from random workflow ...
    # ... new states ...
    ecg_screening:
      label: "ECG screening"
      transitions:
        - dest: echo_screening
          # The patient is marked as POSITIVE for ECG screen if...
          #   Has HCM + ECG result is <= the sensitivity of the ECG screening, or
          #   Does not have HCM + ECG result is >= the specificity of the ECG screening
          if: (has_hcm and (ecg_result <= ecg_sensitivity)) or (not has_hcm and (ecg_result >= ecg_specificity))
        - dest: undiagnosed
    echo_screening:
      label: "Echo screening"
      transitions:
        - dest: hcm_triage
          # The patient is marked as POSITIVE for Echo screen if...
          #   Has HCM + Echo result is <= the sensitivity of the Echo screening, or
          #   Does not have HCM + Echo result is >= the specificity of the Echo screening
          if: (has_hcm and (echo_result <= echo_sensitivity)) or (not has_hcm and (echo_result >= echo_specificity))
        - dest: undiagnosed
    hcm_triage:
      label: "HCM triage"
      transitions:
        - dest: visit_hcm_clinic
          if: patient_screen_idx <= n_patients_screened
        - dest: hcm_flex_waitlist
    hcm_flex_waitlist:
      label: "Patients not worked up at HCM clinic with minimal delay"
      transitions:
        - dest: visit_hcm_clinic_delayed
          if: patient_delayed_screen_idx <= n_patients_delayed_screened
        - dest: undiagnosed
    visit_hcm_clinic_delayed:
      label: "Patients worked up at HCM clinic (delay up to 1 year)"
      transitions:
        - dest: diagnosed_delayed
          if: has_hcm # Assume perfect split between diagnosed and undiagnosed
        - dest: undiagnosed
    diagnosed_delayed:
      label: "Diagnosed (delay up to 1 year)"
      transitions:
        - dest: true_positive_delayed
    true_positive_delayed:
      label: "Delayed True Positive"
      utility: 1

And we'll also need to add some variables to account for the ECG and Echo screen sensitivity and specificity.

.. code-block:: yaml

  variables:
    # ... existing variables ...
    # ... new variables ...
    patient_delayed_screen_idx:
      type: property
      column: patient_delayed_screen_idx # Random index of the patient; used for determining which patients get screened given a capacity constraint. All patients with ``patient_random_idx <= n_patients_delayed_screened`` will be screened.
    ecg_result: # Used for determining if the patient is positive for ECG screening
      type: property
      distribution: uniform
      start: 0
      end: 1
    echo_result: # Used for determining if the patient is positive for Echo screening
      type: property
      distribution: uniform
      start: 0
      end: 1
    n_patients_delayed_screened:
      value: 100 # Capacity constraint on the number of patients worked up at HCM clinic with delay after failing initial HCM triage, aka flex capacity ('F' in the diagram)
    ecg_sensitivity:
      value: 0.95 # Sensitivity of ECG screening ('g' in the diagram)
    ecg_specificity:
      value: 0.95 # Specificity of ECG screening ('f' in the diagram)
    echo_sensitivity:
      value: 0.95 # Sensitivity of Echo screening ('h' in the diagram)
    echo_specificity:
      value: 0.95 # Specificity of Echo screening ('j' in the diagram)

Putting it all together, we get the following config:

.. code-block:: yaml

  metadata:
    name: "HCM (AI-Guided)"
    path_to_properties: "input/hcm/properties.csv"

  variables:
    # Patient properties
    has_hcm:
      type: property
      column: has_hcm # Boolean property that is true if the patient has HCM
    patient_screen_idx:
      type: property
      column: patient_screen_idx # Random index of the patient; used for determining which patients get screened given a capacity constraint. All patients with ``patient_random_idx <= n_patients_screened`` will be screened.
    patient_delayed_screen_idx:
      type: property
      column: patient_delayed_screen_idx # Random index of the patient; used for determining which patients get screened given a capacity constraint. All patients with ``patient_random_idx <= n_patients_delayed_screened`` will be screened.
    ecg_result: # Used for determining if the patient is positive for ECG screening
      type: property
      distribution: uniform
      start: 0
      end: 1
    echo_result: # Used for determining if the patient is positive for Echo screening
      type: property
      distribution: uniform
      start: 0
      end: 1
    # Fixed constants
    hcm_prevalence:
      value: 1/350 # Prevalence of HCM in the population ('p' in the diagram)
    n_patients_screened:
      value: 1_000 # Capacity constraint on the number of patients screened ('C' in the diagram)
    n_patients_delayed_screened:
      value: 100 # Capacity constraint on the number of patients worked up at HCM clinic with delay after failing initial HCM triage, aka flex capacity ('F' in the diagram)
    ecg_sensitivity:
      value: 0.95 # Sensitivity of ECG screening ('g' in the diagram)
    ecg_specificity:
      value: 0.95 # Specificity of ECG screening ('f' in the diagram)
    echo_sensitivity:
      value: 0.95 # Sensitivity of Echo screening ('h' in the diagram)
    echo_specificity:
      value: 0.95 # Specificity of Echo screening ('j' in the diagram)

  states:
    start:
      type: start
      label: "Start"
      transitions:
        - dest: ecg_screening
    ecg_screening:
      label: "ECG screening"
      transitions:
        - dest: echo_screening
          # The patient is marked as POSITIVE for ECG screen if...
          #   Has HCM + ECG result is <= the sensitivity of the ECG screening, or
          #   Does not have HCM + ECG result is >= the specificity of the ECG screening
          if: (has_hcm and (ecg_result <= ecg_sensitivity)) or (not has_hcm and (ecg_result >= ecg_specificity))
        - dest: undiagnosed
    echo_screening:
      label: "Echo screening"
      transitions:
        - dest: hcm_triage
          # The patient is marked as POSITIVE for Echo screen if...
          #   Has HCM + Echo result is <= the sensitivity of the Echo screening, or
          #   Does not have HCM + Echo result is >= the specificity of the Echo screening
          if: (has_hcm and (echo_result <= echo_sensitivity)) or (not has_hcm and (echo_result >= echo_specificity))
        - dest: undiagnosed
    hcm_triage:
      label: "HCM triage"
      transitions:
        - dest: visit_hcm_clinic
          if: patient_screen_idx <= n_patients_screened
        - dest: hcm_flex_waitlist
    visit_hcm_clinic:
      label: "Patients worked up at HCM clinic"
      transitions:
        - dest: diagnosed
          if: has_hcm
        - dest: undiagnosed
    hcm_flex_waitlist:
      label: "Patients not worked up at HCM clinic with minimal delay"
      transitions:
        - dest: visit_hcm_clinic_delayed
          if: patient_delayed_screen_idx <= n_patients_delayed_screened
        - dest: undiagnosed
    visit_hcm_clinic_delayed:
      label: "Patients worked up at HCM clinic (delay up to 1 year)"
      transitions:
        - dest: diagnosed_delayed
          if: has_hcm # Assume perfect split between diagnosed and undiagnosed
        - dest: undiagnosed
    diagnosed:
      label: "Diagnosed"
      transitions:
        - dest: untreated
    diagnosed_delayed:
      label: "Diagnosed (delay up to 1 year)"
      transitions:
        - dest: true_positive_delayed
    undiagnosed:
      label: "Undiagnosed"
      transitions:
        - dest: untreated
    true_positive_delayed:
      label: "Delayed True Positive"
      utility: 1
    untreated:
      label: "Untreated"
      utility: 0
    true_positive:
      label: "True Positive"
      utility: 1
    false_positive:
      label: "False Positive"
      utility: -1
    true_negative:
      label: "True Negative"
      utility: 1
    false_negative:
      label: "False Negative"
      utility: -1