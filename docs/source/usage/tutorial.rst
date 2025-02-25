Tutorial
=========

.. toctree::
   :maxdepth: 2
   :caption: Tutorial

In this section, we present a case study of conducting a novel usefulness assessment via APLUS of ML models for the early detection of PAD.

🏥 Clinical Motivation
---------------------

PAD is a chronic condition which occurs when the arteries in a patient's limbs are constricted by atherosclerosis, thereby reducing blood flow.
A total of 8–12 million people in the US have PAD. Left untreated, PAD is associated with a higher risk of mortality, serious cardiovascular events, and lower quality of life. This costs the US healthcare system over $21 billion annually.

Despite these risks, PAD is often missed by healthcare providers as roughly half of all PAD patients are asymptomatic. 

Thus, developing methods for the early detection of PAD can help reduce the burden of PAD on the US healthcare system.

🤖 ML Model
-----------
`Ghanzouri et al. 2022 <https://www.nature.com/articles/s41598-022-17180-5>`_ proposed three ML models to classify patients for PAD based on EHR data: a deep learning model, a random forest, and a logistic regression with respective AUROCs of 0.96, 0.91 and 0.81. Each model assigns a probabilistic risk score to each visiting patient which indicates their likelihood of having PAD. 

We use APLUS to conduct a usefulness assessment on incorporating this model into clinical decision making at Stanford Health Care to quantify the benefits of deploying this model in practice.

👩‍⚕️ Workflow
--------------

.. image:: ../_static/pad_workflows.png
   :width: 700
   :alt: Workflows for PAD screening

Based on interviews with practitioners, we identified two workflows to consider:

1. A **nurse-driven** workflow which assumes the existence of a centralized team of nurses reviewing the PAD model's predictions for all patients visiting their clinic each day. This nursing staff decides which patients to directly refer to the specialist, thus cutting out any intermediate steps with a non-specialist physician. The main constraints on this workflow are the capacity of the nursing staff and the specialist's schedule. 
2. A **doctor-driven** workflow which assumes that the PAD model's predictions appear as a real-time alert in a patient's EHR during their visit to the clinic. If the attending physician notices this alert, she can choose to either ignore the alert or act on it. We assume that physicians ignore alerts at random. The main constraints on this workflow are the probability that the attending physician reads the alert (previous studies have shown that up to 96 % of alerts are overridden [54], [55], [56]) and the specialist's schedule.

In both workflows, we assume the existence of a downstream cardiovascular specialist who evaluates patients after they are referred by a doctor or nurse. 
We assume that the specialist has a set capacity for how many patients she can see per day. 
However, once a patient reaches the specialist, we assume that the specialist makes the optimal treatment decision for that patient.

We model 3 possible end outcomes for patients: **Untreated**, **Medication**, or **Surgery.** 

An important distinction between the **nurse-driven** and **doctor-driven** workflows is that the nurse-driven workflow is centralized whereas the doctor-driven workflow is decentralized.
In other words, the nurse-driven workflow batches together all model predictions for each day before patients are chosen for follow-up, while in the doctor-driven workflow each doctor decides whether to act on a PAD alert immediately upon receipt of the alert (independent from the decisions of other doctors). 
Thus, we consider two possible strategies that the nurses can leverage for processing this batch of predictions: (1) ranked screening, in which the nursing staff follows up with the K patients with the highest PAD risk scores; or (2) thresholded screening, in which a random subset of patients are selected from the batch of predictions whose predicted PAD risk score exceeds some cutoff threshold.

The corresponding YAML workflow specification files are available in the APLUS Github repository.

⚖️ Utilities
------------

The utility of each of these outcomes depends on the ground truth PAD status for a specific patient. 

For example, **Untreated** is the best option for patients without PAD but has the largest cost for patients with PAD. 
**Medication** is the ideal outcome for patients with moderate PAD but is undesirable for patients without PAD. 
**Surgery** is the costliest outcome for all patients, but the relatively best option for patients with severe PAD. 

We combined clinician input with utility estimates from `Itoga et al. 2018 <https://pubmed.ncbi.nlm.nih.gov/29930023/>`_ to define the utilities associated with the end outcomes of each workflow in terms of a multiplier on remaining years living to reflect quality-adjustment on lifespan [32]. 
Given that a healthy patient with no PAD has a baseline utility of 1, the utilities for different combinations of PAD severity and treatment are shown in the table below:

.. list-table:: Utility Values by PAD Severity and Treatment
   :header-rows: 1
   :widths: 25 25 25 25

   * - 
     - Untreated
     - Medication
     - Surgery
   * - No PAD
     - 1.0
     - 0.95
     - 0.7
   * - Moderate PAD
     - 0.85
     - 0.9
     - N/A
   * - Severe PAD
     - 0.6
     - N/A
     - 0.68

💽 Data
-------
We acquired a dataset of 4,452 patients who had both ground-truth labels of PAD diagnosis and risk score predictions from all three ML models developed in Ghanzouri et al. 2022.

We simulated 500 consecutive days of patient visits. We sampled :math:`k \sim \text{Poisson}(35)` to determine the number of patients visiting on a given day, then sampled :math:`k` patients with replacement from our dataset.

🅰️ APLUS Specification
----------------------

Parameters
^^^^^^^^
Based on clinician interviews and a literature review, we identified the following parameters for our simulation:

* The ABI score :math:`a` of... 

    * Patients with PAD: :math:`a \sim \text{Normal}(0.65, 0.15)`
    * Patients without PAD: :math:`a \sim \text{Normal}(1.09, 0.11)`. 

* Cutoff threshold between PAD and no PAD: 
    
    * :math:`a = 0.90`

* For the doctor-driven workflow...

    * Alerts generated for all patients with a model-generated PAD risk score ≥ 0.5

Constraints
^^^^^^^^^^^^^

For the nurse-driven workflow...

    * :math:`k` = Number of patients per day that the nursing team can follow-up with for an ABI test
    * :math:`c` = Number of patients per day that the specialist can see

For the doctor-driven workflow...

    * :math:`p` = Probability that a PAD alert is read
    * :math:`c` = Number of patients per day that the specialist can see

Baselines
^^^^^^^^
We evaluated each PAD model's utility relative to three baselines: 

* **Treat None** -- We simply predict a PAD risk score of 0 for all patients; 
* **Treat All** -- We predict a PAD risk score of 1 for all patients; 
* **Optimistic** -- There were no workflow constraints or resource limits

Concretely, we measured each model's expected utility achieved per patient above the **Treat None** baseline as a percentage of the utility achieved under the **Optimistic** scenario. In other words, we measured how much of the total possible utility gained from using a model was actually achieved under each workflow's constraints, relative to simply doing nothing. 

📊 Results
----------

TODO
