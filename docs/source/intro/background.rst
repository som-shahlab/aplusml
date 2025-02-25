Background
===========

.. toctree::
   :maxdepth: 2
   :caption: Background

APLUS ML (**A** **P**\ ython **L**\ ibrary for **U**\ sefulness **S**\ imulations of **M**\ achine **L**\ earning Models) is a simulation framework for conducting usefulness assessments of machine learning models in workflows, as originally published in this `2023 JBI paper <https://www.sciencedirect.com/science/article/pii/S1532046423000400?via%3Dihub>`_.

It aims to quantitatively answer the question: *If I use this ML model within this workflow, will the benefits outweigh the costs, and by how much?*

.. image:: ../_static/graphical_abstract.png
   :width: 700
   :alt: APLUS graphical abstract

**APLUS** was originally developed for clinical workflows in healthcare settings. However, APLUS is a applicable to any workflow that involves a model making decisions on a stream of datapoints, and we encourage contributors from any domain to use and extend APLUS.


Motivation
----------

Despite rapid advancements in machine learning (ML) for healthcare, model deployment remains limited as traditional ML evaluation metrics (e.g., AUROC, F1 Score) fail to account for the complexities of real-world workflows such as staffing constraints, resource limitations, treatment heterogeneity, and variable patient flow.

These operational conditions can greatly distort the realized impact of introducing an ML model into a clinical setting.

**APLUS** addresses this problem by evaluating ML models under realistic workflow constraints via simulation.

Components
-----------

Below, we provide a high-level overview of the **three components** of APLUS. We provide a more formal specification of the concepts underlying APLUS on the :doc:`specification` page.

1. Simulation Engine
^^^^^^^^^^^^^^^^^^^^^^^
APLUS uses discrete-event simulation to model workflows.

Each patient's journey through the workflow is represented as an ordered sequence of states over evenly-spaced time steps, which enables modeling cycles and resource-dependent prioritization (e.g., Stage IV cancer patients receiving preferential access to treatment over Stage II patients). 

Transitions between states can be determined probabilistically or via conditional expressions, which enables modeling heterogeneous treatment effects, patient subpopulations, and system-level resource constraints.

APLUS outputs a structured dictionary that contains the full history of the simulation (e.g. states, transitions, achieved utilities, etc.) to enable arbitrary downstream analyses.

2. Workflow Specification Language
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
APLUS represents workflows as a **state machine** specified via a **YAML** config file. The config consists of three sections:  

1. **Metadata** - Essential initialization details, such as:

    * Workflow name  
    * Paths to required files (e.g., CSV files with model predictions)  
    * Column mappings for tabular data (e.g. which column in a CSV corresponds to patient IDs)

2. **Variables** – Parameters of the simulation. There are four types of variables:

    * **Simulation-Level Variables**: Tracked by the simulation engine itself and measure the progression of time within the simulation. (e.g. the number of timesteps the simulation has run, the duration of time that a patient has been in a state, etc.).  
    * **Patient-Level Properties**: Unique, individual-level properties associated with each patient(e.g., age, cancer stage, ML model predictions).  
    * **System-Level Resources**: Attributes of the overall system shared across all patients (e.g., MRI availability, budget, specialist capacity).  
    * **Constants**: Fixed values. These can be any primitive Python type (integer, float, string, or boolean) or basic Python data structure (list, dict, set).

3. **States** – The steps of the workflow. There are three types of states:

    * **Start**: Where all patients begin their journey through the workflow. There is always exactly 1 start state.
    * **Intermediate**: Transition points within the workflow. There can be 0+ intermediate states.
    * **End**: Terminal states, which indicate the completion of the workflow. There must be 1+ end states.

3. Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
APLUS has a set of predefined visualizations for workflows which are generated using **Graphviz**.