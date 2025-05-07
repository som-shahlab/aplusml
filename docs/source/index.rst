.. APLUS ML documentation master file, created by
   sphinx-quickstart on Mon Feb 24 13:52:47 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to APLUS ML's documentation!
====================================
.. image:: _static/graphical_abstract.png
   :width: 700
   :alt: APLUS graphical abstract

🧑‍💻 Installation
---------------------

First, install the **aplusml** package:

.. code-block:: bash

   pip install aplusml

Second, install **graphviz** to enable workflow visualization:

.. code-block:: bash

   brew install graphviz

Please see the :doc:`intro/background` for a high-level conceptual overview of APLUS, or jump straight to :doc:`usage/tutorial` for a step-by-step walkthrough of using APLUS to model a clinical workflow.

🚀 Quick Start
---------------

.. code-block:: python

   import aplusml

   # Create config
   config = aplusml.config.Config(
      metadata = aplusml.config.ConfigMetadata(
         name = 'My Simulation',
      ),
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
                  value = 1,
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
   
   # Create simulation
   sim = aplusml.Simulation.create_from_config(config)
   
   # Run simulation
   patients = sim.create_patients_for_simulation([ aplusml.Patient(id=1, start_timestep=0) ])
   patients = sim.run(patients)
   
   # Visualize first patient's trajectory through workflow
   print(patients[0].history)

Key Features
------------

APLUS ML (**A** **P**\ ython **L**\ ibrary for **U**\ sefulness **S**\ imulations of **M**\ achine **L**\ earning Models) is a simulation framework for conducting usefulness assessments of machine learning models in workflows, as originally published in this `2023 JBI paper <https://www.sciencedirect.com/science/article/pii/S1532046423000400?via%3Dihub>`_.

It aims to quantitatively answer the question: *If I use this ML model within this workflow, will the benefits outweigh the costs, and by how much?*

* Easy-to-use simulation framework
* Comprehensive model evaluation tools
* Extensible architecture for custom simulations
* Built-in visualization capabilities

Documentation
=============

.. toctree::
   :maxdepth: 2
   :caption: 🚦 Introduction

   intro/installation
   intro/background

.. toctree::
   :maxdepth: 2
   :caption: 📚 User Guide

   usage/quick
   usage/models
   usage/tutorial_hello
   usage/tutorial_pad
   usage/tutorial_hcm

.. toctree::
   :maxdepth: 2
   :caption: 📖 API

   api/config
   api/api

Citation
===========

.. code-block:: bibtex

   @article{wornow2023aplus,
     title={APLUS: A Python Library for Usefulness Simulations of Machine Learning Models in Healthcare},
     author={Wornow, Michael and Ross, Elsie Gyang and Callahan, Alison and Shah, Nigam H},
     journal={Journal of Biomedical Informatics},
     pages={104319},
     year={2023},
     publisher={Elsevier}
   }