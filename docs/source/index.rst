.. APLUS ML documentation master file, created by
   sphinx-quickstart on Mon Feb 24 13:52:47 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to APLUS ML's documentation!
====================================

APLUS ML (**A** **P**\ ython **L**\ ibrary for **U**\ sefulness **S**\ imulations of **M**\ achine **L**\ earning Models) is a simulation framework for conducting usefulness assessments of machine learning models in workflows, as originally published in this `2023 JBI paper <https://www.sciencedirect.com/science/article/pii/S1532046423000400?via%3Dihub>`_.

It aims to quantitatively answer the question: *If I use this ML model within this workflow, will the benefits outweigh the costs, and by how much?*

.. image:: _static/graphical_abstract.png
   :width: 700
   :alt: APLUS graphical abstract

Key Features
------------

* Easy-to-use simulation framework
* Comprehensive model evaluation tools
* Extensible architecture for custom simulations
* Built-in visualization capabilities

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
   
   # Create a simulation
   sim = aplusml.Simulation()
   
   # Run evaluation
   results = sim.evaluate_model(model)
   
   # Visualize results
   sim.plot_results(results)

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