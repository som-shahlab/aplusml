# APLUS

**A** **P**ython **L**ibrary for **U**sefulness **S**imulations of Machine Learning Models in Healthcare

----

![Graphical Abstract](img/graphical%20abstract.png)

## Motivation

APLUS is a general simulation framework for systematically conducting usefulness assessments of ML models in clinical workflows.

It aims to quantitatively answer the question: *If I use this ML model to guide this clinical workflow, will the benefits outweigh the costs, and by how much?*

## Installation

1) Run the following command to install **APLUS**:

```bash
git clone https://github.com/som-shahlab/aplus.git
cd aplus
pip3 install -r requirements.txt
```

2. Install **graphviz** by [downloading it here](https://graphviz.org/download/). If you're on Mac with `homebrew`, simply run:
```
brew install graphviz
```

## Tutorials

We showcase APLUS on two use cases: PAD and ACP.

### PAD

The code used to generate the figures in our paper is located in the `tutorials/` directory in `pad.ipynb`. This notebook loads de-identified patient data from Stanford Hospital, which can be provided upon request.

The workflows analyzed can be found in the `workflows/` folder. The doctor-driven workflow is in `pad_doctor.yaml` while the nurse-driven workflow is in `pad_nurse.yaml`

### ACP

The code used to replicate the findings of [Jung et al. 2021](https://pubmed.ncbi.nlm.nih.gov/33355350/) can be found in the `tutorials/` directory in `acp_jung_replication.ipynb`. This notebook loads de-identified patient data from Stanford Hospital, which can be provided upon request.

The workflows analyzed can be found in the `workflows/` folder in `acp_jung_replication.yaml`

## File Structure

Core APLUS module (listed in the order that they should be used):
1. `parse.py` - Functions to parse YAML files into Python objects for use in `sim.py`
1. `sim.py` - Core simulation engine which progresses patients through the desired clinical workflow
1. `run.py` - Wrapper functions around `sim.py` to help run/track multiple simulations
1. `plot.py` - Plotting functions

Supporting files:
* `pad.py` - Helper functions for PAD-specific workflow analysis
* `tutorials/` - Contains Jupyter notebooks that demonstrate how to use APLUS
    * `pad.ipynb` - Demonstrates how to use APLUS to simulate the novel PAD workflow described in the paper
    * `acp_jung_replication.ipynb` - Demonstrates how to use APLUS to replicate the plots of [Jung et al. 2021](https://pubmed.ncbi.nlm.nih.gov/33355350/)
* `workflows/` - Contains YAML files that define the workflows analyzed in the paper
    * `pad_doctor.yaml` - The doctor-driven PAD workflow
    * `pad_nurse.yaml` - The nurse-driven PAD workflow
    * `acp_jung_replication.yaml` - The exact same ACP workflow analyzed in [Jung et al. 2021](https://pubmed.ncbi.nlm.nih.gov/33355350/)
* `tests/` - Contains unit tests for the APLUS framework
    * `run_tests.py` - Script to run all unit tests
    * `test_*.py` - Tests for each module
    * `test*.yaml` - Workflow YAML files for each corresponding test
    * `utils.py` - Utility functions for testing
* `input/` - Contains input data fed into the simulation
* `output/` - Contains output data from the simulations (this is useful for caching results so you don't have to re-run time-consuming simulations)

## Tests

The file `tests/run_tests.py` runs all of the `test[d].py` files in the `tests/` directory. Each `test[d].py` file has a corresponding `test[d].yaml` file that serves as its input.

To run tests:
```
cd tests
python3 run_tests.py
```
