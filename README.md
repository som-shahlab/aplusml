# APLUS

**A** **P**ython **L**ibrary for **U**sefulness **S**imulations of Machine Learning Models in Healthcare

----

APLUS is a general simulation framework for systematically conducting usefulness assessments of ML models in clinical workflows.

It aims to quantitatively answer the question: *If I use this ML model to guide this clinical workflow, will the benefits outweigh the costs, and by how much?*

## Examples

### PAD

The code used to generate the figures in our paper is located in the `tutorials/` directory in `pad.ipynb`. This notebook loads de-identified patient data from Stanford Hospital, which can be provided upon request.

The workflows analyzed can be found in the `workflows/` folder. The doctor-driven workflow is in `pad_doctor.yaml` while the nurse-driven workflow is in `pad_nurse.yaml`

### ACP

The code used to replicate the findings of Jung et al. 2021 can be found in the `tutorials/` directory in `acp_jung_replication.ipynb`. This notebook loads de-identified patient data from Stanford Hospital, which can be provided upon request.

The workflows analyzed can be found in the `workflows/` folder in `acp_jung_replication.yaml`
