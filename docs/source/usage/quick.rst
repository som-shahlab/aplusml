Quick Start
============

.. toctree::
   :maxdepth: 2
   :caption: Quick Start

TODO

Transitions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Patients move between states through **transitions**. There are three types of transitions:

  * **Deterministic** – A single, predefined transition per state.  
  * **Probabilistic** – Transitions occur with specified probabilities (e.g., 30% "high-risk", 70% "low-risk").  
  * **Conditional** – Transitions based on Boolean expressions evaluating patient-level or system-level conditions (e.g., a gene therapy succeeding only for patients with a specific mutation).  

Utilities and Performance Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
APLUS supports tracking **multiple utility measures** to evaluate workflow effectiveness:

  * **Time-based** (e.g., patient length-of-stay)  
  * **Clinical** (e.g., patient outcomes)  
  * **Financial** (e.g., cost impact)  
  * **Resource-related** (e.g., staff utilization)  

Utilities can be **conditioned on expressions**, allowing individual-level utility analyses based on patient properties [29]. These values must be derived from literature, expert consultation, or financial modeling [32].  

Temporal and Resource Modeling
""""""""""""""""""""""""""""""""

**Time Duration:** Each state and transition can have an associated **time duration**, defining the number of simulation time steps before progression. Example:  

  * A hospital workflow where post-surgery patients stay in a "rest" state before daily evaluations.  

**Resource Constraints:** Transitions can alter **system-level resources** via **resource deltas**, simulating real-world capacity limits. Example:  

  * A transition directing a patient to an MRI scan may decrease **MRI capacity by 1**.  

