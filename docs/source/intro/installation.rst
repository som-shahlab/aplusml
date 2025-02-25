Installation
============

.. toctree::
   :maxdepth: 2
   :caption: Installation

⚡️ Basic
------------------

You can install **aplusml** using pip:

.. code-block:: bash

   pip install aplusml

*Note:* You may also need to install **graphviz** to enable workflow visualization:

.. code-block:: bash

   brew install graphviz


🧑‍💻 Development
---------------

For development, clone the repository and install it in editable mode:

.. code-block:: bash

   # Clone repo
   git clone https://github.com/som-shahlab/aplusml.git
   cd aplusml
   # Create virtual env
   conda create -n aplus python=3.10 -y
   conda activate aplus
   # Install dependencies
   poetry install

