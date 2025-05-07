API Reference
=============

.. toctree::
   :maxdepth: 2
   :caption: Contents:

The API is organized as follows:

* ``Config``: Configuration for the simulation. This :class:`~aplusml.config.Config` class can be used as an alternative to specifying a YAML config file.
* ``Models``: Models used by the simulation engine. Namely, :class:`~aplusml.models.Patient`, :class:`~aplusml.models.State`, :class:`~aplusml.models.Transition`, :class:`~aplusml.models.History`, and :class:`~aplusml.models.Utility`.
* ``Simulation``: Simulation engine. This :class:`~aplusml.sim.Simulation` class is responsible for running patients through the workflow (i.e. running the actual simulation).
* ``Run``: Helper functions for running a set of simulations.
* ``Draw``: Helper functions for drawing the workflow.
* ``Parse``: Helper functions for parsing the YAML config file.

----------

Config
^^^^^^^^^^^^^^
.. automodule:: aplusml.config
   :members:
   :special-members: __init__
   :exclude-members: model_computed_fields, model_config, model_fields

----------

Models
^^^^^^^^^^^^^^
.. automodule:: aplusml.models
   :members:
   :special-members: __init__

----------

Simulation
^^^^^^^^^^^^^^^^^^
.. automodule:: aplusml.sim
   :members:
   :undoc-members:
   :special-members: __init__

----------

Run
^^^^^^^^^^^
.. automodule:: aplusml.run
   :members:
   :undoc-members:
   :special-members: __init__

----------

Draw
^^^^^^^^^^^^^
.. automodule:: aplusml.draw
   :members:
   :undoc-members:
   :private-members:
   :special-members: __init__

----------

Plot
^^^^^^^^^^^^
.. automodule:: aplusml.plot
   :members:
   :undoc-members:
   :special-members: __init__

----------

Parse
^^^^^^^^^^^^^
.. automodule:: aplusml.parse
   :members:
   :undoc-members:
   :special-members: __init__
