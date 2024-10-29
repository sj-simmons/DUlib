DUlib
=====

Machine and Deep Learning tools.

============
Installation
============

.. code-block::

    pip3 install DUlib --user

-----
Notes
-----

DUlib assumes that you have installed your desired version (e.g., cpu vs. gpu)
of the `torch <https://pypi.org/project/torch/>`_ package as well as the `torchvision <https://pypi.org/project/torchvision/>`_ package.

Similarly, a modern version of `matplotlib <https://pypi.org/project/matplotlib/>`_ is
assumed to have been installed on your system.  Matplotlib is required solely for (optional) display
of a real-time graph while training and for various demos; additionally, a running X server is required.

To upgrade to the newest release of DUlib in the presence of a previously installed one, issue the command::

    pip3 install -U DUlib --user

See the `upstream repo <https://github.com/sj-simmons/DUlib>`_ for detailed installation and trouble-shooting
notes as well as release information.
