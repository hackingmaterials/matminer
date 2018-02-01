.. title:: Installing matminer
.. _installation tutorial:



===================
Installing matminer
===================

Matminer requires Python 3.x (Python 2.x might work sporadically, but you may get errors for certain features and is **unsupported**. We really recommend you upgrade to Py3.x).

There are a couple of quick and easy ways to install matminer:

Install and update via pip
--------------------------

If you have installed pip, simply run the command in a bash terminal:

.. code-block:: bash

    $ pip install matminer

or, to install matminer in your user home folder, run the command:

.. code-block:: bash

    $ pip install matminer --user

To update matminer, simply type ``pip install --upgrade matminer``.

Install in development mode
-----------------------------

To install from the latest source of the matminer code in developmental mode, clone the Git source:

.. code-block:: bash

    $ git clone https://github.com/hackingmaterials/matminer.git

and then enter the cloned repository/folder to install in developer mode:

.. code-block:: bash

    $ cd matminer
    $ python setup.py develop

To update matminer, enter your cloned folder and type ``git pull`` followed by ``python setup.py develop``.


Troubleshooting/Issues
----------------------

Having issues installing? Open up an issue on our `forum <https://groups.google.com/forum/#!forum/matminer>`_  describing your problem in full (including your system specifications, Python version information, and input/output log).