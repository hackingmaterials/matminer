.. title:: Installing matminer
.. _installation tutorial:



===================
Installing matminer
===================

Matminer requires Python 3.6+.

There are a couple of quick and easy ways to install matminer (see also some **tips** below):

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


Tips
----

* Make sure you are using Python 3.6 or higher
* If you have trouble with the installation of a component library (sympy, pymatgen, mdf-forge, etc.), you can try to run ``pip install <<component>>`` or (if you are using `Anaconda <https://www.anaconda.com/distribution/>`_) ``conda install <<component>>`` first, and then re-try the installation.

    - For example, installing pymatgen on a Windows platform is easiest with Anaconda via ``conda install -c conda-forge pymatgen``.

* If you still have trouble, open up a a ticket on our `forum <https://discuss.matsci.org/c/matminer>`_  describing your problem in full (including your system specifications, Python version information, and input/output log). There is a good likelihood that someone else is running into the same issue, and by posting it on the forum we can help make the documentation clearer and smoother.
