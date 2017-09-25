.. title:: Installing matminer
.. _installation tutorial:



===================
Installing matminer
===================

There are a couple of quick and easy ways to install matminer:


Quick install
-------------

(Note: this may not install the latest changes to matminer. To install the version with the latest commits, install in development mode)

For a quick install of matminer, and all of its dependencies, simply run the command in a bash terminal:

.. code-block:: bash

    $ pip install matminer

or, to install matminer in your user home folder, run the command:

.. code-block:: bash

    $ pip install matminer --user

One way to obtain :code:`pip` if not already installed is through :code:`conda`, which is useful when you are working with many python packages and want to use separate configuration settings and environment for each package. You can then install matminer and packages required by it in its own environment. Some useful links are `here <https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/>`_ and `here <http://conda.pydata.org/docs/using/index.html>`_.

Install in development mode
-----------------------------

To install the full and latest source of the matminer code in developmental mode, along with its important dependencies, clone the Git source in a folder of your choosing by entering the following command:

.. code-block:: bash

    $ git clone https://github.com/hackingmaterials/matminer.git

and then entering the cloned repository/folder to install in developer mode:

.. code-block:: bash

    $ cd matminer
    $ python setup.py develop

Depending on how many of the required dependencies were already installed on your system, you'll see a few or many warnings, but everything should be installed successfully.



Troubleshooting/Issues
----------------------

Having issues installing? Open up an issue on our `Github repo <https://github.com/hackingmaterials/matminer>`_  describing your problem in full, with all your system specifications and python version information.