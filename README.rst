========
MatMiner
========

MatMiner is an open-source Python library for performing data mining and analysis in the field of Materials Science. It is meant to make accessible the application of state-of-the-art statistical and machine learning algorithms to materials science data with just a *few* lines of code. It is currently in development, however it is a **working code**.

Citing MatMiner
--------

We are currently in the process of revising the first draft of a paper on MatMiner, so be on the lookout for a paper with the following cite structure::

    Bajaj, S.; Jain, A.; [MatMiner]
    
Installation
--------

There are a few ways to install MatMiner:-

- **Quick install**

(Beware: this may not install the latest changes to MatMiner. To install the version with the latest commits, skip to the next steps)

For a quick install of MatMiner, and all of its dependencies, simply run the command in a bash terminal:

.. code-block:: bash

    $ pip install matminer

or, to install MatMiner in your user $HOME folder, run the command:

.. code-block:: bash

    $ pip install matminer --user 

- **Install in developmental mode**

To install the full and latest source of the MatMiner code in developmental mode, along with its important dependencies, clone the Git source in a folder of your choosing by entering the following command:

.. code-block:: bash

    $ git clone https://github.com/hackingmaterials/MatMiner.git

and then entering the cloned repository/folder to install in developer mode:

.. code-block:: bash

    $ cd MatMiner
    $ python setup.py develop
    
Depending on how many of the required dependencies were already installed on your system, you will see a few or many warnings, but everything should be installed successfully.

- **Install in virtual environments**

If you are working with many python packages and want to use separate configuration settings for each package, it is generally recommended to create and use a separate environment for each package. You can use either of the following two options to create a virtual environment, and then install MatMiner in it using any of the above two installation options.

1. Conda-based install

You can install conda using:

.. code-block:: bash

    $ pip install conda
    
(Note: depending on your operating system and other settings, you may also need to install other packages like *ruamel.yaml*, *pycosat*, etc.)

You could also instead download and install an operating-system specific version of conda from `here <http://conda.pydata.org/miniconda.html>`_. For Windows, make sure it is the Miniconda3 installer, and simply double-click the exe file. For Linux or Mac, run the following in a bash terminal:

.. code-block:: bash

    # If Mac
    $ bash Miniconda3-latest-MacOSX-x86_64.sh

    # If Linux
    $ bash Miniconda3-latest-Linux-x86_64.sh

Note: you may need to open a new terminal window after this step in order for the environmental variables added by conda to be loaded.

To check if conda is successfully installed and in your *PATH*:

.. code-block:: bash

    $ conda -V
    conda 4.2.7

To create a virtual environemt for your project with Python 2.x (MatMiner is currently not supported for Python 3.x):

.. code-block:: bash

    $ conda create --name [virtualenv_name] python=2

where, *[virtualenv_name]* is the name of the virtual environment. Press :code:`y` to proceed with installation. The installed environment can be activated using:

.. code-block:: bash

    $ source activate [virtualenv_name]
    
Once activated, MatMiner, or any other package, can be installed using any of the above options of :code:`pip install` or :code:`git clone` followed by :code:`python setup.py develop`.

2. Using virtualenv

*virtualenv* creates a folder that contains all the necessary executables to use the packages that your Python project may require. It can be installed via:

.. code-block:: bash

    $ pip install virtualenv
    
To create a virtual environemt for MatMiner (or any other project) :

.. code-block:: bash

    $ mkdir [project_folder]
    $ cd [project_folder]
    $ virtualenv [virtualenv_name]

where *[project_folder]* and *[virtualenv_name]* are names of the project folder containing the virtual environment, and could be for example, *matminer_project*, *matminer*. This will create a folder named *[virtualenv_name]* in the current directory, and will contain executable files for Python and the pip library. The virtual environemt can be activated using:

.. code-block:: bash

    $ source [virtualenv_name]/bin/activate
    
and deactivated using:

.. code-block:: bash

    $ deactivate

When activated, the pip library can be used to install MatMiner (or any other package) using again one of the above two options :code:`pip install` or :code:`git clone` followed by :code:`python setup.py develop`.

Overview
--------

It includes tools and utilities that make it easier to,

- Retrieve data from the biggest materials databases, such as the `Materials Project <https://www.materialsproject.org/>`_ and `Citrine's databases <https://citrination.com/>`_, in a Pandas dataframe format
- Decorate the dataframe with composition, structural, and/or band structure descriptors/features
- Solve for and add thermal and mechanical properties to the dataframe

Example notebooks
-----------------

A few examples demonstrating the features of the code in this repository have been added in the form of ipython notebooks. You can also use the `Binder <http://mybinder.org/>`_ service (in beta) to launch an interactive notebook upon a click. Click the button below to open the tree structure of this repository and navigate to matminer/data_retrieval/example_notebooks to use/edit the notebook right away!

.. image:: http://mybinder.org/badge.svg 
   :target: http://mybinder.org/repo/hackingmaterials/MatMiner   
