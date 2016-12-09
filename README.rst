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
