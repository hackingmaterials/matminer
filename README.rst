========
MatMiner
========

MatMiner is an open-source Python library for performing data mining and analysis in the field of Materials Science. It is meant to make accessible the application of state-of-the-art statistical and machine learning algorithms to materials science data with just a *few* lines of code. It is currently in development, however it is a **working code**.

Citing MatMiner
--------

We are currently in the process of revising the first draft of a paper on MatMiner, so be on the lookout for a paper with the following cite structure::

    Bajaj, S.; Jain, A.; [MatMiner]
    
Example notebooks
-----------------

A few examples demonstrating some of the features available in MatMiner have been created in the form of Jupyter notebooks: 

1. Get all experimentally measured band gaps of PbTe from Citrine's database: `Notebook <https://github.com/hackingmaterials/MatMiner/blob/master/example_notebooks/get_Citrine_experimental_bandgaps_PbTe.ipynb>`_

2. Compare and plot experimentally band gaps from Citrine with computed values from the Materials Project: `Notebook <https://github.com/hackingmaterials/MatMiner/blob/master/example_notebooks/experiment_vs_computed_bandgap.ipynb>`_

3. Use machine learning models to fit and predict band gaps by using MatMiner's tools to retrieve computed band gaps and descriptors from the Materials Project, and composition descriptors from pymatgen: `Notebook <https://github.com/hackingmaterials/MatMiner/blob/master/example_notebooks/machine_learning_to_predict_bandgap.ipynb>`_

You can also use the `Binder <http://mybinder.org/>`_ service (in beta) to launch an interactive notebook upon a click. Click the button below to open the tree structure of this repository and navigate to the folder **example_notebooks** in the current working directory to use/edit the above notebooks right away! To open/run/edit other notebooks, go to "File->Open" within the page and navigate to the notebook of your choice. 

.. image:: http://mybinder.org/badge.svg 
   :target: http://mybinder.org/repo/hackingmaterials/MatMiner  
   
Installation
--------

There are a couple of quick and easy ways to install MatMiner:-

- **Quick install**

(Note: this may not install the latest changes to MatMiner. To install the version with the latest commits, skip to the next steps)

For a quick install of MatMiner, and all of its dependencies, simply run the command in a bash terminal:

.. code-block:: bash

    $ pip install matminer

or, to install MatMiner in your user $HOME folder, run the command:

.. code-block:: bash

    $ pip install matminer --user 

One way to obtain :code:`pip` if not already installed is through :code:`conda`, which is useful when you are working with many python packages and want to use separate configuration settings and environment for each package. You can then install MatMiner and packages required by it in its own environment. Some useful links are `here <https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/>`_ and `here <http://conda.pydata.org/docs/using/index.html>`_.

- **Install in developmental mode**

To install the full and latest source of the MatMiner code in developmental mode, along with its important dependencies, clone the Git source in a folder of your choosing by entering the following command:

.. code-block:: bash

    $ git clone https://github.com/hackingmaterials/MatMiner.git

and then entering the cloned repository/folder to install in developer mode:

.. code-block:: bash

    $ cd MatMiner
    $ python setup.py develop
    
Depending on how many of the required dependencies were already installed on your system, you will see a few or many warnings, but everything should be installed successfully.

Overview
--------

Below is a general workflow that shows the different tools and utilities available within MatMiner, and how they could be implemented with one another, as well as external libraries, in your own materials data analysis study. 

|
.. image:: https://github.com/hackingmaterials/MatMiner/blob/master/Flowchart.png
|
|

It basically includes tools and utilities that make it easier to,

- Retrieve data from the biggest materials databases, such as the `Materials Project <https://www.materialsproject.org/>`_ and `Citrine's databases <https://citrination.com/>`_, in a Pandas dataframe format
- Decorate the dataframe with composition, structural, and/or band structure descriptors/features
- Solve for and add thermal and mechanical properties to the dataframe
 
