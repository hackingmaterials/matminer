========
matminer
========

matminer is an open-source Python library for performing data mining and analysis in the field of Materials Science. It is meant to make accessible the application of state-of-the-art statistical and machine learning algorithms to materials science data with just a *few* lines of code. It is currently in development, however it is a **working code**.

--------
Citing matminer
--------

We are currently in the process of writing a paper on matminer - we will update the citation information once it is submitted.

-----------------
Example notebooks
-----------------

A few examples demonstrating some of the features available in matminer have been created in the form of Jupyter notebooks:

(Note: the Jupyter (Binder) links below are recommended as Github does not render interactive Javascript code or images.)

1. Get all experimentally measured band gaps of PbTe from Citrine's database: `Jupyter <http://mybinder.org/repo/hackingmaterials/matminer/notebooks/example_notebooks/get_Citrine_experimental_bandgaps_PbTe.ipynb>`_  `Github <https://github.com/hackingmaterials/matminer/blob/master/example_notebooks/get_Citrine_experimental_bandgaps_PbTe.ipynb>`_

2. Compare and plot experimentally band gaps from Citrine with computed values from the Materials Project: `Jupyter <http://mybinder.org/repo/hackingmaterials/matminer/notebooks/example_notebooks/experiment_vs_computed_bandgap.ipynb>`_  `Github <https://github.com/hackingmaterials/matminer/blob/master/example_notebooks/experiment_vs_computed_bandgap.ipynb>`_

3. Train and predict band gaps using matminer's tools to retrieve computed band gaps and descriptors from the Materials Project, and composition descriptors from pymatgen: `Jupyter <http://mybinder.org/repo/hackingmaterials/matminer/notebooks/example_notebooks/machine_learning_to_predict_bandgap.ipynb>`_  `Github <https://github.com/hackingmaterials/matminer/blob/master/example_notebooks/machine_learning_to_predict_bandgap.ipynb>`_

4. Training and predict bulk moduli using matminer's tools to retrieve computed bulk moduli and descriptors from the Materials Project, and composition descriptors from pymatgen: `Jupyter <http://mybinder.org/repo/hackingmaterials/matminer/notebooks/example_notebooks/machine_learning_to_predict_BulkModulus.ipynb>`_ `Github <https://github.com/hackingmaterials/matminer/blob/master/example_notebooks/machine_learning_to_predict_BulkModulus.ipynb>`_

|
You can also use the `Binder <http://mybinder.org/>`_ service (in beta) to launch an interactive notebook upon a click. Click the button below to open the tree structure of this repository and navigate to the folder **example_notebooks** in the current working directory to use/edit the above notebooks right away! To open/run/edit other notebooks, go to "File->Open" within the page and navigate to the notebook of your choice. 

.. image:: http://mybinder.org/badge.svg 
   :target: http://mybinder.org/repo/hackingmaterials/matminer

--------
Installation
--------

There are a couple of quick and easy ways to install matminer:-

- **Quick install**

(Note: this may not install the latest changes to matminer. To install the version with the latest commits, skip to the next steps)

For a quick install of matminer, and all of its dependencies, simply run the command in a bash terminal:

.. code-block:: bash

    $ pip install matminer

or, to install matminer in your user $HOME folder, run the command:

.. code-block:: bash

    $ pip install matminer --user 

One way to obtain :code:`pip` if not already installed is through :code:`conda`, which is useful when you are working with many python packages and want to use separate configuration settings and environment for each package. You can then install matminer and packages required by it in its own environment. Some useful links are `here <https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/>`_ and `here <http://conda.pydata.org/docs/using/index.html>`_.

- **Install in developmental mode**

To install the full and latest source of the matminer code in developmental mode, along with its important dependencies, clone the Git source in a folder of your choosing by entering the following command:

.. code-block:: bash

    $ git clone https://github.com/hackingmaterials/matminer.git

and then entering the cloned repository/folder to install in developer mode:

.. code-block:: bash

    $ cd matminer
    $ python setup.py develop
    
Depending on how many of the required dependencies were already installed on your system, you will see a few or many warnings, but everything should be installed successfully.

- **Solutions to *some* errors that may be encountered during installation**

#. *Error*:-

.. code-block:: bash

   ============================================================================
                        * The following required packages can not be built:
                        * freetype, png
   error: Setup script exited with 1

*Solution*:-

.. code-block:: bash

    $ # On Mac OS X, install brew if not available
    $ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    $ # Install "freetype" and "pkg-config" using brew (on Mac OS X; for other systems, see http://stackoverflow.com/a/20533455)
    $ brew install freetype
    $ brew install pkg-config

#. *Error*:-

.. code-block:: bash

    error: Setup script exited with error: library dfftpack has Fortran sources but no Fortran compiler found

*Solution*:-

.. code-block:: bash

    $ # On Mac OS X:
    $ brew install gcc
    
--------
Overview
--------

Below is a general workflow that shows the different tools and utilities available within matminer, and how they could be implemented with each other, as well as with external libraries, in your own materials data mining/analysis study.

|
.. image:: https://github.com/hackingmaterials/matminer/blob/master/Flowchart.png
   :align: center
|
|

Here's a brief description of the available tools (please find implementation examples in a dedicated section elsewhere in this document):

Data retrieval tools
--------------------

- Retrieve data from the biggest materials databases, such as the Materials Project, Citrine, and MPDS (PAULING FILE) databases, in a Pandas dataframe format

The `MPDataRetrieval <https://github.com/hackingmaterials/matminer/blob/master/matminer/data_retrieval/retrieve_MP.py>`_, `CitrineDataRetrieval <https://github.com/hackingmaterials/matminer/blob/master/matminer/data_retrieval/retrieve_Citrine.py>`_, and `MPDSDataRetrieval <https://github.com/hackingmaterials/matminer/blob/master/matminer/data_retrieval/retrieve_MPDS.py>`_ classes can be used to retrieve data from the biggest open-source materials database collections of the `Materials Project <https://www.materialsproject.org/>`_ and `Citrine Informatics <https://citrination.com/>`_, as well as from the partially opened database `MPDS (PAULING FILE) <https://mpds.io/>`_, respectively, in a `Pandas <http://pandas.pydata.org/>`_ dataframe format. The data contained in these databases are a variety of material properties, obtained in-house or from other external databases, that are either calculated, measured from experiments, or learned from trained algorithms. The :code:`get_dataframe` method of these classes executes the data retrieval by searching the respective database using user-specified filters, such as compound/material, property type, etc , extracting the selected data in a JSON/dictionary format through the API, parsing it and output the result to a Pandas dataframe with columns as properties/features measured or calculated and rows as data points.

For example, to compare experimental and computed band gaps of Si, one can employ the following lines of code:

.. code-block:: python

   from matminer.data_retrieval.retrieve_Citrine import CitrineDataRetrieval
   from matminer.data_retrieval.retrieve_MP import MPDataRetrieval

   df_citrine = CitrineDataRetrieval().get_dataframe(formula='Si', property='band gap', 
                                                  data_type='EXPERIMENTAL')   
   df_mp = MPDataRetrieval().get_dataframe(criteria='Si', properties=['band_gap'])
   
`MongoDataRetrieval <https://github.com/hackingmaterials/matminer/blob/master/matminer/data_retrieval/retrieve_MongoDB.py>`_ is another data retrieval tool developed that allows for the parsing of any `MongoDB <https://www.mongodb.com/>`_ collection (which follows a flexible JSON schema), into a Pandas dataframe that has a format similar to the output dataframe from the above data retrieval tools. The arguments of the :code:`get_dataframe` method allow to utilize MongoDB's rich and powerful query/aggregation syntax structure. More information on customization of queries can be found in the `MongoDB documentation <https://docs.mongodb.com/manual/>`_.


Data descriptor tools
----------------

- Decorate the dataframe with composition, structural, and/or band structure descriptors/features

In this module of the matminer library, we have developed utilities to help describe the material by their composition or structure, and represent them in a numeric format such that they are readily usable as features in a data analysis study to predict a target value.

The :code:`get_pymatgen_descriptor` function is used to encode a material's composition using tabulated elemental properties in the `pymatgen <http://pymatgen.org/_modules/pymatgen/core/periodic_table.html>`_ library. There are about 50 attributes available in the pymatgen library for most elements in the periodic table, some of which include electronegativity, atomic numbers, atomic masses, sound velocity, boiling point, etc. The :code:`get_pymatgen_descriptor` function takes as input a material composition and name of the desired property, and returns a list of floating point property values for each atom in that composition. This list can than be fed into a statistical function to obtain a single heuristic quantity representative of the entire composition. The following code block shows a few 
descriptors that can be obtained for LiFePO\ :sub:`4`:

.. code-block:: python
      
   from matminer.descriptors.composition_features import get_pymatgen_descriptor
   import numpy as np
      
   avg_mass = np.mean(get_pymatgen_descriptor('LiFePO4', 'atomic_mass'))    # Average atomic mass
   std_num = np.std(get_pymatgen_descriptor('LiFePO4', 'Z'))    # Standard deviation of atomic numbers
   range_elect = max(get_pymatgen_descriptor('LiFePO4', 'X')) - \
              min(get_pymatgen_descriptor('LiFePO4', 'X'))      # Maximum difference in electronegativity

The function :code:`get_magpie_descriptor` operates in a similar way and obtains its data from the tables accumulated in the `Magpie repository <https://bitbucket.org/wolverton/magpie>`_, some of which are sourced from elemental data compiled by Mathematica (more information can be found `here <https://reference.wolfram.com/language/ref/ElementData.html>`_). Some properties that don't overlap with the pymatgen library include heat capacity, enthalpy of fusion of elements at melting points, pseudopotential radii, etc. 

Some other descriptors that can be obtained from matminer include:

#. Composition descriptors

   #. Cohesive energy
   #. Band center
   
#. Structural descriptors

   #. Packing fraction
   #. Volume per site
   #. Radial and electronic radial distribution functions
   #. Smallest relative distance to nearest neighbor
   #. Order parameters (structure-motif specific as well as unspecific)

#. Band-structure descriptors

   #. Branch point energy
   #. Absolute band positions

#. Mechanical properties

   #. Thermal stress
   #. Fracture toughness
   #. Brittleness index
   #. Critical stress
   #. bulk/elastic, rigid, and shear moduli
   #. bulk modulus from coordination number
   #. Vicker's hardness
   #. Lame's first parameter
   #. p-wave modulus
   #. Sound velocity from elastic constants
   #. Steady-state and maximum allowed heatflow
   #. Strain energy release rate
   
#. Thermal condutivity models

   #. Cahill model
   #. Clarke model
   #. Callaway model
   #. Slack model
   #. Keyes model
   

Note on MPDS (PAULING FILE) database
----------------
Using MPDS generally requires commercial license. However, these data are provided for free:

- All data for compounds containing both Ag and K
- All cell parameters - temperature diagrams
- All cell parameters - pressure diagrams

In case of questions, please, contact `MPDS <https://mpds.io/>`_ directly.

Plotting tools
----------------

- Plot data from either arrays or dataframes using either `Plotly <https://plot.ly/>`_ or `matplotlib <http://matplotlib.org/>`_

In the figrecipes module of the matminer library, we have developed utilities that wrap around two popular plotting libraries, Plotly and matplotlib to produce various types of plots that plot data from either arrays or dataframes. The Plotly part of this module contains classes/functions that wrap around its Python API library and follows its JSON schema. The figrecipes module is aimed at making it easy for the user to create plots from their data using just a few lines of code, utilizing the wide and flexible functionality of Plotly and matplotlib, while at the same time sheilding the complexities involved. 

A few examples demonstrating usage can be found in the notebook hosted on `Jupyter <http://mybinder.org/repo/hackingmaterials/matminer/notebooks/matminer/figrecipes/plotly/examples/plotly_examples.ipynb>`_ and `Github <https://github.com/hackingmaterials/FigRecipes/blob/master/figrecipes/plotly/examples/plotly_examples.ipynb>`_
