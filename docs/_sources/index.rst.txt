.. title:: matminer (Materials Data Mining)

.. image:: _static/matminer_logo_small.png
   :alt: matminer logo


========
matminer
========

matminer is an open-source Python library for performing data mining and analysis in the field of Materials Science. It is meant to make accessible the application of state-of-the-art statistical and machine learning algorithms to materials science data with just a *few* lines of code. It is currently in development, however it is a `working code <https://github.com/hackingmaterials/matminer>`_.


-------------------
Installing matminer
-------------------

Install matminer by following our short :doc:`installation tutorial. </installation>`

--------
Overview
--------

Below is a general workflow that shows the different tools and utilities available within matminer, and how they could be implemented with each other, as well as with external libraries, in your own materials data mining/analysis study.

|
.. image:: _static/Flowchart.png
   :align: center
|
|

Take a tour of matminer's features by scrolling down!

--------------------
Data retrieval tools
--------------------

Retrieve data from the biggest materials databases, such as the Materials Project and Citrine's databases, in a Pandas dataframe format
_______________________________________________________________________________________________________________________________________

The `MPDataRetrieval <https://github.com/hackingmaterials/matminer/blob/master/matminer/data_retrieval/retrieve_MP.py>`_ and `CitrineDataRetrieval <https://github.com/hackingmaterials/matminer/blob/master/matminer/data_retrieval/retrieve_Citrine.py>`_ classes can be used to retrieve data from the biggest open-source materials database collections of the `Materials Project <https://www.materialsproject.org/>`_ and `Citrine Informatics <https://citrination.com/>`_, respectively, in a `Pandas <http://pandas.pydata.org/>`_ dataframe format. The data contained in these databases are a variety of material properties, obtained in-house or from other external databases, that are either calculated, measured from experiments, or learned from trained algorithms. The :code:`get_dataframe` method of these classes executes the data retrieval by searching the respective database using user-specified filters, such as compound/material, property type, etc , extracting the selected data in a JSON/dictionary format through the API, parsing it and output the result to a Pandas dataframe with columns as properties/features measured or calculated and rows as data points.

For example, to compare experimental and computed band gaps of Si, one can employ the following lines of code:

.. code-block:: python

   from matminer.data_retrieval.retrieve_Citrine import CitrineDataRetrieval
   from matminer.data_retrieval.retrieve_MP import MPDataRetrieval

   df_citrine = CitrineDataRetrieval().get_dataframe(formula='Si', property='band gap', data_type='EXPERIMENTAL')
   df_mp = MPDataRetrieval().get_dataframe(criteria='Si', properties=['band_gap'])
   
`MongoDataRetrieval <https://github.com/hackingmaterials/matminer/blob/master/matminer/data_retrieval/retrieve_MongoDB.py>`_ is another data retrieval tool developed that allows for the parsing of any `MongoDB <https://www.mongodb.com/>`_ collection (which follows a flexible JSON schema), into a Pandas dataframe that has a format similar to the output dataframe from the above data retrieval tools. The arguments of the :code:`get_dataframe` method allow to utilize MongoDB's rich and powerful query/aggregation syntax structure. More information on customization of queries can be found in the `MongoDB documentation <https://docs.mongodb.com/manual/>`_.


---------------------
Data descriptor tools
---------------------

Decorate the dataframe with composition, structural, and/or band structure descriptors/features
_______________________________________________________________________________________________

We have developed utilities to help describe a material from its composition or structure, and represent them in number format such that they are readily usable as features.

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

Other descriptors provided by matminer can be found in the `Github repo. <https://github.com/hackingmaterials/matminer/tree/master/matminer/featurizers>`_
   
--------------
Plotting tools
--------------

Plot data from either arrays or dataframes using either `Plotly <https://plot.ly/>`_ or `matplotlib <http://matplotlib.org/>`_ with figrecipes
______________________________________________________________________________________________________________________________________________

In the figrecipes module of the matminer library, we have developed utilities that make it easier and faster to plot common figures with Plotly and matplotlib. The figrecipes module is aimed at making it easy for the user to create plots from their data using just a few lines of code, utilizing the wide and flexible functionality of Plotly and matplotlib, while at the same time sheilding the complexities involved.

The Plotly module contains the :code:`PlotlyFig` class that wraps around Plotly's Python API and follows its JSON schema. The matplotlib module contains plotting wrapper classes for each kind of popular plot, including XY-scatter plots and heat maps.

A few examples demonstrating usage can be found in the notebook hosted on `Github <https://github.com/hackingmaterials/FigRecipes/blob/master/figrecipes/plotly/examples/plotly_examples.ipynb>`_. *Note: these examples may be out of date*.


--------
Examples
--------
Check out some examples of how to use matminer!

1. :doc:`Use matminer and sklearn to train/predict bulk moduli. </example_bulkmod>`

.. image:: _static/example_bulkmod_rf.png
   :scale: 50

2. Get all experimentally measured band gaps of PbTe from Citrine's database (`Jupyter Notebook <https://gist.github.com/saurabh02/cf37de8ab77505a05e1bec952f0cb0c3>`_)

3. Compare and plot experimentally band gaps from Citrine with computed values from the Materials Project (`Jupyter Notebook <https://gist.github.com/saurabh02/8f7727b2ed1f95d2a40fdefd0a90bec0>`_)

4. Use matminer and sklearn to train/predict band gaps. (`Jupyter Notebook <https://gist.github.com/saurabh02/b0296747064599ad2a6ab69ddc64eb92>`_)

5. Analyze Uranium-Oxygen bond lengths from gathered from the MPDS database. (`Jupyter Notebook <https://gist.github.com/blokhin/a9eddca705aa6d54552bc8b6d7bddbb8>`_)


---------------
Citing matminer
---------------

We are currently in the process of writing a paper on matminer - we will update the citation information once it is submitted.


---------
Changelog
---------

Check out our full changelog :doc:`here. </changelog>`

-----------------------------
Contributions and Bug Reports
-----------------------------
Want to see something added or changed? Here's a few ways you can!

* Help us improve the documentation. Tell us where you got 'stuck' and improve the install process for everyone.
* Let us know about areas of the code that are difficult to understand or use.
* Contribute code! Fork our `Github repo <https://github.com/hackingmaterials/matminer>`_ and make a pull request.

Submit all questions and contact to the `Google group <https://groups.google.com/forum/#!forum/matminer>`_

A full list of contributors can be found :doc:`here. </contributors>`

