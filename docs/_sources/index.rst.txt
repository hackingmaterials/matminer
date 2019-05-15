.. title:: matminer (Materials Data Mining)

.. image:: _static/matminer_logo_small.png
   :alt: matminer logo


========
matminer
========

matminer is a Python library for data mining the properties of materials. It contains routines for obtaining data on materials properties from various databases, featurizing complex materials attributes (e.g., composition, crystal structure, band structure) into physically-relevant numerical quantities, and analyzing the results of data mining.

matminer works with the `pandas <https://pandas.pydata.org>`_ data format in order to make various downstream machine learning libraries and tools available to materials science applications.

matminer is `open source <https://github.com/hackingmaterials/matminer>`_ via a BSD-style license.


-------------------
Installing matminer
-------------------

To install matminer, follow the short :doc:`installation tutorial. </installation>`

--------
Overview
--------

Matminer makes it easy to:

* **obtain materials data from various sources** into the `pandas <https://pandas.pydata.org>`_ data format. Through pandas, matminer enables professional-level data manipulation and analysis capabilities for materials data.
* **transform and featurize complex materials attributes into numerical descriptors for data mining.** For example, matminer can turn a composition such as "Fe3O4" into arrays of numbers representing things like average electronegativity or difference in ionic radii of the substituent elements. Matminer also contains sophisticated crystal structure and site featurizers (e.g., obtaining the coordination number or local environment of atoms in the structure) as well as featurizers for complex materials data such as band structures and density of states. All of these various featurizers are available under a consistent interface, making it easy to try different types of materials descriptors for an analysis and to transform materials science objects into physically-relevant numbers for data mining. A full :doc:`Table of Featurizers</featurizer_summary>` is available.
* **perform data mining on materials**. Although matminer itself does not contain implementations of machine learning algorithms, it makes it easy to prepare and transform data sets for use with standard data mining packages such as `scikit-learn <http://scikit-learn.org>`_. See our examples for more details.
* **generate interactive plots** through an interface to the `plotly <https://plot.ly>`_ visualization package.


A general workflow and overview of matminer's capabilities is presented below:

|
.. image:: _static/Flowchart.png
   :align: center
   :width: 1000px
   :alt: Flow chart of matminer features
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

   df_citrine = CitrineDataRetrieval().get_dataframe(criteria='Si', properties=['band_gap'])
   df_mp = MPDataRetrieval().get_dataframe(criteria='Si', properties=['band_gap'])

`MongoDataRetrieval <https://github.com/hackingmaterials/matminer/blob/master/matminer/data_retrieval/retrieve_MongoDB.py>`_ is another data retrieval tool developed that allows for the parsing of any `MongoDB <https://www.mongodb.com/>`_ collection (which follows a flexible JSON schema), into a Pandas dataframe that has a format similar to the output dataframe from the above data retrieval tools. The arguments of the :code:`get_dataframe` method allow to utilize MongoDB's rich and powerful query/aggregation syntax structure. More information on customization of queries can be found in the `MongoDB documentation <https://docs.mongodb.com/manual/>`_.


Access ready-made datasets for exploratory analysis, benchmarking, and testing without ever leaving the Python interpreter
____________________________________________________________________________________________________________________________

The datasets module provides an ever growing collection of materials science datasets that have been collected, formatted as pandas dataframes, and made available through a unified interface.

Loading a dataset as a pandas dataframe is as simple as:

.. code-block:: python

    from matminer.datasets import load_dataset

    df = load_dataset("jarvis_dft_3d")

Or use the dataset specific convenience loader to access operations common to that dataset:

.. code-block:: python

    from matminer.datasets.convenience_loaders import load_jarvis_dft_3d

    df = load_jarvis_dft_3d(drop_nan_columns=["bulk modulus"])

See :doc:`the dataset summary page </dataset_summary>` for a comprehensive summary of
datasets available within matminer. If you would like to contribute a dataset to matminer's
repository see :doc:`the dataset addition guide </dataset_addition_guide>`.



---------------------
Data descriptor tools
---------------------

Decorate the dataframe with :doc:`composition, structural, and/or band structure descriptors/features </featurizer_summary>`
____________________________________________________________________________________________________________________________

We have developed utilities to help describe a material from its composition or structure, and represent them in number format such that they are readily usable as features.

|
.. image:: _static/featurizer_diagram.png
   :align: center
   :width: 1200px
   :alt: matminer featurizers
|
|

For now, check out the examples below to see how to use the descriptor functionality, or tour our :doc:`Table of Featurizers. </featurizer_summary>`

--------------
Plotting tools
--------------

Plot data from either arrays or dataframes using `Plotly <https://plot.ly/>`_ with figrecipes
_____________________________________________________________________________________________

In the figrecipes module of the matminer library, we have developed utilities that make it easier and faster to plot common figures with Plotly. The figrecipes module is aimed at making it easy for the user to create plots from their data using just a few lines of code, utilizing the wide and flexible functionality of Plotly, while at the same time sheilding the complexities involved.
Check out an example code and figure generated with figrecipes:

.. code-block:: python

   from matminer import PlotlyFig
   from matminer.datasets import load_dataset
   df = load_dataset("elastic_tensor_2015")
   pf = PlotlyFig(df, y_title='Bulk Modulus (GPa)', x_title='Shear Modulus (GPa)', filename='bulk_shear_moduli')
   pf.xy(('G_VRH', 'K_VRH'), labels='material_id', colors='poisson_ratio', colorscale='Picnic', limits={'x': (0, 300)})

This code generates the following figure from the matminer elastic dataset dataframe.

.. raw:: html


    <iframe src="_static/bulk_shear_moduli.html" height="1000px" width=90%" align="center" frameBorder="0">Browser not compatible.</iframe>

The Plotly module contains the :code:`PlotlyFig` class that wraps around Plotly's Python API and follows its JSON schema. Check out the examples below to see how to use the plotting functionality!

--------
Examples
--------

Check out some examples of how to use matminer!

0. Examples index. (`Jupyter Notebook <https://nbviewer.jupyter.org/github/hackingmaterials/matminer_examples/blob/master/matminer_examples/index.ipynb>`_)

1. Use matminer and scikit-learn to create a model that predicts bulk modulus of materials. (`Jupyter Notebook <https://nbviewer.jupyter.org/github/hackingmaterials/matminer_examples/blob/master/matminer_examples/machine_learning-nb/bulk_modulus.ipynb>`_)

2. Compare and plot experimentally band gaps from Citrine with computed values from the Materials Project (`Jupyter Notebook <https://nbviewer.jupyter.org/github/hackingmaterials/matminer_examples/blob/master/matminer_examples/data_retrieval-nb/expt_vs_comp_bandgap.ipynb>`_)

3. Compare and plot U-O bond lengths in various compounds from the MPDS (`Jupyter Notebook <https://nbviewer.jupyter.org/github/hackingmaterials/matminer_examples/blob/master/matminer_examples/data_retrieval-nb/mpds.ipynb>`_)

4. Retrieve data from various online materials repositories (`Jupyter Notebook <https://nbviewer.jupyter.org/github/hackingmaterials/matminer_examples/blob/master/matminer_examples/data_retrieval-nb/data_retrieval_basics.ipynb>`_)

5. Basic Visualization using FigRecipes (`Jupyter Notebook <https://nbviewer.jupyter.org/github/hackingmaterials/matminer_examples/blob/master/matminer_examples/figrecipes-nb/figrecipes_basics.ipynb>`_)

6. Advanced Visualization (`Jupyter Notebook <https://nbviewer.jupyter.org/github/hackingmaterials/matminer_examples/blob/master/matminer_examples/figrecipes-nb/figrecipes_advanced.ipynb>`_)

7. Many more examples! See the `matminer_examples <https://github.com/hackingmaterials/matminer_examples>`_ repo for details.

---------------
Citing matminer
---------------


If you find matminer useful, please encourage its development by citing the following paper in your research

.. code-block:: text

   Ward, L., Dunn, A., Faghaninia, A., Zimmermann, N. E. R., Bajaj, S., Wang, Q.,
   Montoya, J. H., Chen, J., Bystrom, K., Dylla, M., Chard, K., Asta, M., Persson,
   K., Snyder, G. J., Foster, I., Jain, A., Matminer: An open source toolkit for
   materials data mining. Comput. Mater. Sci. 152, 60-69 (2018).

Matminer helps users apply methods and data sets developed by the community. Please also cite the original sources, as this will add clarity to your article and credit the original authors:

* If you use one or more **data retrieval methods**, check the code documentation on the relevant paper(s) to cite.
* If you use one or more **featurizers**, please take advantage of the ``citations()`` function present for every featurizer in matminer. This function will provide a list of BibTeX-formatted citations for that featurizer, making it easy to keep track of and cite the original publications.

---------
Changelog
---------

Check out our full changelog :doc:`here. </changelog>`

-------------------------
Contributions and Support
-------------------------
Want to see something added or changed? Here's a few ways you can!

* Help us improve the documentation. Tell us where you got 'stuck' and improve the install process for everyone.
* Let us know about areas of the code that are difficult to understand or use.
* Contribute code! Fork our `Github repo <https://github.com/hackingmaterials/matminer>`_ and make a pull request.

Submit all questions and contact to the `Google group <https://groups.google.com/forum/#!forum/matminer>`_

A comprehensive guide to contributions can be found `here. <https://github.com/hackingmaterials/matminer/blob/master/CONTRIBUTING.md>`_

A full list of contributors can be found :doc:`here. </contributors>`

==================
Code documentation
==================
Autogenerated code documentation below:

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


