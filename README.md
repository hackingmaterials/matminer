# <img alt="matminer" src="docs_rst/_static/matminer_logo_small.png" width="300">

matminer is a library for performing data mining in the field of materials science.

- **[Website (including documentation)](https://hackingmaterials.github.io/matminer/)**
- **[Examples Repository](https://github.com/hackingmaterials/matminer_examples)** 
- **[Help/Support Forum](https://matsci.org/c/matminer/16)** 
- **[Source Repository](https://github.com/hackingmaterials/matminer)** 

matminer supports Python 3.8+.

#### Related packages:

- If you like matminer, you might also try [automatminer](https://github.com/hackingmaterials/automatminer).
- If you are interested in furthering development of datasets in matminer, you may be interested in [matbench](https://github.com/hackingmaterials/matbench).
- If you are looking for figrecipes, it is now in its [own repo](https://github.com/hackingmaterials/figrecipes).


#### Citation

If you find matminer useful, please encourage its development by citing the following paper in your research:
```
Ward, L., Dunn, A., Faghaninia, A., Zimmermann, N. E. R., Bajaj, S., Wang, Q.,
Montoya, J. H., Chen, J., Bystrom, K., Dylla, M., Chard, K., Asta, M., Persson,
K., Snyder, G. J., Foster, I., Jain, A., Matminer: An open source toolkit for
materials data mining. Comput. Mater. Sci. 152, 60-69 (2018).
```

Matminer helps users apply methods and data sets developed by the community. Please also cite the original sources, as this will add clarity to your article and credit the original authors:

- If you use one or more **datasets** accessed through matminer, check the dataset metadata info for relevant citations on the original datasets.
- If you use one or more **data retrieval methods**, check ``citations()`` method of the data retrieval class. This method will provide a list of BibTeX-formatted citations for that featurizer, making it easy to keep track of and cite the original publications.
- If you use one or more **featurizers**, please take advantage of the ```citations()``` function present for every featurizer in matminer. 
