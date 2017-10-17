.. title:: MatMiner Changlog


==================
matminer Changelog
==================

**v0.1.4**

* add a band structure featurizer (A. Faghaninia)
* add global structure featurizer (A. Jain)
* improve CoulombMatrix, SineCoulombMatrix, and OrbitalFieldMatrix featurizers (K. Bostrom)
* fix some code structure / interfaces (A. Faghaninia, A. Jain)
* bug fixes (A. Jain, A. Faghaninia, L. Ward)
* code cleanup (A. Jain)
* doc updates (A. Dunn, A. Jain, K. Bostrom)

**v0.1.3**

* remove git-lfs
* updated CSV data sets (K. Bostrom)
* better oxidation state determination in multiple composition descriptors
* refactor structure descriptors
* multiple fixes to cohesive energy
* fixes to data loaders
* fix complex Mongo retrieval queries, better logic for query projections
* more unit tests
* enforce lower case feature names
* sort data by atomic number not electronegativity in data getters, this will avoid pernicious behavior
* many minor cleanups, bug fixes, and consistency fixes


**v0.1.2**

* Several new structure fingerprint methods (L. Ward, K. Bostrom)
* Refactor structure descriptors into new OOP style (N. Zimmermann)
* move large files to git-lfs (K. Bostrom, A. Jain)
* update example notebooks to new style
* misc. cleanups and bug fixes

**v0.1.1**

* refactor and redesign of codebase to be more OOP (J. Chen, L. Ward)
* Py3 compatibility (K. Mathew)
* Element fraction feature (A. Aggarwal)
* misc fixes / improvements (A. Jain, J. Chen, L. Ward, K. Mathew, J. Frost)

**v0.1.0**

* Add MPDS data retrieval (E. Blokhin)
* Add partial RDF descriptor (L. Ward)
* Add local environment motif descriptors (N. Zimmermann)
* fix misc. bugs and installation issues (A. Dunn, S. Bajaj, L. Ward)

For changelog before v0.1.0, consult the git history of matminer.
