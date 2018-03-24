.. title:: MatMiner Changlog


==================
matminer Changelog
==================

**v0.2.8**

* new FunctionFeaturizer to combine features into mini functions (J. Montoya)
* updates to PlotlyFig (A. Dunn)
* Update default n_jobs to cpu_count() (A. Dunn)
* test fixes and updates (A. Dunn, N. Zimmermann, J. Montoya)
* move Jupyter notebooks to matminer_examples repo, separate from matminer (J. Montoya)
* add presets for AFS, GRDF featurizes (M. Dylla)
* update CircleCI testing (A. Dunn)
* code cleanups (A. Dunn, A. Jain, J. Montoya)

**v0.2.6**

* modify ChemicalRSO to use fit() method (Q. Wang)
* more updates to FigRecipes (A. Dunn, A. Faghaninia)
* misc code cleanups (M. Dylla, A. Faghaninia, A. Jain, K. Bostrom, Q. Wang)
* fix missing yaml file from package data (A. Jain)

**v0.2.5**

* Major rework of BaseFeaturizer to subclass BaseEstimator/TransformerMixin of sklearn. Allows for support of fit() function needed by many featurizers (L. Ward)
* BaseFeaturizer can return errors as a new column (A. Dunn)
* Clean up data getter signatures (J. Montoya)
* Re-implement PRDF (L. Ward)
* GaussianSymmFunc featurizer (Q. Wang)
* misc code clean up (L. Ward, A. Jain)

**v0.2.4**

* updates to PlotlyFig (A. Dunn, A. Faghaninia)
* adapt to new OP parameters (N. Zimmermann)
* bugfixes, cleanups, doc updates (A. Faghaninia, A. Dunn, Q. Wang, N. Zimmermann, A. Jain)

**v0.2.3**

* MDF data retrieval (J. Montoya)
* new VoronoiFingerprint descriptors (Q. Wang)
* new ChemicalSRO descriptors (Q. Wang)
* bugfixes to featurize_many (A. Dunn)
* minor bug fixes, cleanups, slighly improved docs, etc.

**v0.2.2**

.. warning:: Py2 compatibility is officially dropped in this version. Please upgrade to Python 3.x.

* multiprocessing for pandas dataframes (A. Dunn, L. Ward)
* new CoordinationNumber site featurizer based on NearNeighbor algos (N. Zimmermann)
* update OP fingerprints for latest pymatgen (N. Zimmermann)
* OPStructureFingerprint -> SiteStatsFingerprint that takes in any site fingerprint function (A. Jain)
* Add BondFractions featurizer (A. Dunn)
* multi-index for pandas dataframes (A. Dunn)
* cleanup of formatting for citations, implementors, feature_labels to always be list (N. Zimmermann)
* minor bug fixes, cleanups, slighly improved docs, etc.

**v0.2.1**

* further improvements to test data sets (K. Bystrom)
* new MultiFeaturizer to combine multiple featurizers (L. Ward)

**v0.2.0**

* improvements to test data sets (K. Bystrom)
* new conversion utility functions (A. Jain)
* updated example and removed outdated examples (A. Jain)
* some featurizer internal fixes (A. Faghaninia, M. Dylla, A. Jain)
* minor bugfixes (L. Ward, A. Jain)

**v0.1.9**

* overhaul of data API classes (L. Ward)
* change to oxidation-state dependent classes, now require oxidation set in advance (L. Ward)
* Ewald site and structure energy featurizers (L. Ward)
* AtomicOrbital featurizer (M. Dylla)
* Updates to OP fingerprints based on new bcc renormalization (N. Zimmermann)
* fix to include sample data sets in pip install (A. Jain, K. Bostrom)
* add several utility functions for turning strings to compositions, dicts/jsons to pymatgen objects, and quickly adding oxidation state to structure (A. Jain)
* code cleanups (L. Ward, A. Jain)

**v0.1.8**

* extend Miedema model to ternaries and higher (Q. Wang, A. Faghaninia)
* cleanups/refactor to DOS featurizer (A. Faghaninia)

**v0.1.7**

* lots of code cleanup / refactoring / review, including trimming of unused / moved packages (A. Jain)
* new Chemenv structure fingerprint (N. Zimmermann)
* various updates to BSFeaturizer (A. Faghaninia)
* cleanup / rework of DOSFeaturizer (A. Faghaninia)
* Updated citation for OFM paper (L. Ward)
* CNSiteFingerprint goes to CN=16 by default, includes two presets ("cn" and "ops") (A. Jain)
* stats use double colon instead of double underscore for params (A. Jain)
* Various cleanups to Miedema featurizer (Q. Wang, A. Faghaninia, A. Dunn)


**v0.1.6**

* new CrystalSiteFingerprint and CNSiteFingerprint (A. Jain)
* Miedema model (Q. Wang)
* Voronoi index site fingerprint (Q. Wang)
* updates to CitrineDataRetrieval (S. Bajaj)
* updates to BandStructureFeaturizer (A. Faghaninia)
* allow featurize_dataframe() to ignore errors (A. Dunn)
* some patches of DOSFeaturizer (A. Jain)

**v0.1.5**

* new Site and Structure fingerprints based on order parameters (N. Zimmermann)
* DOSFeaturizer (M. Dylla)
* Structure fingerprint can do cations/anions only (A. Jain)
* include the degeneracy of the CBM/VBM in BandFeaturizer (A. Faghaninia)
* fixes / updates to CitrineDataRetrieval (S. Bajaj)
* more property stats (L. Ward)
* fixes to AGNIFingerprint (L. Ward)
* FigRecipes cleanup (A. Dunn)
* updated examples, docs (A. Dunn)
* various bugfixes, code cleanup (A. Jain)

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
