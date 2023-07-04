.. title:: MatMiner Changelog


==================
matminer Changelog
==================

.. caution:: Starting v0.6.6 onwards, the changelog is no longer maintained. Please check the Github commit log for a record of changes.

**v0.6.5**

* update BV parameters to 2020 version; the existing 2016 version included errors including Rh4+
* Fix matbench dataset urls

**v0.6.4**

* make BaseFeaturizer an ABC with abstractmethods (J. Riebesell)
* default ewald summation to be per atom; change default feature name to reflect this (A. Jain)
* add feature descriptions for magpie (A. Dunn)
* correct yield strengths being accidentally in GPa instead of MPa (A. Dunn)
* minor fixes / updates (B. Krull, A. Dunn, A. Ganose, A. Jain)

**v0.6.3**

* add IntersticeDistribution featurizer (Q. Wang)
* change Dimensionality featurizer to be more accurate (A. Ganose)
* add CohesiveEnergyMP featurizer that gets MP cohesive energy from MP Rester (A. Jain)
* default mp dataretrieval to decode mp entities by default (A. Dunn)
* update dependencies and tests (A. Ganose)
* misc fixes / documentation updates (Q. Wang, A. Dunn, A. Jain, L. Ward, S.P. Ong, A. Ganose)


**v0.6.2**

* Update forum to Discourse link (A. Ganose)
* Add StructuralComplexity featurizer (K. Muraoka)
* Resolve optional requirements problems, update sklearn requirement (A. Dunn)
* Update references to DScribe (A. Dunn)

**v0.6.1**

This version was skipped due to an upload issue

**v0.6.0**

* Ensure Yang omega is never NaN in YangSolidSolution featurizer (L. Ward)
* More complete BV sum table, some code cleanups (N. Wagner)

**v0.5.9**

* add Meredig composition featurizer (A. Trewartha)
* update / fix Miedema model parameters (Q. Wang)
* update code for latest pymatgen (A. Dunn)

**v0.5.8**

* optimizations for Global Instability Index featurizer (N. Wagner, L. Ward)

**v0.5.7**

* remove SOAP normalization flag (K. Murakoa)
* fix precheck - Miedema and YangSolidSolution (Q. Wang)
* improvements to Miedema - default structure types, docs (Q. Wang)
* fix CGCNN optimizer (Q. Wang)

**v0.5.6**

* add Global Instability Index featurizer (N. Wagner)
* fix Citrine code in docs (D. Nishikawa)
* fix Bond Valence data (K. Muraoka)
* fix MPDataRetrieval (A. Ganose)
* update citation / implementor list for SOAP (A. Jain)
* misc bug fixes (K. Muraoka, C. Legaspi)

**v0.5.5**

* Add a precheck() and precheck_dataframe() function that can be used to quickly see if a featurizer is likely to give NaN values (A. Dunn)
* Add MEGNET 1Neuron element embeddings (A. Dunn)
* fix inplace setting (S. Cherfaoui, A. Dunn, A. Ganose)
* add a conversion featurizer to get a structure from composition using MP most stable structure (A. Jain)
* misc code cleanups (A. Jain, A. Dunn)

**v0.5.4**

* add elementproperty source name to feature labels (A. Dunn)
* update Citrine API key detection logic (matSciMalcolm + A. Jain)
* misc. fixes (A. Dunn)

**v0.5.3**

* fix typo bug that got introduced in 0.5.2 pypi release

**v0.5.2**

* better flattening for ColoumbMatrix featurizers, making them more usable (A. Dunn)
* SOAP featurizer using the dscribe package (A. Dunn)
* DosAsymmetry featurizer (M. Dylla)

**v0.5.1**

* AFLOW data retrieval (M. Dylla)
* SiteDOS featurizer (M. Dylla)
* fix various testing (A. Dunn, M. Dylla, L. Ward)

**v0.5.0**

* fix for Py3.7 and pytorch (L. Ward)

**v0.4.9**

* fix PIP setup of JARVIS data files (A. Dunn)
* some test configuration fixes (A. Dunn)

**v0.4.8**

* CGCNN featurizer / model (Q. Wang, T. Xie)
* Text-mined element embedding featurizer (A. Dunn)
* add Brgoch data set (D. Dopp)
* add quartile to PropertyStats (A. Dunn)
* minor fixes, improvements (A. Dunn, A. Jain)

**v0.4.6**

* Jarvis CFID descriptors (A. Dunn, K. Choudhary)
* Allow oxi conversion featurizers to return original object (A. Ganose)
* better contribution docs (A. Dunn, A. Jain)

**v0.4.5**

* fix for missing data set loader file (D. Dopp)
* fix MDF unit tests (L. Ward)

**v0.4.4**

.. warning:: Data set loaders may not work properly due to a missing file in this release

* Further revamp data set loaders and management (D. Dopp)
* Better default chunksize for multiprocessing should improve performance (L. Ward)
* Improve oxidation state featurizer (A. Dunn)

**v0.4.3**

* Revamped test / example data loader classes (D. Dopp, A. Ganose, A. Dunn)
* Add chunksize support to improve performance of dataframe featurization (A. Ganose)
* Improve performance of BandCenter with large coefficients (A. Faghaninia)
* Revamp of MultiFeaturizer (A. Ganose)
* Custom progress bar for running in notebook (A. Ganose)
* Improved multi-index for conversion featurizerse (A. Ganose)
* Minor fixes / improvements (D. Dopp, A. Ganose, A. Faghaninia)

**v0.4.2**

* Refactor conversion utils to be featurizers for consistency and parallelism (A. Ganose)
* Average Bond Length and Bond Angle implementations (A. Rui, L. Ward)
* Add ability to serialize dataframes as JSON with MontyEncoder (A. Ganose)
* support added for fractional compositions in AtomicOrbitals (M. Dylla)
* Add ability to flatten OFM (A. Dunn)
* updates to FunctionFeaturizer (J. Montoya)
* Various bugfixes (L. Ward, A. Ganose)

**v0.4.1**

* Better elemental properties for Magpie features (L. Ward)
* Improvements to Seko representation (L. Ward)
* Some bugfixes for multiplefeaturizer and compatibility with progress bars (L. Ward, A. Dunn)
* More intuitive input arguments for featurize_many (L. Ward)
* Bugfixes for BOOP features (L. Ward, A. Thompson)

**v0.4.0**

* Progressbar for featurizers (A. Dunn)
* Add BOOP features (L. Ward)
* Add Seko features, including more lookuip tables for MagpieData and elemental property site features + covariance, skew, kurtosis (L. Ward)
* New scheme for GRDF/AFS bin functions (L. Ward)
* misc fixes (A. Dunn., L. Ward)

**v0.3.9**

* BandEdge renamed to Hybridization, gives smoother featurizations (M. Dylla, A. Faghaninia)
* Add hoverinfo option for many plots (A. Dunn)
* minor fixes (A. Faghaninia)

**v0.3.8**

.. warning:: This is an unsupported / aborted release


**v0.3.7**

* faster implementation of GaussianSymmFunc (L. Ward)
* more resilient Yang and AtomicPackingEfficiency (L. Ward)
* some fixes for PRDF featurizer (A. Faghaninia)
* add *.tsv files to package_data, should fix Miedema PyPI install (A. Faghaninia)

**v0.3.6**

* Improve MPDataRetrieval to serialize objects (A. Faghaninia)
* Some fixes to GDRF and AFS (L. Williams, M. Dylla)
* Some fixes for Ewald (A. Faghaninia)
* improve error messages (A. Jain)

**v0.3.5**

* some tools for sklearn Pipeline integration (J. Brenneck)
* ability to add a chemical descriptor to CNFingerprint (N. Zimmermann, hat tip to S. Dwaraknath and A. Jain)
* add phase diagram-like "triangle" plot (A. Faghaninia)
* add harmonic mean (holder_mean::-1) to PropertyStats (A. Jain)

**v0.3.4**

* add XRDPowderPattern featurizer (A. Jain)
* add multi-index support for featurizers (A. Dunn)
* add BandEdge featurizer (A. Faghaninia)
* better labels support in xy plots + debugs and cleanups (A. Faghaninia)
* deprecate CrystalSiteFingerprint
* remove  a few old and unused site OP functions/methods (A. Jain)
* doc improvements (A. Faghaninia)
* bug fixes, minor code improvements, etc. (N. Zimmermann, A. Dunn, Q. Wang, A. Faghaninia)

**v0.3.3**

* add StackedFeaturizer (L. Ward)
* changes to reference energies in BranchPointEnergy featurizer (A. Faghaninia)
* doc improvements (A. Dunn)

**v0.3.2**

* Major overhaul / redesign of data retrieval classes for consistency (A. Faghaninia, A. Dunn)
* Updates / redesign of function featurizer (J. Montoya)
* Add Yang's solid solution features (L. Ward)
* Add cluster packing efficiency features (L. Ward)
* update to MDF data retrieval (L. Ward)
* update to Citrine data retrieval for new pycc (S. Bajaj)
* Branch point energy takes into account symmetry (A. Faghaninia)
* minor code and doc updates (A. Jain, A. Faghaninia)

**v0.3.1**

* add caching for featurizers (L. Ward)
* add CrystalNNFingerprint (A. Jain)
* some x-y plot updates (A. Faghaninia)
* speedup to chemenv featurizer (D. Waroquiers)
* minor code cleanups, bugfixes (A. Dunn, L. Ward, N. Zimmermann, A. Jain)

**v0.3.0**

* add structural heterogeneity features (L. Ward)
* add maximum packing efficiency feature (L. Ward)
* add chemical ordering features (L. Ward)
* New BagofBonds based on original paper, old featurizer now BondFractions (A. Dunn)
* add DopingFermi featurizer (A. Faghaninia, A. Jain)
* shortcut for getting composition features from structure (L. Ward)
* fix static mode output in PlotlyFig (A. Dunn)
* some misc Figrecipes updates (A. Dunn)
* add fit_featurize method to base (A. Dunn)
* minor cleanups, doc updates and new docs (A. Jain, L. Ward, A. Dunn)

**v0.2.9**

* fix pymatgen dep (A. Jain)

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
