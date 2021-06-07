Composition featurizers for elemental data and stoichiometry.
Composition featurizers for elemental data and stoichiometry.
Composition featurizers for elemental data and stoichiometry.
Composition featurizers for elemental data and stoichiometry.
Composition featurizers for orbital data.
Composition featurizers for orbital data.
Composition featurizers for composite features containing more than 1 category of general-purpose data.
Composition featurizers for composite features containing more than 1 category of general-purpose data.
Composition featurizers for determining packing characteristics.
Composition featurizers specialized for use with alloys.
Composition featurizers specialized for use with alloys.
Composition featurizers specialized for use with alloys.
Composition featurizers for compositions with ionic data.
Composition featurizers for compositions with ionic data.
Composition featurizers for compositions with ionic data.
Composition featurizers for compositions with ionic data.
Composition featurizers for thermodynamic properties.
Composition featurizers for thermodynamic properties.
Site featurizers based on bonding.
Site featurizers based on bonding.
Site featurizers based on bonding.
Site featurizers based on local chemical information, rather than geometry alone.
Site featurizers based on local chemical information, rather than geometry alone.
Site featurizers based on local chemical information, rather than geometry alone.
Site featurizers based on local chemical information, rather than geometry alone.
Site featurizers requiring external libraries for core functionality.
Site featurizers that fingerprint a site using local geometry.
Site featurizers that fingerprint a site using local geometry.
Site featurizers that fingerprint a site using local geometry.
Site featurizers that fingerprint a site using local geometry.
Site featurizers that fingerprint a site using local geometry.
Miscellaneous site featurizers.
Miscellaneous site featurizers.
Site featurizers based on distribution functions.
Site featurizers based on distribution functions.
Site featurizers based on distribution functions.
Structure featurizers generating a matrix for each structure.Most matrix structure featurizers contain the ability to flatten matrices to be dataframe-friendly.
Structure featurizers generating a matrix for each structure.Most matrix structure featurizers contain the ability to flatten matrices to be dataframe-friendly.
Structure featurizers generating a matrix for each structure.Most matrix structure featurizers contain the ability to flatten matrices to be dataframe-friendly.
Structure featurizers based on bonding.
Structure featurizers based on bonding.
Structure featurizers based on bonding.
Structure featurizers based on bonding.
Structure featurizers based on bonding.
Structure featurizers based on packing or ordering.
Structure featurizers based on packing or ordering.
Structure featurizers based on packing or ordering.
Structure featurizers based on packing or ordering.
Structure featurizers producing more than one kind of structue feature data.
Miscellaneous structure featurizers.
Miscellaneous structure featurizers.
Miscellaneous structure featurizers.
Structure featurizers implementing radial distribution functions.
Structure featurizers implementing radial distribution functions.
Structure featurizers implementing radial distribution functions.
Structure featurizers based on aggregating site features.
Structure featurizers based on symmetry.
Structure featurizers based on symmetry.
-------------
bandstructure
-------------
Features derived from a material's electronic bandstructure.
------------------------------------------------------------


.. list-table::
   :align: left
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - :code:`BranchPointEnergy`
     - Branch point energy and absolute band edge position.     
   * - :code:`BandFeaturizer`
     - Featurizes a pymatgen band structure object.     



----
base
----
Parent classes and meta-featurizers.
------------------------------------


.. list-table::
   :align: left
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - :code:`MultipleFeaturizer`
     - Class to run multiple featurizers on the same input data.     
   * - :code:`StackedFeaturizer`
     - Use the output of a machine learning model as features     
   * - :code:`BaseFeaturizer`
     - Abstract class to calculate features from raw materials input data     



-----------
composition
-----------
Features based on a material's composition.
-------------------------------------------

alloy - :code:`matminer.featurizers.composition.alloy`
______________________________________________________
Composition featurizers specialized for use with alloys.

.. list-table::
   :align: left
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - :code:`Miedema`
     - Formation enthalpies of intermetallic compounds, from Miedema et al.     
   * - :code:`YangSolidSolution`
     - Mixing thermochemistry and size mismatch terms of Yang and Zhang (2012)     
   * - :code:`WenAlloys`
     - Calculate features for alloy properties.     



composite - :code:`matminer.featurizers.composition.composite`
______________________________________________________________
Composition featurizers for composite features containing more than 1 category of general-purpose data.

.. list-table::
   :align: left
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - :code:`ElementProperty`
     - Class to calculate elemental property attributes.     
   * - :code:`Meredig`
     - Class to calculate features as defined in Meredig et. al.     



element - :code:`matminer.featurizers.composition.element`
__________________________________________________________
Composition featurizers for elemental data and stoichiometry.

.. list-table::
   :align: left
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - :code:`ElementFraction`
     - Class to calculate the atomic fraction of each element in a composition.     
   * - :code:`TMetalFraction`
     - Class to calculate fraction of magnetic transition metals in a composition.     
   * - :code:`Stoichiometry`
     - Calculate norms of stoichiometric attributes.     
   * - :code:`BandCenter`
     - Estimation of absolute position of band center using electronegativity.     



ion - :code:`matminer.featurizers.composition.ion`
__________________________________________________
Composition featurizers for compositions with ionic data.

.. list-table::
   :align: left
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - :code:`OxidationStates`
     - Statistics about the oxidation states for each specie.     
   * - :code:`IonProperty`
     - Ionic property attributes. Similar to ElementProperty.     
   * - :code:`ElectronAffinity`
     - Calculate average electron affinity times formal charge of anion elements.     
   * - :code:`ElectronegativityDiff`
     - Features from electronegativity differences between anions and cations.     



orbital - :code:`matminer.featurizers.composition.orbital`
__________________________________________________________
Composition featurizers for orbital data.

.. list-table::
   :align: left
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - :code:`AtomicOrbitals`
     - Determine HOMO/LUMO features based on a composition.     
   * - :code:`ValenceOrbital`
     - Attributes of valence orbital shells     



packing - :code:`matminer.featurizers.composition.packing`
__________________________________________________________
Composition featurizers for determining packing characteristics.

.. list-table::
   :align: left
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - :code:`AtomicPackingEfficiency`
     - Packing efficiency based on a geometric theory of the amorphous packing     



thermo - :code:`matminer.featurizers.composition.thermo`
________________________________________________________
Composition featurizers for thermodynamic properties.

.. list-table::
   :align: left
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - :code:`CohesiveEnergy`
     - Cohesive energy per atom using elemental cohesive energies and     
   * - :code:`CohesiveEnergyMP`
     - Cohesive energy per atom lookup using Materials Project     



-----------
conversions
-----------
Conversion utilities.
---------------------


.. list-table::
   :align: left
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - :code:`ConversionFeaturizer`
     - Abstract class to perform data conversions.     
   * - :code:`StrToComposition`
     - Utility featurizer to convert a string to a Composition     
   * - :code:`StructureToComposition`
     - Utility featurizer to convert a Structure to a Composition.     
   * - :code:`StructureToIStructure`
     - Utility featurizer to convert a Structure to an immutable IStructure.     
   * - :code:`DictToObject`
     - Utility featurizer to decode a dict to Python object via MSON.     
   * - :code:`JsonToObject`
     - Utility featurizer to decode json data to a Python object via MSON.     
   * - :code:`StructureToOxidStructure`
     - Utility featurizer to add oxidation states to a pymatgen Structure.     
   * - :code:`CompositionToOxidComposition`
     - Utility featurizer to add oxidation states to a pymatgen Composition.     
   * - :code:`CompositionToStructureFromMP`
     - Featurizer to get a Structure object from Materials Project using the     
   * - :code:`PymatgenFunctionApplicator`
     - Featurizer to run any function using on/from pymatgen primitives.     
   * - :code:`ASEAtomstoStructure`
     - Convert dataframes of ase structures to pymatgen structures for further use with     



---
dos
---
Features based on a material's electronic density of states.
------------------------------------------------------------


.. list-table::
   :align: left
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - :code:`SiteDOS`
     - report the fractional s/p/d/f dos for a particular site. a CompleteDos     
   * - :code:`DOSFeaturizer`
     - Significant character and contribution of the density of state from a     
   * - :code:`DopingFermi`
     - The fermi level (w.r.t. selected reference energy) associated with a     
   * - :code:`Hybridization`
     - quantify s/p/d/f orbital character and their hybridizations at band edges     
   * - :code:`DosAsymmetry`
     - Quantifies the asymmetry of the DOS near the Fermi level.     



--------
function
--------
Classes for expanding sets of features calculated with other featurizers.
-------------------------------------------------------------------------


.. list-table::
   :align: left
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - :code:`FunctionFeaturizer`
     - Features from functions applied to existing features, e.g. "1/x"     



----
site
----
Features from individual sites in a material's crystal structure.
-----------------------------------------------------------------

bonding - :code:`matminer.featurizers.site.bonding`
___________________________________________________
Site featurizers based on bonding.

.. list-table::
   :align: left
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - :code:`BondOrientationalParameter`
     - Averages of spherical harmonics of local neighbors     
   * - :code:`AverageBondLength`
     - Determines the average bond length between one specific site     
   * - :code:`AverageBondAngle`
     - Determines the average bond angles of a specific site with     



chemical - :code:`matminer.featurizers.site.chemical`
_____________________________________________________
Site featurizers based on local chemical information, rather than geometry alone.

.. list-table::
   :align: left
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - :code:`ChemicalSRO`
     - Chemical short range ordering, deviation of local site and nominal structure compositions     
   * - :code:`EwaldSiteEnergy`
     - Compute site energy from Coulombic interactions     
   * - :code:`LocalPropertyDifference`
     - Differences in elemental properties between site and its neighboring sites.     
   * - :code:`SiteElementalProperty`
     - Elemental properties of atom on a certain site     



external - :code:`matminer.featurizers.site.external`
_____________________________________________________
Site featurizers requiring external libraries for core functionality.

.. list-table::
   :align: left
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - :code:`SOAP`
     - Smooth overlap of atomic positions (interface via DScribe).     



fingerprint - :code:`matminer.featurizers.site.fingerprint`
___________________________________________________________
Site featurizers that fingerprint a site using local geometry.

.. list-table::
   :align: left
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - :code:`AGNIFingerprints`
     - Product integral of RDF and Gaussian window function, from `Botu et al <http://pubs.acs.org/doi/abs/10.1021/acs.jpcc.6b10908>`_.     
   * - :code:`OPSiteFingerprint`
     - Local structure order parameters computed from a site's neighbor env.     
   * - :code:`CrystalNNFingerprint`
     - A local order parameter fingerprint for periodic crystals.     
   * - :code:`VoronoiFingerprint`
     - Voronoi tessellation-based features around target site.     
   * - :code:`ChemEnvSiteFingerprint`
     - Resemblance of given sites to ideal environments     



misc - :code:`matminer.featurizers.site.misc`
_____________________________________________
Miscellaneous site featurizers.

.. list-table::
   :align: left
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - :code:`IntersticeDistribution`
     - Interstice distribution in the neighboring cluster around an atom site.     
   * - :code:`CoordinationNumber`
     - Number of first nearest neighbors of a site.     



rdf - :code:`matminer.featurizers.site.rdf`
___________________________________________
Site featurizers based on distribution functions.

.. list-table::
   :align: left
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - :code:`GaussianSymmFunc`
     - Gaussian symmetry function features suggested by Behler et al.     
   * - :code:`GeneralizedRadialDistributionFunction`
     - Compute the general radial distribution function (GRDF) for a site.     
   * - :code:`AngularFourierSeries`
     - Compute the angular Fourier series (AFS), including both angular and radial info     



---------
structure
---------
Generating features based on a material's crystal structure.
------------------------------------------------------------

bonding - :code:`matminer.featurizers.structure.bonding`
________________________________________________________
Structure featurizers based on bonding.

.. list-table::
   :align: left
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - :code:`BondFractions`
     - Compute the fraction of each bond in a structure, based on NearestNeighbors.     
   * - :code:`BagofBonds`
     - Compute a Bag of Bonds vector, as first described by Hansen et al. (2015).     
   * - :code:`GlobalInstabilityIndex`
     - The global instability index of a structure.     
   * - :code:`StructuralHeterogeneity`
     - Variance in the bond lengths and atomic volumes in a structure     
   * - :code:`MinimumRelativeDistances`
     - Determines the relative distance of each site to its closest neighbor.     



composite - :code:`matminer.featurizers.structure.composite`
____________________________________________________________
Structure featurizers producing more than one kind of structue feature data.

.. list-table::
   :align: left
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - :code:`JarvisCFID`
     - Classical Force-Field Inspired Descriptors (CFID) from Jarvis-ML.     



matrix - :code:`matminer.featurizers.structure.matrix`
______________________________________________________
Structure featurizers generating a matrix for each structure.Most matrix structure featurizers contain the ability to flatten matrices to be dataframe-friendly.

.. list-table::
   :align: left
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - :code:`CoulombMatrix`
     - The Coulomb matrix, a representation of nuclear coulombic interaction.     
   * - :code:`SineCoulombMatrix`
     - A variant of the Coulomb matrix developed for periodic crystals.     
   * - :code:`OrbitalFieldMatrix`
     - Representation based on the valence shell electrons of neighboring atoms.     



misc - :code:`matminer.featurizers.structure.misc`
__________________________________________________
Miscellaneous structure featurizers.

.. list-table::
   :align: left
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - :code:`EwaldEnergy`
     - Compute the energy from Coulombic interactions.     
   * - :code:`StructureComposition`
     - Features related to the composition of a structure     
   * - :code:`XRDPowderPattern`
     - 1D array representing powder diffraction of a structure as calculated by     



order - :code:`matminer.featurizers.structure.order`
____________________________________________________
Structure featurizers based on packing or ordering.

.. list-table::
   :align: left
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - :code:`DensityFeatures`
     - Calculates density and density-like features     
   * - :code:`ChemicalOrdering`
     - How much the ordering of species in the structure differs from random     
   * - :code:`MaximumPackingEfficiency`
     - Maximum possible packing efficiency of this structure     
   * - :code:`StructuralComplexity`
     - Shannon information entropy of a structure.     



rdf - :code:`matminer.featurizers.structure.rdf`
________________________________________________
Structure featurizers implementing radial distribution functions.

.. list-table::
   :align: left
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - :code:`RadialDistributionFunction`
     - Calculate the radial distribution function (RDF) of a crystal structure.     
   * - :code:`PartialRadialDistributionFunction`
     - Compute the partial radial distribution function (PRDF) of an xtal structure     
   * - :code:`ElectronicRadialDistributionFunction`
     - Calculate the inherent electronic radial distribution function (ReDF)     



sites - :code:`matminer.featurizers.structure.sites`
____________________________________________________
Structure featurizers based on aggregating site features.

.. list-table::
   :align: left
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - :code:`SiteStatsFingerprint`
     - Computes statistics of properties across all sites in a structure.     



symmetry - :code:`matminer.featurizers.structure.symmetry`
__________________________________________________________
Structure featurizers based on symmetry.

.. list-table::
   :align: left
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - :code:`GlobalSymmetryFeatures`
     - Determines symmetry features, e.g. spacegroup number and  crystal system     
   * - :code:`Dimensionality`
     - Returns dimensionality of structure: 1 means linear chains of atoms OR     



