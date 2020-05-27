====================
Table of Featurizers
====================

Below, you will find a description of each featurizer, listed in tables grouped by module.

-------------
bandstructure
-------------
Features derived from a material's electronic bandstructure.
------------------------------------------------------------

(matminer.featurizers.bandstructure)

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

(matminer.featurizers.base)

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

(matminer.featurizers.composition)

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
   * - :code:`OxidationStates`
     - Statistics about the oxidation states for each specie.     
   * - :code:`AtomicOrbitals`
     - Determine HOMO/LUMO features based on a composition.     
   * - :code:`BandCenter`
     - Estimation of absolute position of band center using electronegativity.     
   * - :code:`ElectronegativityDiff`
     - Features from electronegativity differences between anions and cations.     
   * - :code:`ElectronAffinity`
     - Calculate average electron affinity times formal charge of anion elements.     
   * - :code:`Stoichiometry`
     - Calculate norms of stoichiometric attributes.     
   * - :code:`ValenceOrbital`
     - Attributes of valence orbital shells     
   * - :code:`IonProperty`
     - Ionic property attributes. Similar to ElementProperty.     
   * - :code:`ElementFraction`
     - Class to calculate the atomic fraction of each element in a composition.     
   * - :code:`TMetalFraction`
     - Class to calculate fraction of magnetic transition metals in a composition.     
   * - :code:`CohesiveEnergy`
     - Cohesive energy per atom using elemental cohesive energies and     
   * - :code:`CohesiveEnergyMP`
     - Cohesive energy per atom lookup using Materials Project     
   * - :code:`Miedema`
     - Formation enthalpies of intermetallic compounds, from Miedema et al.     
   * - :code:`YangSolidSolution`
     - Mixing thermochemistry and size mismatch terms of Yang and Zhang (2012)     
   * - :code:`AtomicPackingEfficiency`
     - Packing efficiency based on a geometric theory of the amorphous packing     



-----------
conversions
-----------
Conversion utilities.
---------------------

(matminer.featurizers.conversions)

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



---
dos
---
Features based on a material's electronic density of states.
------------------------------------------------------------

(matminer.featurizers.dos)

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

(matminer.featurizers.function)

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

(matminer.featurizers.site)

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
   * - :code:`IntersticeDistribution`
     - Interstice distribution in the neighboring cluster around an atom site.     
   * - :code:`ChemicalSRO`
     - Chemical short range ordering, deviation of local site and nominal structure compositions     
   * - :code:`GaussianSymmFunc`
     - Gaussian symmetry function features suggested by Behler et al.     
   * - :code:`EwaldSiteEnergy`
     - Compute site energy from Coulombic interactions     
   * - :code:`ChemEnvSiteFingerprint`
     - Resemblance of given sites to ideal environments     
   * - :code:`CoordinationNumber`
     - Number of first nearest neighbors of a site.     
   * - :code:`GeneralizedRadialDistributionFunction`
     - Compute the general radial distribution function (GRDF) for a site.     
   * - :code:`AngularFourierSeries`
     - Compute the angular Fourier series (AFS), including both angular and radial info     
   * - :code:`LocalPropertyDifference`
     - Differences in elemental properties between site and its neighboring sites.     
   * - :code:`BondOrientationalParameter`
     - Averages of spherical harmonics of local neighbors     
   * - :code:`SiteElementalProperty`
     - Elemental properties of atom on a certain site     
   * - :code:`AverageBondLength`
     - Determines the average bond length between one specific site     
   * - :code:`AverageBondAngle`
     - Determines the average bond angles of a specific site with     



---------
structure
---------
Generating features based on a material's crystal structure.
------------------------------------------------------------

(matminer.featurizers.structure)

.. list-table::
   :align: left
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Description
   * - :code:`DensityFeatures`
     - Calculates density and density-like features     
   * - :code:`GlobalSymmetryFeatures`
     - Determines symmetry features, e.g. spacegroup number and  crystal system     
   * - :code:`Dimensionality`
     - Returns dimensionality of structure: 1 means linear chains of atoms OR     
   * - :code:`RadialDistributionFunction`
     - Calculate the radial distribution function (RDF) of a crystal structure.     
   * - :code:`PartialRadialDistributionFunction`
     - Compute the partial radial distribution function (PRDF) of an xtal structure     
   * - :code:`ElectronicRadialDistributionFunction`
     - Calculate the inherent electronic radial distribution function (ReDF)     
   * - :code:`CoulombMatrix`
     - The Coulomb matrix, a representation of nuclear coulombic interaction.     
   * - :code:`SineCoulombMatrix`
     - A variant of the Coulomb matrix developed for periodic crystals.     
   * - :code:`OrbitalFieldMatrix`
     - Representation based on the valence shell electrons of neighboring atoms.     
   * - :code:`MinimumRelativeDistances`
     - Determines the relative distance of each site to its closest neighbor.     
   * - :code:`SiteStatsFingerprint`
     - Computes statistics of properties across all sites in a structure.     
   * - :code:`EwaldEnergy`
     - Compute the energy from Coulombic interactions.     
   * - :code:`BondFractions`
     - Compute the fraction of each bond in a structure, based on NearestNeighbors.     
   * - :code:`BagofBonds`
     - Compute a Bag of Bonds vector, as first described by Hansen et al. (2015).     
   * - :code:`StructuralHeterogeneity`
     - Variance in the bond lengths and atomic volumes in a structure     
   * - :code:`MaximumPackingEfficiency`
     - Maximum possible packing efficiency of this structure     
   * - :code:`ChemicalOrdering`
     - How much the ordering of species in the structure differs from random     
   * - :code:`StructureComposition`
     - Features related to the composition of a structure     
   * - :code:`XRDPowderPattern`
     - 1D array representing powder diffraction of a structure as calculated by     
   * - :code:`CGCNNFeaturizer`
     - Features generated by training a Crystal Graph Convolutional Neural Network     
   * - :code:`JarvisCFID`
     - Classical Force-Field Inspired Descriptors (CFID) from Jarvis-ML.     
   * - :code:`SOAP`
     - Smooth overlap of atomic positions (interface via dscribe).     
   * - :code:`GlobalInstabilityIndex`
     - The global instability index of a structure.     
   * - :code:`StructuralComplexity`
     - Shannon information entropy of a structure.     



