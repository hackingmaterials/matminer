==============================
Datasets Available in Matminer
==============================

Below you will find descriptions and reference data on each available dataset, ordered by load_dataset() keyword argument


boltztrap_mp
brgoch_superhard_training
castelli_perovskites
citrine_thermal_conductivity
dielectric_constant
double_perovskites_gap
double_perovskites_gap_lumo
elastic_tensor_2015
expt_formation_enthalpy
expt_gap
flla
glass_binary
glass_binary_v2
glass_ternary_hipt
glass_ternary_landolt
heusler_magnetic
jarvis_dft_2d
jarvis_dft_3d
jarvis_ml_dft_training
m2ax
matbench_dielectric
matbench_expt_gap
matbench_expt_is_metal
matbench_glass
matbench_jdft2d
matbench_log_gvrh
matbench_log_kvrh
matbench_mp_e_form
matbench_mp_gap
matbench_mp_is_metal
matbench_perovskites
matbench_phonons
matbench_steels
mp_all
mp_nostruct
phonon_dielectric_mp
piezoelectric_tensor
steel_strength
wolverton_oxides

------------
boltztrap_mp
------------
Effective mass and thermoelectric properties of 8924 compounds in The  Materials Project database that are calculated by the BoltzTraP software package run on the GGA-PBE or GGA+U density functional theory calculation results. The properties are reported at the temperature of 300 Kelvin and the carrier concentration of 1e18 1/cm3.

**Number of entries:** 8924

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`formula`
     - Chemical formula of the entry
   * - :code:`m_n`
     - n-type/conduction band effective mass. Units: m_e where m_e is the mass of an electron; i.e. m_n is a unitless ratio
   * - :code:`m_p`
     - p-type/valence band effective mass.
   * - :code:`mpid`
     - Materials Project identifier
   * - :code:`pf_n`
     - n-type thermoelectric power factor in uW/cm2.K where uW is microwatts and a constant relaxation time of 1e-14 assumed.
   * - :code:`pf_p`
     - p-type power factor in uW/cm2.K
   * - :code:`s_n`
     - n-type Seebeck coefficient in micro Volts per Kelvin
   * - :code:`s_p`
     - p-type Seebeck coefficient in micro Volts per Kelvin
   * - :code:`structure`
     - pymatgen Structure object describing the crystal structure of the material



**Reference**

Ricci, F. et al. An ab initio electronic transport database for inorganic materials. Sci. Data 4:170085 doi: 10.1038/sdata.2017.85 (2017).
Ricci F, Chen W, Aydemir U, Snyder J, Rignanese G, Jain A, Hautier G (2017) Data from: An ab initio electronic transport database for inorganic materials. Dryad Digital Repository. https://doi.org/10.5061/dryad.gn001



**Bibtex Formatted Citations**

@Article{Ricci2017,
author={Ricci, Francesco
and Chen, Wei
and Aydemir, Umut
and Snyder, G. Jeffrey
and Rignanese, Gian-Marco
and Jain, Anubhav
and Hautier, Geoffroy},
title={An ab initio electronic transport database for inorganic materials},
journal={Scientific Data},
year={2017},
month={Jul},
day={04},
publisher={The Author(s)},
volume={4},
pages={170085},
note={Data Descriptor},
url={http://dx.doi.org/10.1038/sdata.2017.85}
}

@misc{dryad_gn001,
title = {Data from: An ab initio electronic transport database for inorganic materials},
author = {Ricci, F and Chen, W and Aydemir, U and Snyder, J and Rignanese, G and Jain, A and Hautier, G},
year = {2017},
journal = {Scientific Data},
URL = {https://doi.org/10.5061/dryad.gn001},
doi = {doi:10.5061/dryad.gn001},
publisher = {Dryad Digital Repository}
}




-------------------------
brgoch_superhard_training
-------------------------
2574 materials used for training regressors that predict shear and bulk modulus.

**Number of entries:** 2574

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`brgoch_feats`
     - features used in brgoch study compressed to a dictionary
   * - :code:`bulk_modulus`
     - VRH bulk modulus
   * - :code:`composition`
     - pymatgen composition object
   * - :code:`formula`
     - Chemical formula as a string
   * - :code:`material_id`
     - materials project id
   * - :code:`structure`
     - pymatgen structure object
   * - :code:`shear_modulus`
     - VRH shear modulus
   * - :code:`suspect_value`
     - True if bulk or shear value did not closely match (within 5%/1GPa of MP) materials project value at time of cross reference or if no material could be found



**Reference**

Machine Learning Directed Search for Ultraincompressible, Superhard Materials
Aria Mansouri Tehrani, Anton O. Oliynyk, Marcus Parry, Zeshan Rizvi, Samantha Couper, Feng Lin, Lowell Miyagi, Taylor D. Sparks, and Jakoah Brgoch
Journal of the American Chemical Society 2018 140 (31), 9844-9853
DOI: 10.1021/jacs.8b02717



**Bibtex Formatted Citations**

@article{doi:10.1021/jacs.8b02717,
author = {Mansouri Tehrani, Aria and Oliynyk, Anton O. and Parry, Marcus and Rizvi, Zeshan and Couper, Samantha and Lin, Feng and Miyagi, Lowell and Sparks, Taylor D. and Brgoch, Jakoah},
title = {Machine Learning Directed Search for Ultraincompressible, Superhard Materials},
journal = {Journal of the American Chemical Society},
volume = {140},
number = {31},
pages = {9844-9853},
year = {2018},
doi = {10.1021/jacs.8b02717},
note ={PMID: 30010335},
URL = {
https://doi.org/10.1021/jacs.8b02717
},
eprint = {
https://doi.org/10.1021/jacs.8b02717
}
}




--------------------
castelli_perovskites
--------------------
18,928 perovskites generated with ABX combinatorics, calculating gllbsc band gap and pbe structure, and also reporting absolute band edge positions and heat of formation.

**Number of entries:** 18928

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`cbm`
     - similar to vbm but for conduction band
   * - :code:`e_form`
     - heat of formation in eV
   * - :code:`fermi level`
     - the thermodynamic work required to add one electron to the body in eV
   * - :code:`fermi width`
     - fermi bandwidth
   * - :code:`formula`
     - Chemical formula of the material
   * - :code:`gap gllbsc`
     - electronic band gap in eV calculated via gllbsc functional
   * - :code:`gap is direct`
     - boolean indicator for direct gap
   * - :code:`mu_b`
     - magnetic moment in terms of Bohr magneton
   * - :code:`structure`
     - crystal structure represented by pymatgen Structure object
   * - :code:`vbm`
     - absolute value of valence band edge calculated via gllbsc



**Reference**

Ivano E. Castelli, David D. Landis, Kristian S. Thygesen, Søren Dahl, Ib Chorkendorff, Thomas F. Jaramillo and Karsten W. Jacobsen (2012) New cubic perovskites for one- and two-photon water splitting using the computational materials repository. Energy Environ. Sci., 2012,5, 9034-9043 https://doi.org/10.1039/C2EE22341D



**Bibtex Formatted Citations**

@Article{C2EE22341D,
author ="Castelli, Ivano E. and Landis, David D. and Thygesen, Kristian S. and Dahl, Søren and Chorkendorff, Ib and Jaramillo, Thomas F. and Jacobsen, Karsten W.",
title  ="New cubic perovskites for one- and two-photon water splitting using the computational materials repository",
journal  ="Energy Environ. Sci.",
year  ="2012",
volume  ="5",
issue  ="10",
pages  ="9034-9043",
publisher  ="The Royal Society of Chemistry",
doi  ="10.1039/C2EE22341D",
url  ="http://dx.doi.org/10.1039/C2EE22341D",
abstract  ="A new efficient photoelectrochemical cell (PEC) is one of the possible solutions to the energy and climate problems of our time. Such a device requires development of new semiconducting materials with tailored properties with respect to stability and light absorption. Here we perform computational screening of around 19 000 oxides{,} oxynitrides{,} oxysulfides{,} oxyfluorides{,} and oxyfluoronitrides in the cubic perovskite structure with PEC applications in mind. We address three main applications: light absorbers for one- and two-photon water splitting and high-stability transparent shields to protect against corrosion. We end up with 20{,} 12{,} and 15 different combinations of oxides{,} oxynitrides and oxyfluorides{,} respectively{,} inviting further experimental investigation."}




----------------------------
citrine_thermal_conductivity
----------------------------
Thermal conductivity of 872 compounds measured experimentally and retrieved from Citrine database from various references. The reported values are measured at various temperatures of which 295 are at room temperature.

**Number of entries:** 872

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`formula`
     - Chemical formula of the dataset entry
   * - :code:`k-units`
     - units of thermal conductivity
   * - :code:`k_condition`
     - Temperature description of testing conditions
   * - :code:`k_condition_units`
     - units of testing condition temperature representation
   * - :code:`k_expt`
     - the experimentally measured thermal conductivity in SI units of W/m.K



**Reference**

https://www.citrination.com



**Bibtex Formatted Citations**

@misc{Citrine Informatics,
title = {Citrination},
howpublished = {\url{https://www.citrination.com/}},
}




-------------------
dielectric_constant
-------------------
1,056 structures with dielectric properties, calculated with DFPT-PBE.

**Number of entries:** 1056

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`band_gap`
     - Measure of the conductivity of a material
   * - :code:`cif`
     - optional: Description string for structure
   * - :code:`e_electronic`
     - electronic contribution to dielectric tensor
   * - :code:`e_total`
     - Total dielectric tensor incorporating both electronic and ionic contributions
   * - :code:`formula`
     - Chemical formula of the material
   * - :code:`material_id`
     - Materials Project ID of the material
   * - :code:`meta`
     - optional, metadata descriptor of the datapoint
   * - :code:`n`
     - Refractive Index
   * - :code:`nsites`
     - The \# of atoms in the unit cell of the calculation.
   * - :code:`poly_electronic`
     - the average of the eigenvalues of the electronic contribution to the dielectric tensor
   * - :code:`poly_total`
     - the average of the eigenvalues of the total (electronic and ionic) contributions to the dielectric tensor
   * - :code:`poscar`
     - optional: Poscar metadata
   * - :code:`pot_ferroelectric`
     - Whether the material is potentially ferroelectric
   * - :code:`space_group`
     - Integer specifying the crystallographic structure of the material
   * - :code:`structure`
     - pandas Series defining the structure of the material
   * - :code:`volume`
     - Volume of the unit cell in cubic angstroms, For supercell calculations, this quantity refers to the volume of the full supercell. 



**Reference**

Petousis, I., Mrdjenovich, D., Ballouz, E., Liu, M., Winston, D.,
Chen, W., Graf, T., Schladt, T. D., Persson, K. A. & Prinz, F. B.
High-throughput screening of inorganic compounds for the discovery
of novel dielectric and optical materials. Sci. Data 4, 160134 (2017).



**Bibtex Formatted Citations**

@Article{Petousis2017,
author={Petousis, Ioannis and Mrdjenovich, David and Ballouz, Eric
and Liu, Miao and Winston, Donald and Chen, Wei and Graf, Tanja
and Schladt, Thomas D. and Persson, Kristin A. and Prinz, Fritz B.},
title={High-throughput screening of inorganic compounds for the
discovery of novel dielectric and optical materials},
journal={Scientific Data},
year={2017},
month={Jan},
day={31},
publisher={The Author(s)},
volume={4},
pages={160134},
note={Data Descriptor},
url={http://dx.doi.org/10.1038/sdata.2016.134}
}




----------------------
double_perovskites_gap
----------------------
Band gap of 1306 double perovskites (a_1-b_1-a_2-b_2-O6) calculated using ﻿Gritsenko, van Leeuwen, van Lenthe and Baerends potential (gllbsc) in GPAW.

**Number of entries:** 1306

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`a_1`
     - Species occupying the a1 perovskite site
   * - :code:`a_2`
     - Species occupying the a2 site
   * - :code:`b_1`
     - Species occupying the b1 site
   * - :code:`b_2`
     - Species occupying the b2 site
   * - :code:`formula`
     - Chemical formula of the entry
   * - :code:`gap gllbsc`
     - electronic band gap (in eV) calculated via gllbsc



**Reference**

Dataset discussed in:
Pilania, G. et al. Machine learning bandgaps of double perovskites. Sci. Rep. 6, 19375; doi: 10.1038/srep19375 (2016).
Dataset sourced from:
https://cmr.fysik.dtu.dk/



**Bibtex Formatted Citations**

@Article{Pilania2016,
author={Pilania, G.
and Mannodi-Kanakkithodi, A.
and Uberuaga, B. P.
and Ramprasad, R.
and Gubernatis, J. E.
and Lookman, T.},
title={Machine learning bandgaps of double perovskites},
journal={Scientific Reports},
year={2016},
month={Jan},
day={19},
publisher={The Author(s)},
volume={6},
pages={19375},
note={Article},
url={http://dx.doi.org/10.1038/srep19375}
}

@misc{Computational Materials Repository,
title = {Computational Materials Repository},
howpublished = {\url{https://cmr.fysik.dtu.dk/}},
}




---------------------------
double_perovskites_gap_lumo
---------------------------
Supplementary lumo data of 55 atoms for the double_perovskites_gap dataset.

**Number of entries:** 55

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`atom`
     - Name of the atom whos lumo is listed
   * - :code:`lumo`
     - Lowest unoccupied molecular obital energy level (in eV)



**Reference**

Dataset discussed in:
Pilania, G. et al. Machine learning bandgaps of double perovskites. Sci. Rep. 6, 19375; doi: 10.1038/srep19375 (2016).
Dataset sourced from:
https://cmr.fysik.dtu.dk/



**Bibtex Formatted Citations**

@Article{Pilania2016,
author={Pilania, G.
and Mannodi-Kanakkithodi, A.
and Uberuaga, B. P.
and Ramprasad, R.
and Gubernatis, J. E.
and Lookman, T.},
title={Machine learning bandgaps of double perovskites},
journal={Scientific Reports},
year={2016},
month={Jan},
day={19},
publisher={The Author(s)},
volume={6},
pages={19375},
note={Article},
url={http://dx.doi.org/10.1038/srep19375}
}

@misc{Computational Materials Repository,
title = {Computational Materials Repository},
howpublished = {\url{https://cmr.fysik.dtu.dk/}},
}




-------------------
elastic_tensor_2015
-------------------
1,181 structures with elastic properties calculated with DFT-PBE.

**Number of entries:** 1181

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`G_Reuss`
     - Lower bound on shear modulus for polycrystalline material
   * - :code:`G_VRH`
     - Average of G_Reuss and G_Voigt
   * - :code:`G_Voigt`
     - Upper bound on shear modulus for polycrystalline material
   * - :code:`K_Reuss`
     - Lower bound on bulk modulus for polycrystalline material
   * - :code:`K_VRH`
     - Average of K_Reuss and K_Voigt
   * - :code:`K_Voigt`
     - Upper bound on bulk modulus for polycrystalline material
   * - :code:`cif`
     - optional: Description string for structure
   * - :code:`compliance_tensor`
     - Tensor describing elastic behavior
   * - :code:`elastic_anisotropy`
     - measure of directional dependence of the materials elasticity, metric is always >= 0
   * - :code:`elastic_tensor`
     - Tensor describing elastic behavior corresponding to IEEE orientation, symmetrized to crystal structure 
   * - :code:`elastic_tensor_original`
     - Tensor describing elastic behavior, unsymmetrized, corresponding to POSCAR conventional standard cell orientation
   * - :code:`formula`
     - Chemical formula of the material
   * - :code:`kpoint_density`
     - optional: Sampling parameter from calculation
   * - :code:`material_id`
     - Materials Project ID of the material
   * - :code:`nsites`
     - The \# of atoms in the unit cell of the calculation.
   * - :code:`poisson_ratio`
     - Describes lateral response to loading
   * - :code:`poscar`
     - optional: Poscar metadata
   * - :code:`space_group`
     - Integer specifying the crystallographic structure of the material
   * - :code:`structure`
     - pandas Series defining the structure of the material
   * - :code:`volume`
     - Volume of the unit cell in cubic angstroms, For supercell calculations, this quantity refers to the volume of the full supercell. 



**Reference**

Jong, M. De, Chen, W., Angsten, T., Jain, A., Notestine, R., Gamst,
A., Sluiter, M., Ande, C. K., Zwaag, S. Van Der, Plata, J. J., Toher,
C., Curtarolo, S., Ceder, G., Persson, K. and Asta, M., "Charting
the complete elastic properties of inorganic crystalline compounds",
Scientific Data volume 2, Article number: 150009 (2015)



**Bibtex Formatted Citations**

@Article{deJong2015,
author={de Jong, Maarten and Chen, Wei and Angsten, Thomas
and Jain, Anubhav and Notestine, Randy and Gamst, Anthony
and Sluiter, Marcel and Krishna Ande, Chaitanya
and van der Zwaag, Sybrand and Plata, Jose J. and Toher, Cormac
and Curtarolo, Stefano and Ceder, Gerbrand and Persson, Kristin A.
and Asta, Mark},
title={Charting the complete elastic properties
of inorganic crystalline compounds},
journal={Scientific Data},
year={2015},
month={Mar},
day={17},
publisher={The Author(s)},
volume={2},
pages={150009},
note={Data Descriptor},
url={http://dx.doi.org/10.1038/sdata.2015.9}
}




-----------------------
expt_formation_enthalpy
-----------------------
Experimental formation enthalpies for inorganic compounds, collected from years of calorimetric experiments. There are 1,276 entries in this dataset, mostly binary compounds. Matching mpids or oqmdids as well as the DFT-computed formation energies are also added (if any).

**Number of entries:** 1276

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`e_form expt`
     - experimental formation enthalpy (in eV/atom)
   * - :code:`e_form mp`
     - formation enthalpy from Materials Project (in eV/atom)
   * - :code:`e_form oqmd`
     - formation enthalpy from OQMD (in eV/atom)
   * - :code:`formula`
     - chemical formula
   * - :code:`mpid`
     - materials project id
   * - :code:`oqmdid`
     - OQMD id
   * - :code:`pearson symbol`
     - pearson symbol of the structure
   * - :code:`space group`
     - space group of the structure



**Reference**

https://www.nature.com/articles/sdata2017162



**Bibtex Formatted Citations**

@Article{Kim2017,
author={Kim, George
and Meschel, S. V.
and Nash, Philip
and Chen, Wei},
title={Experimental formation enthalpies for intermetallic phases and other inorganic compounds},
journal={Scientific Data},
year={2017},
month={Oct},
day={24},
publisher={The Author(s)},
volume={4},
pages={170162},
note={Data Descriptor},
url={https://doi.org/10.1038/sdata.2017.162}}

 @misc{kim_meschel_nash_chen_2017, title={Experimental formation enthalpies for intermetallic phases and other inorganic compounds}, url={https://figshare.com/collections/Experimental_formation_enthalpies_for_intermetallic_phases_and_other_inorganic_compounds/3822835/1}, DOI={10.6084/m9.figshare.c.3822835.v1}, abstractNote={The standard enthalpy of formation of a compound is the energy associated with the reaction to form the compound from its component elements. The standard enthalpy of formation is a fundamental thermodynamic property that determines its phase stability, which can be coupled with other thermodynamic data to calculate phase diagrams. Calorimetry provides the only direct method by which the standard enthalpy of formation is experimentally measured. However, the measurement is often a time and energy intensive process. We present a dataset of enthalpies of formation measured by high-temperature calorimetry. The phases measured in this dataset include intermetallic compounds with transition metal and rare-earth elements, metal borides, metal carbides, and metallic silicides. These measurements were collected from over 50 years of calorimetric experiments. The dataset contains 1,276 entries on experimental enthalpy of formation values and structural information. Most of the entries are for binary compounds but ternary and quaternary compounds are being added as they become available. The dataset also contains predictions of enthalpy of formation from first-principles calculations for comparison.}, publisher={figshare}, author={Kim, George and Meschel, Susan and Nash, Philip and Chen, Wei}, year={2017}, month={Oct}}




--------
expt_gap
--------
Experimental band gap of 6354 inorganic semiconductors.

**Number of entries:** 6354

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`formula`
     - chemical formula
   * - :code:`gap expt`
     - band gap (in eV) measured experimentally



**Reference**

https://pubs.acs.org/doi/suppl/10.1021/acs.jpclett.8b00124



**Bibtex Formatted Citations**

@article{doi:10.1021/acs.jpclett.8b00124,
author = {Zhuo, Ya and Mansouri Tehrani, Aria and Brgoch, Jakoah},
title = {Predicting the Band Gaps of Inorganic Solids by Machine Learning},
journal = {The Journal of Physical Chemistry Letters},
volume = {9},
number = {7},
pages = {1668-1673},
year = {2018},
doi = {10.1021/acs.jpclett.8b00124},
note ={PMID: 29532658},
eprint = {
https://doi.org/10.1021/acs.jpclett.8b00124

}}




----
flla
----
3938 structures and computed formation energies from "Crystal Structure Representations for Machine Learning Models of Formation Energies."

**Number of entries:** 3938

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`e_above_hull`
     - The energy of decomposition of this material into the set of most stable materials at this chemical composition, in eV/atom.
   * - :code:`formation_energy`
     - Computed formation energy at 0K, 0atm using a reference state of zero for the pure elements.
   * - :code:`formation_energy_per_atom`
     - See formation_energy
   * - :code:`formula`
     - Chemical formula of the material
   * - :code:`material_id`
     - Materials Project ID of the material
   * - :code:`nsites`
     - The \# of atoms in the unit cell of the calculation.
   * - :code:`structure`
     - pandas Series defining the structure of the material



**Reference**

1) F. Faber, A. Lindmaa, O.A. von Lilienfeld, R. Armiento,
"Crystal structure representations for machine learning models of
formation energies", Int. J. Quantum Chem. 115 (2015) 1094–1101.
doi:10.1002/qua.24917.

(raw data)
2) Jain, A., Ong, S. P., Hautier, G., Chen, W., Richards, W. D.,
Dacek, S., Cholia, S., Gunter, D., Skinner, D., Ceder, G. & Persson,
K. A. Commentary: The Materials Project: A materials genome approach
to accelerating materials innovation. APL Mater. 1, 11002 (2013).



**Bibtex Formatted Citations**

@article{doi:10.1002/qua.24917,
author = {Faber, Felix and Lindmaa, Alexander and von Lilienfeld, O. Anatole and Armiento, Rickard},
title = {Crystal structure representations for machine learning models of formation energies},
journal = {International Journal of Quantum Chemistry},
volume = {115},
number = {16},
pages = {1094-1101},
keywords = {machine learning, formation energies, representations,
crystal structure, periodic systems},
doi = {10.1002/qua.24917},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/qua.24917},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/qua.24917},
abstract = {We introduce and evaluate a set of feature vector
representations of crystal structures for machine learning (ML)
models of formation energies of solids. ML models of atomization
energies of organic molecules have been successful using a Coulomb
matrix representation of the molecule. We consider three ways to
generalize such representations to periodic systems: (i) a matrix
where each element is related to the Ewald sum of the electrostatic
interaction between two different atoms in the unit cell repeated
over the lattice; (ii) an extended Coulomb-like matrix that takes
into account a number of neighboring unit cells; and (iii) an
ansatz that mimics the periodicity and the basic features of the
elements in the Ewald sum matrix using a sine function of the
crystal coordinates of the atoms. The representations are compared
for a Laplacian kernel with Manhattan norm, trained to reproduce
formation energies using a dataset of 3938 crystal structures
obtained from the Materials Project. For training sets consisting
of 3000 crystals, the generalization error in predicting formation
energies of new structures corresponds to (i) 0.49, (ii) 0.64, and
(iii) for the respective representations. © 2015 Wiley Periodicals,
Inc.}
}

@article{doi:10.1063/1.4812323,
author = {Jain,Anubhav  and Ong,Shyue Ping  and Hautier,Geoffroy
and Chen,Wei  and Richards,William Davidson  and Dacek,Stephen
and Cholia,Shreyas  and Gunter,Dan  and Skinner,David
and Ceder,Gerbrand  and Persson,Kristin A. },
title = {Commentary: The Materials Project: A materials genome
approach to accelerating materials innovation},
journal = {APL Materials},
volume = {1},
number = {1},
pages = {011002},
year = {2013},
doi = {10.1063/1.4812323},
URL = {https://doi.org/10.1063/1.4812323},
eprint = {https://doi.org/10.1063/1.4812323}
}




------------
glass_binary
------------
Metallic glass formation data for binary alloys, collected from various experimental techniques such as melt-spinning or mechanical alloying. This dataset covers all compositions with an interval of 5 at. % in 59 binary systems, containing a total of 5959 alloys in the dataset. The target property of this dataset is the glass forming ability (GFA), i.e. whether the composition can form monolithic glass or not, which is either 1 for glass forming or 0 for non-full glass forming.

**Number of entries:** 5959

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`formula`
     - chemical formula
   * - :code:`gfa`
     - glass forming ability, correlated with the phase column, designating whether the composition can form monolithic glass or not, 1: glass forming ("AM"), 0: non-full-forming("CR")



**Reference**

https://pubs.acs.org/doi/10.1021/acs.jpclett.7b01046



**Bibtex Formatted Citations**

@article{doi:10.1021/acs.jpclett.7b01046,
author = {Sun, Y. T. and Bai, H. Y. and Li, M. Z. and Wang, W. H.},
title = {Machine Learning Approach for Prediction and Understanding of Glass-Forming Ability},
journal = {The Journal of Physical Chemistry Letters},
volume = {8},
number = {14},
pages = {3434-3439},
year = {2017},
doi = {10.1021/acs.jpclett.7b01046},
note ={PMID: 28697303},
eprint = {
https://doi.org/10.1021/acs.jpclett.7b01046

}}




---------------
glass_binary_v2
---------------
Identical to glass_binary dataset, but with duplicate entries merged. If there was a disagreement in gfa when merging the class was defaulted to 1.

**Number of entries:** 5483

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`formula`
     - chemical formula
   * - :code:`gfa`
     - glass forming ability, correlated with the phase column, designating whether the composition can form monolithic glass or not, 1: glass forming ("AM"), 0: non-full-forming("CR")



**Reference**

https://pubs.acs.org/doi/10.1021/acs.jpclett.7b01046



**Bibtex Formatted Citations**

@article{doi:10.1021/acs.jpclett.7b01046,
author = {Sun, Y. T. and Bai, H. Y. and Li, M. Z. and Wang, W. H.},
title = {Machine Learning Approach for Prediction and Understanding of Glass-Forming Ability},
journal = {The Journal of Physical Chemistry Letters},
volume = {8},
number = {14},
pages = {3434-3439},
year = {2017},
doi = {10.1021/acs.jpclett.7b01046},
note ={PMID: 28697303},
eprint = {
https://doi.org/10.1021/acs.jpclett.7b01046

}}




------------------
glass_ternary_hipt
------------------
Metallic glass formation dataset for ternary alloys, collected from the high-throughput sputtering experiments measuring whether it is possible to form a glass using sputtering. The hipt experimental data are of the Co-Fe-Zr, Co-Ti-Zr, Co-V-Zr and Fe-Ti-Nb ternary systems.

**Number of entries:** 5170

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`formula`
     - Chemical formula of the entry
   * - :code:`gfa`
     - Glass forming ability: 1 means glass forming and coresponds to AM, 0 means non glass forming and corresponds to CR
   * - :code:`phase`
     - AM: amorphous phase or CR: crystalline phase
   * - :code:`processing`
     - How the point was processed, always sputtering for this dataset
   * - :code:`system`
     - System of dataset experiment, one of: CoFeZr, CoTiZr, CoVZr, or FeTiNb



**Reference**

Accelerated discovery of metallic glasses through iteration of machine learning and high-throughput experiments
By Fang Ren, Logan Ward, Travis Williams, Kevin J. Laws, Christopher Wolverton, Jason Hattrick-Simpers, Apurva Mehta
Science Advances 13 Apr 2018 : eaaq1566



**Bibtex Formatted Citations**

@article {Reneaaq1566,
author = {Ren, Fang and Ward, Logan and Williams, Travis and Laws, Kevin J. and Wolverton, Christopher and Hattrick-Simpers, Jason and Mehta, Apurva},
title = {Accelerated discovery of metallic glasses through iteration of machine learning and high-throughput experiments},
volume = {4},
number = {4},
year = {2018},
doi = {10.1126/sciadv.aaq1566},
publisher = {American Association for the Advancement of Science},
abstract = {With more than a hundred elements in the periodic table, a large number of potential new materials exist to address the technological and societal challenges we face today; however, without some guidance, searching through this vast combinatorial space is frustratingly slow and expensive, especially for materials strongly influenced by processing. We train a machine learning (ML) model on previously reported observations, parameters from physiochemical theories, and make it synthesis method{\textendash}dependent to guide high-throughput (HiTp) experiments to find a new system of metallic glasses in the Co-V-Zr ternary. Experimental observations are in good agreement with the predictions of the model, but there are quantitative discrepancies in the precise compositions predicted. We use these discrepancies to retrain the ML model. The refined model has significantly improved accuracy not only for the Co-V-Zr system but also across all other available validation data. We then use the refined model to guide the discovery of metallic glasses in two additional previously unreported ternaries. Although our approach of iterative use of ML and HiTp experiments has guided us to rapid discovery of three new glass-forming systems, it has also provided us with a quantitatively accurate, synthesis method{\textendash}sensitive predictor for metallic glasses that improves performance with use and thus promises to greatly accelerate discovery of many new metallic glasses. We believe that this discovery paradigm is applicable to a wider range of materials and should prove equally powerful for other materials and properties that are synthesis path{\textendash}dependent and that current physiochemical theories find challenging to predict.},
URL = {http://advances.sciencemag.org/content/4/4/eaaq1566},
eprint = {http://advances.sciencemag.org/content/4/4/eaaq1566.full.pdf},
journal = {Science Advances}
}




---------------------
glass_ternary_landolt
---------------------
Metallic glass formation dataset for ternary alloys, collected from the "Nonequilibrium Phase Diagrams of Ternary Amorphous Alloys,’ a volume of the Landolt– Börnstein collection. This dataset contains experimental measurements of whether it is possible to form a glass using a variety of processing techniques at thousands of compositions from hundreds of ternary systems. The processing techniques are designated in the "processing" column. There are originally 7191 experiments in this dataset, will be reduced to 6203 after deduplicated, and will be further reduced to 6118 if combining multiple data for one composition. There are originally 6780 melt-spinning experiments in this dataset, will be reduced to 5800 if deduplicated, and will be further reduced to 5736 if combining multiple experimental data for one composition.

**Number of entries:** 7191

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`formula`
     - Chemical formula of the entry
   * - :code:`gfa`
     - Glass forming ability: 1 means glass forming and corresponds to AM, 0 means non full glass forming and corresponds to CR AC or QC
   * - :code:`phase`
     - "AM": amorphous phase. "CR": crystalline phase. "AC": amorphous-crystalline composite phase. "QC": quasi-crystalline phase. Phases obtained from glass producing experiments
   * - :code:`processing`
     - processing method, meltspin or sputtering



**Reference**

Y. Kawazoe, T. Masumoto, A.-P. Tsai, J.-Z. Yu, T. Aihara Jr. (1997) Y. Kawazoe, J.-Z. Yu, A.-P. Tsai, T. Masumoto (ed.) SpringerMaterials
Nonequilibrium Phase Diagrams of Ternary Amorphous Alloys · 1 Introduction Landolt-Börnstein - Group III Condensed Matter 37A (Nonequilibrium Phase Diagrams of Ternary Amorphous Alloys) https://www.springer.com/gp/book/9783540605072 (Springer-Verlag Berlin Heidelberg © 1997) Accessed: 03-09-2019



**Bibtex Formatted Citations**

@Misc{LandoltBornstein1997:sm_lbs_978-3-540-47679-5_2,
author="Kawazoe, Y.
and Masumoto, T.
and Tsai, A.-P.
and Yu, J.-Z.
and Aihara Jr., T.",
editor="Kawazoe, Y.
and Yu, J.-Z.
and Tsai, A.-P.
and Masumoto, T.",
title="Nonequilibrium Phase Diagrams of Ternary Amorphous Alloys {\textperiodcentered} 1 Introduction: Datasheet from Landolt-B{\"o}rnstein - Group III Condensed Matter {\textperiodcentered} Volume 37A: ``Nonequilibrium Phase Diagrams of Ternary Amorphous Alloys'' in SpringerMaterials (https://dx.doi.org/10.1007/10510374{\_}2)",
publisher="Springer-Verlag Berlin Heidelberg",
note="Copyright 1997 Springer-Verlag Berlin Heidelberg",
note="Part of SpringerMaterials",
note="accessed 2018-10-23",
doi="10.1007/10510374_2",
url="https://materials.springer.com/lb/docs/sm_lbs_978-3-540-47679-5_2"
}

@Article{Ward2016,
author={Ward, Logan
and Agrawal, Ankit
and Choudhary, Alok
and Wolverton, Christopher},
title={A general-purpose machine learning framework for predicting properties of inorganic materials},
journal={Npj Computational Materials},
year={2016},
month={Aug},
day={26},
publisher={The Author(s)},
volume={2},
pages={16028},
note={Article},
url={http://dx.doi.org/10.1038/npjcompumats.2016.28}
}




----------------
heusler_magnetic
----------------
1153 Heusler alloys with DFT-calculated magnetic and electronic properties. The 1153 alloys include 576 full, 449 half and 128 inverse Heusler alloys. The data are extracted and cleaned (including de-duplicating) from Citrine.

**Number of entries:** 1153

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`e_form`
     - Formation energy in eV/atom
   * - :code:`formula`
     - Chemical formula of the entry
   * - :code:`heusler type`
     - Full, Half, or Inverse Heusler
   * - :code:`latt const`
     - Lattice constant
   * - :code:`mu_b`
     - Magnetic moment
   * - :code:`mu_b saturation`
     - Saturation magnetization in emu/cc
   * - :code:`num_electron`
     - Number of electrons per formula unit
   * - :code:`pol fermi`
     - Polarization at Fermi level in %
   * - :code:`struct type`
     - Structure type
   * - :code:`tetragonality`
     - Tetragonality, i.e. c/a



**Reference**

https://citrination.com/datasets/150561/



**Bibtex Formatted Citations**

@misc{Citrine Informatics,
title = {University of Alabama Heusler database},
howpublished = {\url{https://citrination.com/datasets/150561/}},
}




-------------
jarvis_dft_2d
-------------
Various properties of 636 2D materials computed with the OptB88vdW and TBmBJ functionals taken from the JARVIS DFT database.

**Number of entries:** 636

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`composition`
     - A Pymatgen Composition descriptor of the composition of the material
   * - :code:`e_form`
     - formation energy per atom, in eV/atom
   * - :code:`epsilon_x opt`
     - Static dielectric function in x direction calculated with OptB88vDW functional.
   * - :code:`epsilon_x tbmbj`
     - Static dielectric function in x direction calculuated with TBMBJ functional.
   * - :code:`epsilon_y opt`
     - Static dielectric function in y direction calculated with OptB88vDW functional.
   * - :code:`epsilon_y tbmbj`
     - Static dielectric function in y direction calculuated with TBMBJ functional.
   * - :code:`epsilon_z opt`
     - Static dielectric function in z direction calculated with OptB88vDW functional.
   * - :code:`epsilon_z tbmbj`
     - Static dielectric function in z direction calculuated with TBMBJ functional.
   * - :code:`exfoliation_en`
     - Exfoliation energy (monolayer formation E) in meV/atom
   * - :code:`gap opt`
     - Band gap calculated with OptB88vDW functional, in eV
   * - :code:`gap tbmbj`
     - Band gap calculated with TBMBJ functional, in eV
   * - :code:`jid`
     - JARVIS ID
   * - :code:`mpid`
     - Materials Project ID
   * - :code:`structure`
     - A description of the crystal structure of the material
   * - :code:`structure initial`
     - Initial structure description of the crystal structure of the material



**Reference**

2D Dataset discussed in:
High-throughput Identification and Characterization of Two dimensional Materials using Density functional theory Kamal Choudhary, Irina Kalish, Ryan Beams & Francesca Tavazza Scientific Reports volume 7, Article number: 5179 (2017)
Original 2D Data file sourced from:
choudhary, kamal; https://orcid.org/0000-0001-9737-8074 (2018): jdft_2d-7-7-2018.json. figshare. Dataset.



**Bibtex Formatted Citations**

@Article{Choudhary2017,
author={Choudhary, Kamal
and Kalish, Irina
and Beams, Ryan
and Tavazza, Francesca},
title={High-throughput Identification and Characterization of Two-dimensional Materials using Density functional theory},
journal={Scientific Reports},
year={2017},
volume={7},
number={1},
pages={5179},
abstract={We introduce a simple criterion to identify two-dimensional (2D) materials based on the comparison between experimental lattice constants and lattice constants mainly obtained from Materials-Project (MP) density functional theory (DFT) calculation repository. Specifically, if the relative difference between the two lattice constants for a specific material is greater than or equal to 5%, we predict them to be good candidates for 2D materials. We have predicted at least 1356 such 2D materials. For all the systems satisfying our criterion, we manually create single layer systems and calculate their energetics, structural, electronic, and elastic properties for both the bulk and the single layer cases. Currently the database consists of 1012 bulk and 430 single layer materials, of which 371 systems are common to bulk and single layer. The rest of calculations are underway. To validate our criterion, we calculated the exfoliation energy of the suggested layered materials, and we found that in 88.9% of the cases the currently accepted criterion for exfoliation was satisfied. Also, using molybdenum telluride as a test case, we performed X-ray diffraction and Raman scattering experiments to benchmark our calculations and understand their applicability and limitations. The data is publicly available at the website http://www.ctcms.nist.gov/{	extasciitilde}knc6/JVASP.html.},
issn={2045-2322},
doi={10.1038/s41598-017-05402-0},
url={https://doi.org/10.1038/s41598-017-05402-0}
}

@misc{choudhary__2018, title={jdft_2d-7-7-2018.json}, url={https://figshare.com/articles/jdft_2d-7-7-2018_json/6815705/1}, DOI={10.6084/m9.figshare.6815705.v1}, abstractNote={2D materials}, publisher={figshare}, author={choudhary, kamal and https://orcid.org/0000-0001-9737-8074}, year={2018}, month={Jul}}




-------------
jarvis_dft_3d
-------------
Various properties of 25,923 bulk materials computed with the OptB88vdW and TBmBJ functionals taken from the JARVIS DFT database.

**Number of entries:** 25923

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`bulk modulus`
     - VRH average calculation of bulk modulus
   * - :code:`composition`
     - A Pymatgen Composition descriptor of the composition of the material
   * - :code:`e_form`
     - formation energy per atom, in eV/atom
   * - :code:`epsilon_x opt`
     - Static dielectric function in x direction calculated with OptB88vDW functional.
   * - :code:`epsilon_x tbmbj`
     - Static dielectric function in x direction calculuated with TBMBJ functional.
   * - :code:`epsilon_y opt`
     - Static dielectric function in y direction calculated with OptB88vDW functional.
   * - :code:`epsilon_y tbmbj`
     - Static dielectric function in y direction calculuated with TBMBJ functional.
   * - :code:`epsilon_z opt`
     - Static dielectric function in z direction calculated with OptB88vDW functional.
   * - :code:`epsilon_z tbmbj`
     - Static dielectric function in z direction calculuated with TBMBJ functional.
   * - :code:`gap opt`
     - Band gap calculated with OptB88vDW functional, in eV
   * - :code:`gap tbmbj`
     - Band gap calculated with TBMBJ functional, in eV
   * - :code:`jid`
     - JARVIS ID
   * - :code:`mpid`
     - Materials Project ID
   * - :code:`shear modulus`
     - VRH average calculation of shear modulus
   * - :code:`structure`
     - A description of the crystal structure of the material
   * - :code:`structure initial`
     - Initial structure description of the crystal structure of the material



**Reference**

3D Dataset discussed in:
Elastic properties of bulk and low-dimensional materials using van der Waals density functional Kamal Choudhary, Gowoon Cheon, Evan Reed, and Francesca Tavazza Phys. Rev. B 98, 014107
Original 3D Data file sourced from:
choudhary, kamal; https://orcid.org/0000-0001-9737-8074 (2018): jdft_3d.json. figshare. Dataset.



**Bibtex Formatted Citations**

@article{PhysRevB.98.014107,
title = {Elastic properties of bulk and low-dimensional materials using van der Waals density functional},
author = {Choudhary, Kamal and Cheon, Gowoon and Reed, Evan and Tavazza, Francesca},
journal = {Phys. Rev. B},
volume = {98},
issue = {1},
pages = {014107},
numpages = {12},
year = {2018},
month = {Jul},
publisher = {American Physical Society},
doi = {10.1103/PhysRevB.98.014107},
url = {https://link.aps.org/doi/10.1103/PhysRevB.98.014107}
}

@misc{choudhary__2018, title={jdft_3d.json}, url={https://figshare.com/articles/jdft_3d-7-7-2018_json/6815699/2}, DOI={10.6084/m9.figshare.6815699.v2}, abstractNote={https://jarvis.nist.gov/
The Density functional theory section of JARVIS (JARVIS-DFT) consists of thousands of VASP based calculations for 3D-bulk, single layer (2D), nanowire (1D) and molecular (0D) systems. Most of the calculations are carried out with optB88vDW functional. JARVIS-DFT includes materials data such as: energetics, diffraction pattern, radial distribution function, band-structure, density of states, carrier effective mass, temperature and carrier concentration dependent thermoelectric properties, elastic constants and gamma-point phonons.}, publisher={figshare}, author={choudhary, kamal and https://orcid.org/0000-0001-9737-8074}, year={2018}, month={Jul}}




----------------------
jarvis_ml_dft_training
----------------------
Various properties of 24,759 bulk and 2D materials computed with the OptB88vdW and TBmBJ functionals taken from the JARVIS DFT database.

**Number of entries:** 24759

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`bulk modulus`
     - VRH average calculation of bulk modulus
   * - :code:`composition`
     - A descriptor of the composition of the material
   * - :code:`e mass_x`
     - Effective electron mass in x direction (BoltzTraP)
   * - :code:`e mass_y`
     - Effective electron mass in y direction (BoltzTraP)
   * - :code:`e mass_z`
     - Effective electron mass in z direction (BoltzTraP)
   * - :code:`e_exfol`
     - exfoliation energy per atom in eV/atom
   * - :code:`e_form`
     - formation energy per atom, in eV/atom
   * - :code:`epsilon_x opt`
     - Static dielectric function in x direction calculated with OptB88vDW functional.
   * - :code:`epsilon_x tbmbj`
     - Static dielectric function in x direction calculated with TBMBJ functional.
   * - :code:`epsilon_y opt`
     - Static dielectric function in y direction calculated with OptB88vDW functional.
   * - :code:`epsilon_y tbmbj`
     - Static dielectric function in y direction calculated with TBMBJ functional.
   * - :code:`epsilon_z opt`
     - Static dielectric function in z direction calculated with OptB88vDW functional.
   * - :code:`epsilon_z tbmbj`
     - Static dielectric function in z direction calculated with TBMBJ functional.
   * - :code:`gap opt`
     - Band gap calculated with OptB88vDW functional, in eV
   * - :code:`gap tbmbj`
     - Band gap calculated with TBMBJ functional, in eV
   * - :code:`hole mass_x`
     - Effective hole mass in x direction (BoltzTraP)
   * - :code:`hole mass_y`
     - Effective hole mass in y direction (BoltzTraP)
   * - :code:`hole mass_z`
     - Effective hole mass in z direction (BoltzTraP)
   * - :code:`jid`
     - JARVIS ID
   * - :code:`mpid`
     - Materials Project ID
   * - :code:`mu_b`
     - Magnetic moment, in Bohr Magneton
   * - :code:`shear modulus`
     - VRH average calculation of shear modulus
   * - :code:`structure`
     - A Pymatgen Structure object describing the crystal structure of the material



**Reference**

Dataset discussed in:
Machine learning with force-field-inspired descriptors for materials: Fast screening and mapping energy landscape Kamal Choudhary, Brian DeCost, and Francesca Tavazza Phys. Rev. Materials 2, 083801

Original Data file sourced from:
choudhary, kamal (2018): JARVIS-ML-CFID-descriptors and material properties. figshare. Dataset.



**Bibtex Formatted Citations**

@article{PhysRevMaterials.2.083801,
title = {Machine learning with force-field-inspired descriptors for materials: Fast screening and mapping energy landscape},
author = {Choudhary, Kamal and DeCost, Brian and Tavazza, Francesca},
journal = {Phys. Rev. Materials},
volume = {2},
issue = {8},
pages = {083801},
numpages = {8},
year = {2018},
month = {Aug},
publisher = {American Physical Society},
doi = {10.1103/PhysRevMaterials.2.083801},
url = {https://link.aps.org/doi/10.1103/PhysRevMaterials.2.083801}
}

@misc{choudhary_2018, title={JARVIS-ML-CFID-descriptors and material properties}, url={https://figshare.com/articles/JARVIS-ML-CFID-descriptors_and_material_properties/6870101/1}, DOI={10.6084/m9.figshare.6870101.v1}, abstractNote={Classical force-field inspired descriptors (CFID) for more than 25000 materials and their material properties such as bandgap, formation energies, modulus of elasticity etc. See JARVIS-ML: https://jarvis.nist.gov/}, publisher={figshare}, author={choudhary, kamal}, year={2018}, month={Jul}}




----
m2ax
----
Elastic properties of 223 stable M2AX compounds from "A comprehensive survey of M2AX phase elastic properties" by Cover et al. Calculations are PAW PW91.

**Number of entries:** 223

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`a`
     - Lattice parameter a, in A (angstrom)
   * - :code:`bulk modulus`
     - In GPa
   * - :code:`c`
     - lattice parameter c, in A (angstrom)
   * - :code:`c11`
     - Elastic constants of the M2AX material. These are specific to hexagonal materials.
   * - :code:`c12`
     - Elastic constants of the M2AX material. These are specific to hexagonal materials.
   * - :code:`c13`
     - Elastic constants of the M2AX material. These are specific to hexagonal materials.
   * - :code:`c33`
     - Elastic constants of the M2AX material. These are specific to hexagonal materials.
   * - :code:`c44`
     - Elastic constants of the M2AX material. These are specific to hexagonal materials.
   * - :code:`d_ma`
     - distance from the M atom to the A atom
   * - :code:`d_mx`
     - distance from the M atom to the X atom
   * - :code:`elastic modulus`
     - In GPa
   * - :code:`formula`
     - chemical formula
   * - :code:`shear modulus`
     - In GPa



**Reference**

http://iopscience.iop.org/article/10.1088/0953-8984/21/30/305403/meta



**Bibtex Formatted Citations**

@article{M F Cover,
author={M F Cover and O Warschkow and M M M Bilek and D R McKenzie},
title={A comprehensive survey of M 2 AX phase elastic properties},
journal={Journal of Physics: Condensed Matter},
volume={21},
number={30},
pages={305403},
url={http://stacks.iop.org/0953-8984/21/i=30/a=305403},
year={2009},
abstract={M 2 AX phases are a family of nanolaminate, ternary alloys that are composed of slabs of transition metal carbide or nitride (M 2 X) separated by single atomic layers of a main group element. In this combination, they manifest many of the beneficial properties of both ceramic and metallic compounds, making them attractive for many technological applications. We report here the results of a large scale computational survey of the elastic properties of all 240 elemental combinations using first-principles density functional theory calculations. We found correlations revealing the governing role of the A element and its interaction with the M element on the c axis compressibility and shearability of the material. The role of the X element is relatively minor, with the strongest effect seen in the in-plane constants C 11 and C 12 . We identify several elemental compositions with extremal properties such as W 2 SnC, which has by far the lowest value of C 44 , suggesting potential applications as a...}}




-------------------
matbench_dielectric
-------------------
Matbench v0.1 test dataset for predicting refractive index from structure. Adapted from Materials Project database. Removed entries having a formation energy (or energy above the convex hull) more than 150meV and those having refractive indices less than 1 and those containing noble gases. Retrieved April 2, 2019.

**Number of entries:** 4764

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`n`
     - Target variable. Refractive index (unitless).
   * - :code:`structure`
     - Pymatgen Structure of the material.



**Reference**

Petousis, I., Mrdjenovich, D., Ballouz, E., Liu, M., Winston, D.,
Chen, W., Graf, T., Schladt, T. D., Persson, K. A. & Prinz, F. B.
High-throughput screening of inorganic compounds for the discovery
of novel dielectric and optical materials. Sci. Data 4, 160134 (2017).



**Bibtex Formatted Citations**

@article{Jain2013,
author = {Jain, Anubhav and Ong, Shyue Ping and Hautier, Geoffroy and Chen, Wei and Richards, William Davidson and Dacek, Stephen and Cholia, Shreyas and Gunter, Dan and Skinner, David and Ceder, Gerbrand and Persson, Kristin a.},
doi = {10.1063/1.4812323},
issn = {2166532X},
journal = {APL Materials},
number = {1},
pages = {011002},
title = {{The Materials Project: A materials genome approach to accelerating materials innovation}},
url = {http://link.aip.org/link/AMPADS/v1/i1/p011002/s1\&Agg=doi},
volume = {1},
year = {2013}
}

@article{Petousis2017,
author={Petousis, Ioannis and Mrdjenovich, David and Ballouz, Eric
and Liu, Miao and Winston, Donald and Chen, Wei and Graf, Tanja
and Schladt, Thomas D. and Persson, Kristin A. and Prinz, Fritz B.},
title={High-throughput screening of inorganic compounds for the
discovery of novel dielectric and optical materials},
journal={Scientific Data},
year={2017},
month={Jan},
day={31},
publisher={The Author(s)},
volume={4},
pages={160134},
note={Data Descriptor},
url={http://dx.doi.org/10.1038/sdata.2016.134}
}




-----------------
matbench_expt_gap
-----------------
Matbench v0.1 test dataset for predicting experimental band gap from composition alone. Retrieved from Zhuo et al. supplementary information. Deduplicated according to composition, removing compositions with reported band gaps spanning more than a 0.1eV range; remaining compositions were assigned values based on the closest experimental value to the mean experimental value for that composition among all reports.

**Number of entries:** 4604

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`composition`
     - Chemical formula.
   * - :code:`gap expt`
     - Target variable. Experimentally measured gap, in eV.



**Reference**

Y. Zhuo, A. Masouri Tehrani, J. Brgoch (2018) Predicting the Band Gaps of Inorganic Solids by Machine Learning J. Phys. Chem. Lett. 2018, 9, 7, 1668-1673 https:doi.org/10.1021/acs.jpclett.8b00124.



**Bibtex Formatted Citations**

@article{doi:10.1021/acs.jpclett.8b00124,
author = {Zhuo, Ya and Mansouri Tehrani, Aria and Brgoch, Jakoah},
title = {Predicting the Band Gaps of Inorganic Solids by Machine Learning},
journal = {The Journal of Physical Chemistry Letters},
volume = {9},
number = {7},
pages = {1668-1673},
year = {2018},
doi = {10.1021/acs.jpclett.8b00124},
note ={PMID: 29532658},
eprint = {
https://doi.org/10.1021/acs.jpclett.8b00124

}}




----------------------
matbench_expt_is_metal
----------------------
Matbench v0.1 test dataset for classifying metallicity from composition alone. Retrieved from Zhuo et al. supplementary information. Deduplicated according to composition, ensuring no conflicting reports were entered for any compositions (i.e., no reported compositions were both metal and nonmetal).

**Number of entries:** 4921

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`composition`
     - Chemical formula.
   * - :code:`is_metal`
     - Target variable. 1 if is a metal, 0 if nonmetal.



**Reference**

Y. Zhuo, A. Masouri Tehrani, J. Brgoch (2018) Predicting the Band Gaps of Inorganic Solids by Machine Learning J. Phys. Chem. Lett. 2018, 9, 7, 1668-1673 
 https//:doi.org/10.1021/acs.jpclett.8b00124.



**Bibtex Formatted Citations**

@article{doi:10.1021/acs.jpclett.8b00124,
author = {Zhuo, Ya and Mansouri Tehrani, Aria and Brgoch, Jakoah},
title = {Predicting the Band Gaps of Inorganic Solids by Machine Learning},
journal = {The Journal of Physical Chemistry Letters},
volume = {9},
number = {7},
pages = {1668-1673},
year = {2018},
doi = {10.1021/acs.jpclett.8b00124},
note ={PMID: 29532658},
eprint = {
https://doi.org/10.1021/acs.jpclett.8b00124

}}




--------------
matbench_glass
--------------
Matbench v0.1 test dataset for predicting full bulk metallic glass formation ability from chemical formula. Retrieved from "Nonequilibrium Phase Diagrams of Ternary Amorphous Alloys,’ a volume of the Landolt– Börnstein collection. Deduplicated according to composition, ensuring no compositions were reported as both GFA and not GFA (i.e., all reports agreed on the classification designation).

**Number of entries:** 5680

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`composition`
     - Chemical formula.
   * - :code:`gfa`
     - Target variable. Glass forming ability: 1 means glass forming and corresponds to amorphous, 0 means non full glass forming.



**Reference**

Y. Kawazoe, T. Masumoto, A.-P. Tsai, J.-Z. Yu, T. Aihara Jr. (1997) Y. Kawazoe, J.-Z. Yu, A.-P. Tsai, T. Masumoto (ed.) SpringerMaterials
Nonequilibrium Phase Diagrams of Ternary Amorphous Alloys · 1 Introduction Landolt-Börnstein - Group III Condensed Matter 37A (Nonequilibrium Phase Diagrams of Ternary Amorphous Alloys) https://www.springer.com/gp/book/9783540605072 (Springer-Verlag Berlin Heidelberg © 1997) Accessed: 03-09-2019



**Bibtex Formatted Citations**

@Misc{LandoltBornstein1997:sm_lbs_978-3-540-47679-5_2,
author="Kawazoe, Y.
and Masumoto, T.
and Tsai, A.-P.
and Yu, J.-Z.
and Aihara Jr., T.",
editor="Kawazoe, Y.
and Yu, J.-Z.
and Tsai, A.-P.
and Masumoto, T.",
title="Nonequilibrium Phase Diagrams of Ternary Amorphous Alloys {\textperiodcentered} 1 Introduction: Datasheet from Landolt-B{\"o}rnstein - Group III Condensed Matter {\textperiodcentered} Volume 37A: ``Nonequilibrium Phase Diagrams of Ternary Amorphous Alloys'' in SpringerMaterials (https://dx.doi.org/10.1007/10510374{\_}2)",
publisher="Springer-Verlag Berlin Heidelberg",
note="Copyright 1997 Springer-Verlag Berlin Heidelberg",
note="Part of SpringerMaterials",
note="accessed 2018-10-23",
doi="10.1007/10510374_2",
url="https://materials.springer.com/lb/docs/sm_lbs_978-3-540-47679-5_2"
}

@Article{Ward2016,
author={Ward, Logan
and Agrawal, Ankit
and Choudhary, Alok
and Wolverton, Christopher},
title={A general-purpose machine learning framework for predicting properties of inorganic materials},
journal={Npj Computational Materials},
year={2016},
month={Aug},
day={26},
publisher={The Author(s)},
volume={2},
pages={16028},
note={Article},
url={http://dx.doi.org/10.1038/npjcompumats.2016.28}
}




---------------
matbench_jdft2d
---------------
Matbench v0.1 test dataset for predicting exfoliation energies from crystal structure (computed with the OptB88vdW and TBmBJ functionals). Adapted from the JARVIS DFT database.

**Number of entries:** 636

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`exfoliation_en`
     - Target variable. Exfoliation energy (meV).
   * - :code:`structure`
     - Pymatgen Structure of the material.



**Reference**

2D Dataset discussed in:
High-throughput Identification and Characterization of Two dimensional Materials using Density functional theory Kamal Choudhary, Irina Kalish, Ryan Beams & Francesca Tavazza Scientific Reports volume 7, Article number: 5179 (2017)
Original 2D Data file sourced from:
choudhary, kamal; https://orcid.org/0000-0001-9737-8074 (2018): jdft_2d-7-7-2018.json. figshare. Dataset.



**Bibtex Formatted Citations**

@Article{Choudhary2017,
author={Choudhary, Kamal
and Kalish, Irina
and Beams, Ryan
and Tavazza, Francesca},
title={High-throughput Identification and Characterization of Two-dimensional Materials using Density functional theory},
journal={Scientific Reports},
year={2017},
volume={7},
number={1},
pages={5179},
abstract={We introduce a simple criterion to identify two-dimensional (2D) materials based on the comparison between experimental lattice constants and lattice constants mainly obtained from Materials-Project (MP) density functional theory (DFT) calculation repository. Specifically, if the relative difference between the two lattice constants for a specific material is greater than or equal to 5%, we predict them to be good candidates for 2D materials. We have predicted at least 1356 such 2D materials. For all the systems satisfying our criterion, we manually create single layer systems and calculate their energetics, structural, electronic, and elastic properties for both the bulk and the single layer cases. Currently the database consists of 1012 bulk and 430 single layer materials, of which 371 systems are common to bulk and single layer. The rest of calculations are underway. To validate our criterion, we calculated the exfoliation energy of the suggested layered materials, and we found that in 88.9% of the cases the currently accepted criterion for exfoliation was satisfied. Also, using molybdenum telluride as a test case, we performed X-ray diffraction and Raman scattering experiments to benchmark our calculations and understand their applicability and limitations. The data is publicly available at the website http://www.ctcms.nist.gov/{	extasciitilde}knc6/JVASP.html.},
issn={2045-2322},
doi={10.1038/s41598-017-05402-0},
url={https://doi.org/10.1038/s41598-017-05402-0}
}

@misc{choudhary__2018, title={jdft_2d-7-7-2018.json}, url={https://figshare.com/articles/jdft_2d-7-7-2018_json/6815705/1}, DOI={10.6084/m9.figshare.6815705.v1}, abstractNote={2D materials}, publisher={figshare}, author={choudhary, kamal and https://orcid.org/0000-0001-9737-8074}, year={2018}, month={Jul}}




-----------------
matbench_log_gvrh
-----------------
Matbench v0.1 test dataset for predicting DFT log10 VRH-average shear modulus from structure. Adapted from Materials Project database. Removed entries having a formation energy (or energy above the convex hull) more than 150meV and those having negative G_Voigt, G_Reuss, G_VRH, K_Voigt, K_Reuss, or K_VRH and those failing G_Reuss <= G_VRH <= G_Voigt or K_Reuss <= K_VRH <= K_Voigt and those containing noble gases. Retrieved April 2, 2019.

**Number of entries:** 10987

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`log10(G_VRH)`
     - Target variable. Base 10 logarithm of the DFT Voigt-Reuss-Hill average shear moduli in GPa
   * - :code:`structure`
     - Pymatgen Structure of the material.



**Reference**

Jong, M. De, Chen, W., Angsten, T., Jain, A., Notestine, R., Gamst,
A., Sluiter, M., Ande, C. K., Zwaag, S. Van Der, Plata, J. J., Toher,
C., Curtarolo, S., Ceder, G., Persson, K. and Asta, M., "Charting
the complete elastic properties of inorganic crystalline compounds",
Scientific Data volume 2, Article number: 150009 (2015)



**Bibtex Formatted Citations**

@Article{deJong2015,
author={de Jong, Maarten and Chen, Wei and Angsten, Thomas
and Jain, Anubhav and Notestine, Randy and Gamst, Anthony
and Sluiter, Marcel and Krishna Ande, Chaitanya
and van der Zwaag, Sybrand and Plata, Jose J. and Toher, Cormac
and Curtarolo, Stefano and Ceder, Gerbrand and Persson, Kristin A.
and Asta, Mark},
title={Charting the complete elastic properties
of inorganic crystalline compounds},
journal={Scientific Data},
year={2015},
month={Mar},
day={17},
publisher={The Author(s)},
volume={2},
pages={150009},
note={Data Descriptor},
url={http://dx.doi.org/10.1038/sdata.2015.9}
}




-----------------
matbench_log_kvrh
-----------------
Matbench v0.1 test dataset for predicting DFT log10 VRH-average bulk modulus from structure. Adapted from Materials Project database. Removed entries having a formation energy (or energy above the convex hull) more than 150meV and those having negative G_Voigt, G_Reuss, G_VRH, K_Voigt, K_Reuss, or K_VRH and those failing G_Reuss <= G_VRH <= G_Voigt or K_Reuss <= K_VRH <= K_Voigt and those containing noble gases. Retrieved April 2, 2019.

**Number of entries:** 10987

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`log10(K_VRH)`
     - Target variable. Base 10 logarithm of the DFT Voigt-Reuss-Hill average bulk moduli in GPa.
   * - :code:`structure`
     - Pymatgen Structure of the material.



**Reference**

Jong, M. De, Chen, W., Angsten, T., Jain, A., Notestine, R., Gamst,
A., Sluiter, M., Ande, C. K., Zwaag, S. Van Der, Plata, J. J., Toher,
C., Curtarolo, S., Ceder, G., Persson, K. and Asta, M., "Charting
the complete elastic properties of inorganic crystalline compounds",
Scientific Data volume 2, Article number: 150009 (2015)



**Bibtex Formatted Citations**

@Article{deJong2015,
author={de Jong, Maarten and Chen, Wei and Angsten, Thomas
and Jain, Anubhav and Notestine, Randy and Gamst, Anthony
and Sluiter, Marcel and Krishna Ande, Chaitanya
and van der Zwaag, Sybrand and Plata, Jose J. and Toher, Cormac
and Curtarolo, Stefano and Ceder, Gerbrand and Persson, Kristin A.
and Asta, Mark},
title={Charting the complete elastic properties
of inorganic crystalline compounds},
journal={Scientific Data},
year={2015},
month={Mar},
day={17},
publisher={The Author(s)},
volume={2},
pages={150009},
note={Data Descriptor},
url={http://dx.doi.org/10.1038/sdata.2015.9}
}




------------------
matbench_mp_e_form
------------------
Matbench v0.1 test dataset for predicting DFT formation energy from structure. Adapted from Materials Project database. Removed entries having formation energy more than 3.0eV and those containing noble gases. Retrieved April 2, 2019.

**Number of entries:** 132752

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`e_form`
     - Target variable. Formation energy in eV as calculated by the Materials Project.
   * - :code:`structure`
     - Pymatgen Structure of the material.



**Reference**

A. Jain*, S.P. Ong*, G. Hautier, W. Chen, W.D. Richards, S. Dacek, S. Cholia, D. Gunter, D. Skinner, G. Ceder, K.A. Persson (*=equal contributions)
The Materials Project: A materials genome approach to accelerating materials innovation
APL Materials, 2013, 1(1), 011002.
doi:10.1063/1.4812323



**Bibtex Formatted Citations**

@article{Jain2013,
author = {Jain, Anubhav and Ong, Shyue Ping and Hautier, Geoffroy and Chen, Wei and Richards, William Davidson and Dacek, Stephen and Cholia, Shreyas and Gunter, Dan and Skinner, David and Ceder, Gerbrand and Persson, Kristin a.},
doi = {10.1063/1.4812323},
issn = {2166532X},
journal = {APL Materials},
number = {1},
pages = {011002},
title = {{The Materials Project: A materials genome approach to accelerating materials innovation}},
url = {http://link.aip.org/link/AMPADS/v1/i1/p011002/s1\&Agg=doi},
volume = {1},
year = {2013}
}




---------------
matbench_mp_gap
---------------
Matbench v0.1 test dataset for predicting DFT PBE band gap from structure. Adapted from Materials Project database. Removed entries having a formation energy (or energy above the convex hull) more than 150meV and those containing noble gases. Retrieved April 2, 2019.

**Number of entries:** 106113

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`gap pbe`
     - Target variable. The band gap as calculated by PBE DFT from the Materials Project, in eV.
   * - :code:`structure`
     - Pymatgen Structure of the material.



**Reference**

A. Jain*, S.P. Ong*, G. Hautier, W. Chen, W.D. Richards, S. Dacek, S. Cholia, D. Gunter, D. Skinner, G. Ceder, K.A. Persson (*=equal contributions)
The Materials Project: A materials genome approach to accelerating materials innovation
APL Materials, 2013, 1(1), 011002.
doi:10.1063/1.4812323



**Bibtex Formatted Citations**

@article{Jain2013,
author = {Jain, Anubhav and Ong, Shyue Ping and Hautier, Geoffroy and Chen, Wei and Richards, William Davidson and Dacek, Stephen and Cholia, Shreyas and Gunter, Dan and Skinner, David and Ceder, Gerbrand and Persson, Kristin a.},
doi = {10.1063/1.4812323},
issn = {2166532X},
journal = {APL Materials},
number = {1},
pages = {011002},
title = {{The Materials Project: A materials genome approach to accelerating materials innovation}},
url = {http://link.aip.org/link/AMPADS/v1/i1/p011002/s1\&Agg=doi},
volume = {1},
year = {2013}
}




--------------------
matbench_mp_is_metal
--------------------
Matbench v0.1 test dataset for predicting DFT metallicity from structure. Adapted from Materials Project database. Removed entries having a formation energy (or energy above the convex hull) more than 150meV and those containing noble gases.. Retrieved April 2, 2019.

**Number of entries:** 106113

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`is_metal`
     - Target variable. 1 if the compound is a metal, 0 if the compound is not a metal. Metallicity determined with pymatgen
   * - :code:`structure`
     - Pymatgen Structure of the material.



**Reference**

A. Jain*, S.P. Ong*, G. Hautier, W. Chen, W.D. Richards, S. Dacek, S. Cholia, D. Gunter, D. Skinner, G. Ceder, K.A. Persson (*=equal contributions)
The Materials Project: A materials genome approach to accelerating materials innovation
APL Materials, 2013, 1(1), 011002.
doi:10.1063/1.4812323



**Bibtex Formatted Citations**

@article{Jain2013,
author = {Jain, Anubhav and Ong, Shyue Ping and Hautier, Geoffroy and Chen, Wei and Richards, William Davidson and Dacek, Stephen and Cholia, Shreyas and Gunter, Dan and Skinner, David and Ceder, Gerbrand and Persson, Kristin a.},
doi = {10.1063/1.4812323},
issn = {2166532X},
journal = {APL Materials},
number = {1},
pages = {011002},
title = {{The Materials Project: A materials genome approach to accelerating materials innovation}},
url = {http://link.aip.org/link/AMPADS/v1/i1/p011002/s1\&Agg=doi},
volume = {1},
year = {2013}
}




--------------------
matbench_perovskites
--------------------
Matbench v0.1 test dataset for predicting formation energy from crystal structure. Adapted from an original dataset generated by Castelli et al.

**Number of entries:** 18928

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`e_form`
     - Target variable. Heat of formation of the perovskite, in eV as calculated by PBE GGA-DFT.
   * - :code:`structure`
     - Pymatgen Structure of the material.



**Reference**

Ivano E. Castelli, David D. Landis, Kristian S. Thygesen, Søren Dahl, Ib Chorkendorff, Thomas F. Jaramillo and Karsten W. Jacobsen (2012) New cubic perovskites for one- and two-photon water splitting using the computational materials repository. Energy Environ. Sci., 2012,5, 9034-9043 https://doi.org/10.1039/C2EE22341D



**Bibtex Formatted Citations**

@Article{C2EE22341D,
author ="Castelli, Ivano E. and Landis, David D. and Thygesen, Kristian S. and Dahl, Søren and Chorkendorff, Ib and Jaramillo, Thomas F. and Jacobsen, Karsten W.",
title  ="New cubic perovskites for one- and two-photon water splitting using the computational materials repository",
journal  ="Energy Environ. Sci.",
year  ="2012",
volume  ="5",
issue  ="10",
pages  ="9034-9043",
publisher  ="The Royal Society of Chemistry",
doi  ="10.1039/C2EE22341D",
url  ="http://dx.doi.org/10.1039/C2EE22341D",
abstract  ="A new efficient photoelectrochemical cell (PEC) is one of the possible solutions to the energy and climate problems of our time. Such a device requires development of new semiconducting materials with tailored properties with respect to stability and light absorption. Here we perform computational screening of around 19 000 oxides{,} oxynitrides{,} oxysulfides{,} oxyfluorides{,} and oxyfluoronitrides in the cubic perovskite structure with PEC applications in mind. We address three main applications: light absorbers for one- and two-photon water splitting and high-stability transparent shields to protect against corrosion. We end up with 20{,} 12{,} and 15 different combinations of oxides{,} oxynitrides and oxyfluorides{,} respectively{,} inviting further experimental investigation."}




----------------
matbench_phonons
----------------
Matbench v0.1 test dataset for predicting vibration properties from crystal structure. Original data retrieved from Petretto et al. Original calculations done via ABINIT in the harmonic approximation based on density functional perturbation theory. Removed entries having a formation energy (or energy above the convex hull) more than 150meV.

**Number of entries:** 1296

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`last phdos peak`
     - Target variable. Frequency of the highest frequency optical phonon mode peak, in units of 1/cm; ; may be used as an estimation of dominant longitudinal optical phonon frequency.
   * - :code:`structure`
     - Pymatgen Structure of the material.



**Reference**

Petretto, G. et al. High-throughput density functional perturbation theory phonons for inorganic materials. Sci. Data 5:180065 doi: 10.1038/sdata.2018.65 (2018).
Petretto, G. et al. High-throughput density functional perturbation theory phonons for inorganic materials. (2018). figshare. Collection.



**Bibtex Formatted Citations**

@Article{Petretto2018,
author={Petretto, Guido
and Dwaraknath, Shyam
and P.C. Miranda, Henrique
and Winston, Donald
and Giantomassi, Matteo
and van Setten, Michiel J.
and Gonze, Xavier
and Persson, Kristin A.
and Hautier, Geoffroy
and Rignanese, Gian-Marco},
title={High-throughput density-functional perturbation theory phonons for inorganic materials},
journal={Scientific Data},
year={2018},
month={May},
day={01},
publisher={The Author(s)},
volume={5},
pages={180065},
note={Data Descriptor},
url={http://dx.doi.org/10.1038/sdata.2018.65}
}

@misc{petretto_dwaraknath_miranda_winston_giantomassi_rignanese_van setten_gonze_persson_hautier_2018, title={High-throughput Density-Functional Perturbation Theory phonons for inorganic materials}, url={https://figshare.com/collections/High-throughput_Density-Functional_Perturbation_Theory_phonons_for_inorganic_materials/3938023/1}, DOI={10.6084/m9.figshare.c.3938023.v1}, abstractNote={The knowledge of the vibrational properties of a material is of key importance to understand physical phenomena such as thermal conductivity, superconductivity, and ferroelectricity among others. However, detailed experimental phonon spectra are available only for a limited number of materials which hinders the large-scale analysis of vibrational properties and their derived quantities. In this work, we perform ab initio calculations of the full phonon dispersion and vibrational density of states for 1521 semiconductor compounds in the harmonic approximation based on density functional perturbation theory. The data is collected along with derived dielectric and thermodynamic properties. We present the procedure used to obtain the results, the details of the provided database and a validation based on the comparison with experimental data.}, publisher={figshare}, author={Petretto, Guido and Dwaraknath, Shyam and Miranda, Henrique P. C. and Winston, Donald and Giantomassi, Matteo and Rignanese, Gian-Marco and Van Setten, Michiel J. and Gonze, Xavier and Persson, Kristin A and Hautier, Geoffroy}, year={2018}, month={Apr}}




---------------
matbench_steels
---------------
Matbench v0.1 dataset for predicting steel yield strengths from chemical composition alone. Retrieved from Citrine informatics. Deduplicated.

**Number of entries:** 312

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`composition`
     - Chemical formula.
   * - :code:`yield strength`
     - Target variable. Experimentally measured steel yield strengths, in GPa.



**Reference**

https://citrination.com/datasets/153092/



**Bibtex Formatted Citations**

@misc{Citrine Informatics,
title = {Mechanical properties of some steels},
howpublished = {\url{https://citrination.com/datasets/153092/},
}




------
mp_all
------
A complete copy of the Materials Project database as of 10/18/2018. Mp_all files contain structure data for each material while mp_nostruct does not.

**Number of entries:** 83989

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`bulk modulus`
     - in GPa, average of Voight, Reuss, and Hill
   * - :code:`e_form`
     - Formation energy per atom (eV)
   * - :code:`e_hull`
     - The calculated energy above the convex hull, in eV per atom
   * - :code:`elastic anisotropy`
     - The ratio of elastic anisotropy.
   * - :code:`formula`
     - The chemical formula of the MP entry
   * - :code:`gap pbe`
     - The band gap in eV calculated with PBE-DFT functional
   * - :code:`initial structure`
     - A Pymatgen Structure object describing the material crystal structure prior to relaxation
   * - :code:`mpid`
     - (input): The Materials Project mpid, as a string.
   * - :code:`mu_b`
     - The total magnetization of the unit cell.
   * - :code:`shear modulus`
     - in GPa, average of Voight, Reuss, and Hill
   * - :code:`structure`
     - A Pymatgen Structure object describing the material crystal structure



**Reference**

A. Jain*, S.P. Ong*, G. Hautier, W. Chen, W.D. Richards, S. Dacek, S. Cholia, D. Gunter, D. Skinner, G. Ceder, K.A. Persson (*=equal contributions)
The Materials Project: A materials genome approach to accelerating materials innovation
APL Materials, 2013, 1(1), 011002.
doi:10.1063/1.4812323



**Bibtex Formatted Citations**

@article{Jain2013,
author = {Jain, Anubhav and Ong, Shyue Ping and Hautier, Geoffroy and Chen, Wei and Richards, William Davidson and Dacek, Stephen and Cholia, Shreyas and Gunter, Dan and Skinner, David and Ceder, Gerbrand and Persson, Kristin a.},
doi = {10.1063/1.4812323},
issn = {2166532X},
journal = {APL Materials},
number = {1},
pages = {011002},
title = {{The Materials Project: A materials genome approach to accelerating materials innovation}},
url = {http://link.aip.org/link/AMPADS/v1/i1/p011002/s1\&Agg=doi},
volume = {1},
year = {2013}
}




-----------
mp_nostruct
-----------
A complete copy of the Materials Project database as of 10/18/2018. Mp_all files contain structure data for each material while mp_nostruct does not.

**Number of entries:** 83989

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`bulk modulus`
     - in GPa, average of Voight, Reuss, and Hill
   * - :code:`e_form`
     - Formation energy per atom (eV)
   * - :code:`e_hull`
     - The calculated energy above the convex hull, in eV per atom
   * - :code:`elastic anisotropy`
     - The ratio of elastic anisotropy.
   * - :code:`formula`
     - The chemical formula of the MP entry
   * - :code:`gap pbe`
     - The band gap in eV calculated with PBE-DFT functional
   * - :code:`mpid`
     - (input): The Materials Project mpid, as a string.
   * - :code:`mu_b`
     - The total magnetization of the unit cell.
   * - :code:`shear modulus`
     - in GPa, average of Voight, Reuss, and Hill



**Reference**

A. Jain*, S.P. Ong*, G. Hautier, W. Chen, W.D. Richards, S. Dacek, S. Cholia, D. Gunter, D. Skinner, G. Ceder, K.A. Persson (*=equal contributions)
The Materials Project: A materials genome approach to accelerating materials innovation
APL Materials, 2013, 1(1), 011002.
doi:10.1063/1.4812323



**Bibtex Formatted Citations**

@article{Jain2013,
author = {Jain, Anubhav and Ong, Shyue Ping and Hautier, Geoffroy and Chen, Wei and Richards, William Davidson and Dacek, Stephen and Cholia, Shreyas and Gunter, Dan and Skinner, David and Ceder, Gerbrand and Persson, Kristin a.},
doi = {10.1063/1.4812323},
issn = {2166532X},
journal = {APL Materials},
number = {1},
pages = {011002},
title = {{The Materials Project: A materials genome approach to accelerating materials innovation}},
url = {http://link.aip.org/link/AMPADS/v1/i1/p011002/s1\&Agg=doi},
volume = {1},
year = {2013}
}




--------------------
phonon_dielectric_mp
--------------------
Phonon (lattice/atoms vibrations) and dielectric properties of 1296 compounds computed via ABINIT software package in the harmonic approximation based on density functional perturbation theory.

**Number of entries:** 1296

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`eps_electronic`
     - A target variable of the dataset, electronic contribution to the calculated dielectric constant; unitless.
   * - :code:`eps_total`
     - A target variable of the dataset, total calculated dielectric constant. Unitless: it is a ratio over the dielectric constant at vacuum.
   * - :code:`formula`
     - The chemical formula of the material
   * - :code:`last phdos peak`
     - A target variable of the dataset, the frequency of the last calculated phonon density of states in 1/cm; may be used as an estimation of dominant longitudinal optical phonon frequency, a descriptor.
   * - :code:`mpid`
     - The Materials Project identifier for the material
   * - :code:`structure`
     - A pymatgen Structure object describing the chemical strucutre of the material



**Reference**

Petretto, G. et al. High-throughput density functional perturbation theory phonons for inorganic materials. Sci. Data 5:180065 doi: 10.1038/sdata.2018.65 (2018).
Petretto, G. et al. High-throughput density functional perturbation theory phonons for inorganic materials. (2018). figshare. Collection.



**Bibtex Formatted Citations**

@Article{Petretto2018,
author={Petretto, Guido
and Dwaraknath, Shyam
and P.C. Miranda, Henrique
and Winston, Donald
and Giantomassi, Matteo
and van Setten, Michiel J.
and Gonze, Xavier
and Persson, Kristin A.
and Hautier, Geoffroy
and Rignanese, Gian-Marco},
title={High-throughput density-functional perturbation theory phonons for inorganic materials},
journal={Scientific Data},
year={2018},
month={May},
day={01},
publisher={The Author(s)},
volume={5},
pages={180065},
note={Data Descriptor},
url={http://dx.doi.org/10.1038/sdata.2018.65}
}

@misc{petretto_dwaraknath_miranda_winston_giantomassi_rignanese_van setten_gonze_persson_hautier_2018, title={High-throughput Density-Functional Perturbation Theory phonons for inorganic materials}, url={https://figshare.com/collections/High-throughput_Density-Functional_Perturbation_Theory_phonons_for_inorganic_materials/3938023/1}, DOI={10.6084/m9.figshare.c.3938023.v1}, abstractNote={The knowledge of the vibrational properties of a material is of key importance to understand physical phenomena such as thermal conductivity, superconductivity, and ferroelectricity among others. However, detailed experimental phonon spectra are available only for a limited number of materials which hinders the large-scale analysis of vibrational properties and their derived quantities. In this work, we perform ab initio calculations of the full phonon dispersion and vibrational density of states for 1521 semiconductor compounds in the harmonic approximation based on density functional perturbation theory. The data is collected along with derived dielectric and thermodynamic properties. We present the procedure used to obtain the results, the details of the provided database and a validation based on the comparison with experimental data.}, publisher={figshare}, author={Petretto, Guido and Dwaraknath, Shyam and Miranda, Henrique P. C. and Winston, Donald and Giantomassi, Matteo and Rignanese, Gian-Marco and Van Setten, Michiel J. and Gonze, Xavier and Persson, Kristin A and Hautier, Geoffroy}, year={2018}, month={Apr}}




--------------------
piezoelectric_tensor
--------------------
941 structures with piezoelectric properties, calculated with DFT-PBE.

**Number of entries:** 941

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`cif`
     - optional: Description string for structure
   * - :code:`eij_max`
     - Piezoelectric modulus
   * - :code:`formula`
     - Chemical formula of the material
   * - :code:`material_id`
     - Materials Project ID of the material
   * - :code:`meta`
     - optional, metadata descriptor of the datapoint
   * - :code:`nsites`
     - The \# of atoms in the unit cell of the calculation.
   * - :code:`piezoelectric_tensor`
     - Tensor describing the piezoelectric properties of the material
   * - :code:`point_group`
     - Descriptor of crystallographic structure of the material
   * - :code:`poscar`
     - optional: Poscar metadata
   * - :code:`space_group`
     - Integer specifying the crystallographic structure of the material
   * - :code:`structure`
     - pandas Series defining the structure of the material
   * - :code:`v_max`
     - Crystallographic direction
   * - :code:`volume`
     - Volume of the unit cell in cubic angstroms, For supercell calculations, this quantity refers to the volume of the full supercell. 



**Reference**

de Jong, M., Chen, W., Geerlings, H., Asta, M. & Persson, K. A.
A database to enable discovery and design of piezoelectric materials.
Sci. Data 2, 150053 (2015)



**Bibtex Formatted Citations**

@Article{deJong2015,
author={de Jong, Maarten and Chen, Wei and Geerlings, Henry
and Asta, Mark and Persson, Kristin Aslaug},
title={A database to enable discovery and design of piezoelectric
materials},
journal={Scientific Data},
year={2015},
month={Sep},
day={29},
publisher={The Author(s)},
volume={2},
pages={150053},
note={Data Descriptor},
url={http://dx.doi.org/10.1038/sdata.2015.53}
}




--------------
steel_strength
--------------
312 steels with experimental yield strength and ultimate tensile strength, extracted and cleaned (including de-duplicating) from Citrine.

**Number of entries:** 312

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`al`
     - weight percent of Al
   * - :code:`c`
     - weight percent of C
   * - :code:`co`
     - weight percent of Co
   * - :code:`cr`
     - weight percent of Cr
   * - :code:`elongation`
     - elongation in %
   * - :code:`formula`
     - Chemical formula of the entry
   * - :code:`mn`
     - weight percent of Mn
   * - :code:`mo`
     - weight percent of Mo
   * - :code:`n`
     - weight percent of N
   * - :code:`nb`
     - weight percent of Nb
   * - :code:`ni`
     - weight percent of Ni
   * - :code:`si`
     - weight percent of Si
   * - :code:`tensile strength`
     - ultimate tensile strength in GPa
   * - :code:`ti`
     - weight percent of Ti
   * - :code:`v`
     - weight percent of V
   * - :code:`w`
     - weight percent of W
   * - :code:`yield strength`
     - yield strength in GPa



**Reference**

https://citrination.com/datasets/153092/



**Bibtex Formatted Citations**

@misc{Citrine Informatics,
title = {Mechanical properties of some steels},
howpublished = {\url{https://citrination.com/datasets/153092/},
}




----------------
wolverton_oxides
----------------
4,914 perovskite oxides containing composition data, lattice constants, and formation + vacancy formation energies. All perovskites are of the form ABO3. Adapted from a dataset presented by Emery and Wolverton.

**Number of entries:** 4914

.. list-table::
   :align: left
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - :code:`a`
     - Lattice parameter a, in A (angstrom)
   * - :code:`alpha`
     - Lattice angle alpha, in degrees
   * - :code:`atom a`
     - The atom in the 'A' site of the pervoskite.
   * - :code:`atom b`
     - The atom in the 'B' site of the perovskite.
   * - :code:`b`
     - Lattice parameter b, in A (angstrom)
   * - :code:`beta`
     - Lattice angle beta, in degrees
   * - :code:`c`
     - Lattice parameter c, in A (angstrom)
   * - :code:`e_form`
     - Formation energy in eV
   * - :code:`e_form oxygen`
     - Formation energy of oxygen vacancy (eV)
   * - :code:`e_hull`
     - Energy above convex hull, wrt. OQMD db (eV)
   * - :code:`formula`
     - Chemical formula of the entry
   * - :code:`gamma`
     - Lattice angle gamma, in degrees
   * - :code:`gap pbe`
     - Bandgap in eV from PBE calculations
   * - :code:`lowest distortion`
     - Local distortion crystal structure with lowest energy among all considered distortions.
   * - :code:`mu_b`
     - Magnetic moment
   * - :code:`vpa`
     - Volume per atom (A^3/atom)



**Reference**

Emery, A. A. & Wolverton, C. High-throughput DFT calculations of formation energy, stability and oxygen vacancy formation energy of ABO3 perovskites. Sci. Data 4:170153 doi: 10.1038/sdata.2017.153 (2017).
Emery, A. A., & Wolverton, C. Figshare http://dx.doi.org/10.6084/m9.figshare.5334142 (2017)



**Bibtex Formatted Citations**

@Article{Emery2017,
author={Emery, Antoine A.
and Wolverton, Chris},
title={High-throughput DFT calculations of formation energy, stability and oxygen vacancy formation energy of ABO3 perovskites},
journal={Scientific Data},
year={2017},
month={Oct},
day={17},
publisher={The Author(s)},
volume={4},
pages={170153},
note={Data Descriptor},
url={http://dx.doi.org/10.1038/sdata.2017.153}
}

@misc{emery_2017, title={High-throughput DFT calculations of formation energy, stability and oxygen vacancy formation energy of ABO3 perovskites}, url={https://figshare.com/articles/High-throughput_DFT_calculations_of_formation_energy_stability_and_oxygen_vacancy_formation_energy_of_ABO3_perovskites/5334142/1}, DOI={10.6084/m9.figshare.5334142.v1}, abstractNote={ABO3 perovskites are oxide materials that are used for a variety of applications such as solid oxide fuel cells, piezo-, ferro-electricity and water splitting. Due to their remarkable stability with respect to cation substitution, new compounds for such applications potentially await discovery. In this work, we present an exhaustive dataset of formation energies of 5,329 cubic and distorted perovskites that were calculated using first-principles density functional theory. In addition to formation energies, several additional properties such as oxidation states, band gap, oxygen vacancy formation energy, and thermodynamic stability with respect to all phases in the Open Quantum Materials Database are also made publicly available. This large dataset for this ubiquitous crystal structure type contains 395 perovskites that are predicted to be thermodynamically stable, of which many have not yet been experimentally reported, and therefore represent theoretical predictions. The dataset thus opens avenues for future use, including materials discovery in many research-active areas.}, publisher={figshare}, author={Emery, Antoine}, year={2017}, month={Aug}}




