==============================
Datasets Available in Matminer
==============================

Below you will find descriptions and reference data on each available dataset, ordered by load_dataset() keyword argument


------------
boltztrap_mp
------------
Effective mass and thermoelectric properties of 8924 compounds in The  Materials Project database that are calculated by the BoltzTraP software package run on the GGA-PBE or GGA+U density functional theory calculation results. The properties are reported at the temperature of 300 Kelvin and the carrier concentration of 1e18 1/cm3.

**Number of entries:** 8924

========= ======================================================================================================================
Column    Description
========= ======================================================================================================================
formula   Chemical formula of the entry
m_n       n-type/conduction band effective mass. Units: m_e where m_e is the mass of an electron; i.e. m_n is a unitless ratio
m_p       p-type/valence band effective mass.
mpid      Materials Project identifier
pf_n      n-type thermoelectric power factor in uW/cm2.K where uW is microwatts and a constant relaxation time of 1e-14 assumed.
pf_p      p-type power factor in uW/cm2.K
s_n       n-type Seebeck coefficient in micro Volts per Kelvin
s_p       p-type Seebeck coefficient in micro Volts per Kelvin
structure pymatgen Sttructure object describing the crystal structure of the material
========= ======================================================================================================================



**Reference**

Ricci, F. et al. An ab initio electronic transport database for inorganic materials. Sci. Data 4:170085 doi: 10.1038/sdata.2017.85 (2017).
Ricci F, Chen W, Aydemir U, Snyder J, Rignanese G, Jain A, Hautier G (2017) Data from: An ab initio electronic transport database for inorganic materials. Dryad Digital Repository. https://doi.org/10.5061/dryad.gn001



--------------------
castelli_perovskites
--------------------
18,928 perovskites generated with ABX combinatorics, calculating gbllsc band gap and pbe structure, and also reporting absolute band edge positions and heat of formation.

**Number of entries:** 18928

============= =====================================================================
Column        Description
============= =====================================================================
cbm           similar to vbm but for conduction band
e_form        heat of formation in eV
fermi level   the thermodynamic work required to add one electron to the body in eV
fermi width   fermi bandwidth
formula       Chemical formula of the material
gap gllbsc    electronic band gap in eV calculated via gllbsc functional
gap is direct boolean indicator for direct gap
mu_b          magnetic moment in terms of Bohr magneton
structure     crystal structure represented by pymatgen Structure object
vbm           absolute value of valence band edge calculated via gllbsc
============= =====================================================================



**Reference**

Ivano E. Castelli, David D. Landis, Kristian S. Thygesen, Søren Dahl, Ib Chorkendorff, Thomas F. Jaramillo and Karsten W. Jacobsen (2012) New cubic perovskites for one- and two-photon water splitting using the computational materials repository. Energy Environ. Sci., 2012,5, 9034-9043 https://doi.org/10.1039/C2EE22341D



----------------------------
citrine_thermal_conductivity
----------------------------
Thermal conductivity of 872 compounds measured experimentally and retrieved from Citrine database from various references. The reported values are measured at various temperatures of which 295 are at room temperature.

**Number of entries:** 872

================= =====================================================================
Column            Description
================= =====================================================================
formula           Chemical formula of the dataset entry
k-units           units of thermal conductivity
k_condition       Temperature description of testing conditions
k_condition_units units of testing condition temperature representation
k_expt            the experimentally measured thermal conductivity in SI units of W/m.K
================= =====================================================================



**Reference**

https://www.citrination.com



-------------------
dielectric_constant
-------------------
1,056 structures with dielectric properties, calculated with DFPT-PBE.

**Number of entries:** 1056

================= ==================================================================================================================================
Column            Description
================= ==================================================================================================================================
band_gap          Measure of the conductivity of a material
cif               optional: Description string for structure
e_electronic      electronic contribution to dielectric tensor
e_total           Total dielectric tensor incorporating both electronic and ionic contributions
formula           Chemical formula of the material
material_id       Materials Project ID of the material
meta              optional, metadata descriptor of the datapoint
n                 Refractive Index
nsites            The \# of atoms in the unit cell of the calculation.
poly_electronic   the average of the eigenvalues of the electronic contribution to the dielectric tensor
poly_total        the average of the eigenvalues of the total (electronic and ionic) contributions to the dielectric tensor
poscar            optional: Poscar metadata
pot_ferroelectric Whether the material is potentially ferroelectric
space_group       Integer specifying the crystallographic structure of the material
structure         pandas Series defining the structure of the material
volume            Volume of the unit cell in cubic angstroms, For supercell calculations, this quantity refers to the volume of the full supercell. 
================= ==================================================================================================================================



**Reference**

Petousis, I., Mrdjenovich, D., Ballouz, E., Liu, M., Winston, D.,
Chen, W., Graf, T., Schladt, T. D., Persson, K. A. & Prinz, F. B.
High-throughput screening of inorganic compounds for the discovery
of novel dielectric and optical materials. Sci. Data 4, 160134 (2017).



----------------------
double_perovskites_gap
----------------------
Band gap of 1306 double perovskites (a_1-b_1-a_2-b_2-O6) calculated using ﻿Gritsenko, van Leeuwen, van Lenthe and Baerends potential (gllbsc) in GPAW.

**Number of entries:** 1306

========== =================================================
Column     Description
========== =================================================
a_1        Species occupying the a1 perovskite site
a_2        Species occupying the a2 site
b_1        Species occupying the b1 site
b_2        Species occupying the b2 site
formula    Chemical formula of the entry
gap gllbsc electronic band gap (in eV) calculated via gllbsc
========== =================================================



**Reference**

Dataset discussed in:
Pilania, G. et al. Machine learning bandgaps of double perovskites. Sci. Rep. 6, 19375; doi: 10.1038/srep19375 (2016).
Dataset sourced from:
https://cmr.fysik.dtu.dk/



---------------------------
double_perovskites_gap_lumo
---------------------------
Supplementary lumo data of 55 atoms for the double_perovskites_gap dataset.

**Number of entries:** 55

====== =======================================================
Column Description
====== =======================================================
atom   Name of the atom whos lumo is listed
lumo   Lowest unoccupied molecular obital energy level (in eV)
====== =======================================================



**Reference**

Dataset discussed in:
Pilania, G. et al. Machine learning bandgaps of double perovskites. Sci. Rep. 6, 19375; doi: 10.1038/srep19375 (2016).
Dataset sourced from:
https://cmr.fysik.dtu.dk/



-------------------
elastic_tensor_2015
-------------------
1,181 structures with elastic properties calculated with DFT-PBE.

**Number of entries:** 1181

======================= ==================================================================================================================================
Column                  Description
======================= ==================================================================================================================================
G_Reuss                 Lower bound on shear modulus for polycrystalline material
G_VRH                   Average of G_Reuss and G_Voigt
G_Voigt                 Upper bound on shear modulus for polycrystalline material
K_Reuss                 Lower bound on bulk modulus for polycrystalline material
K_VRH                   Average of K_Reuss and K_Voigt
K_Voigt                 Upper bound on bulk modulus for polycrystalline material
cif                     optional: Description string for structure
compliance_tensor       Tensor describing elastic behavior
elastic_anisotropy      measure of directional dependence of the materials elasticity, metric is always >= 0
elastic_tensor          Tensor describing elastic behavior corresponding to IEEE orientation, symmetrized to crystal structure 
elastic_tensor_original Tensor describing elastic behavior, unsymmetrized, corresponding to POSCAR conventional standard cell orientation
formula                 Chemical formula of the material
kpoint_density          optional: Sampling parameter from calculation
material_id             Materials Project ID of the material
nsites                  The \# of atoms in the unit cell of the calculation.
poisson_ratio           Describes lateral response to loading
poscar                  optional: Poscar metadata
space_group             Integer specifying the crystallographic structure of the material
structure               pandas Series defining the structure of the material
volume                  Volume of the unit cell in cubic angstroms, For supercell calculations, this quantity refers to the volume of the full supercell. 
======================= ==================================================================================================================================



**Reference**

Jong, M. De, Chen, W., Angsten, T., Jain, A., Notestine, R., Gamst,
A., Sluiter, M., Ande, C. K., Zwaag, S. Van Der, Plata, J. J., Toher,
C., Curtarolo, S., Ceder, G., Persson, K. and Asta, M., "Charting
the complete elastic properties of inorganic crystalline compounds",
Scientific Data volume 2, Article number: 150009 (2015)



-----------------------
expt_formation_enthalpy
-----------------------
Experimental formation enthalpies for inorganic compounds, collected from years of calorimetric experiments. There are 1,276 entries in this dataset, mostly binary compounds. Matching mpids or oqmdids as well as the DFT-computed formation energies are also added (if any).

**Number of entries:** 1276

============== ======================================================
Column         Description
============== ======================================================
e_form expt    experimental formation enthalpy (in eV/atom)
e_form mp      formation enthalpy from Materials Project (in eV/atom)
e_form oqmd    formation enthalpy from OQMD (in eV/atom)
formula        chemical formula
mpid           materials project id
oqmdid         OQMD id
pearson symbol pearson symbol of the structure
space group    space group of the structure
============== ======================================================



**Reference**

https://www.nature.com/articles/sdata2017162



--------
expt_gap
--------
Experimental band gap of 6354 inorganic semiconductors.

**Number of entries:** 6354

======== ========================================
Column   Description
======== ========================================
formula  chemical formula
gap expt band gap (in eV) measured experimentally
======== ========================================



**Reference**

https://pubs.acs.org/doi/suppl/10.1021/acs.jpclett.8b00124



----
flla
----
3938 structures and computed formation energies from "Crystal Structure Representations for Machine Learning Models of Formation Energies."

**Number of entries:** 3938

========================= ============================================================================================================================
Column                    Description
========================= ============================================================================================================================
e_above_hull              The energy of decomposition of this material into the set of most stable materials at this chemical composition, in eV/atom.
formation_energy          Computed formation energy at 0K, 0atm using a reference state of zero for the pure elements.
formation_energy_per_atom See formation_energy
formula                   Chemical formula of the material
material_id               Materials Project ID of the material
nsites                    The \# of atoms in the unit cell of the calculation.
structure                 pandas Series defining the structure of the material
========================= ============================================================================================================================



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



------------
glass_binary
------------
Metallic glass formation data for binary alloys, collected from various experimental techniques such as melt-spinning or mechanical alloying. This dataset covers all compositions with an interval of 5 at. % in 59 binary systems, containing a total of 5959 alloys in the dataset. The target property of this dataset is the glass forming ability (GFA), i.e. whether the composition can form monolithic glass or not, which is either 1 for glass forming or 0 for non-full glass forming.

**Number of entries:** 5959

======= =================================================================================================================================================================================
Column  Description
======= =================================================================================================================================================================================
formula chemical formula
gfa     glass forming ability, correlated with the phase column, designating whether the composition can form monolithic glass or not, 1: glass forming ("AM"), 0: non-full-forming("CR")
======= =================================================================================================================================================================================



**Reference**

https://pubs.acs.org/doi/10.1021/acs.jpclett.7b01046



------------------
glass_ternary_hipt
------------------
Metallic glass formation dataset for ternary alloys, collected from the high-throughput sputtering experiments measuring whether it is possible to form a glass using sputtering. The hipt experimental data are of the Co-Fe-Zr, Co-Ti-Zr, Co-V-Zr and Fe-Ti-Nb ternary systems.

**Number of entries:** 5170

========== ==================================================================================================================
Column     Description
========== ==================================================================================================================
formula    Chemical formula of the entry
gfa        Glass forming ability: 1 means glass forming and coresponds to AM, 0 means non glass forming and corresponds to CR
phase      AM: amorphous phase or CR: crystalline phase
processing How the point was processed, always sputtering for this dataset
system     System of dataset experiment, one of: CoFeZr, CoTiZr, CoVZr, or FeTiNb
========== ==================================================================================================================



**Reference**

Accelerated discovery of metallic glasses through iteration of machine learning and high-throughput experiments
By Fang Ren, Logan Ward, Travis Williams, Kevin J. Laws, Christopher Wolverton, Jason Hattrick-Simpers, Apurva Mehta
Science Advances 13 Apr 2018 : eaaq1566



---------------------
glass_ternary_landolt
---------------------
Metallic glass formation dataset for ternary alloys, collected from the "Nonequilibrium Phase Diagrams of Ternary Amorphous Alloys,’ a volume of the Landolt– Börnstein collection. This dataset contains experimental measurements of whether it is possible to form a glass using a variety of processing techniques at thousands of compositions from hundreds of ternary systems. The processing techniques are designated in the "processing" column. There are originally 7191 experiments in this dataset, will be reduced to 6203 after deduplicated, and will be further reduced to 6118 if combining multiple data for one composition. There are originally 6780 melt-spinning experiments in this dataset, will be reduced to 5800 if deduplicated, and will be further reduced to 5736 if combining multiple experimental data for one composition.

**Number of entries:** 7191

========== ============================================================================================================================================================================
Column     Description
========== ============================================================================================================================================================================
formula    Chemical formula of the entry
gfa        Glass forming ability: 1 means glass forming and corresponds to AM, 0 means non full glass forming and corresponds to CR AC or QC
phase      "AM": amorphous phase. "CR": crystalline phase. "AC": amorphous-crystalline composite phase. "QC": quasi-crystalline phase. Phases obtained from glass producing experiments
processing processing method, meltspin or sputtering
========== ============================================================================================================================================================================



**Reference**

Y. Kawazoe, T. Masumoto, A.-P. Tsai, J.-Z. Yu, T. Aihara Jr. (1997) Y. Kawazoe, J.-Z. Yu, A.-P. Tsai, T. Masumoto (ed.) SpringerMaterials
Nonequilibrium Phase Diagrams of Ternary Amorphous Alloys · 1 Introduction Landolt-Börnstein - Group III Condensed Matter 37A (Nonequilibrium Phase Diagrams of Ternary Amorphous Alloys) https://materials.springer.com/lb/docs/sm_lbs_978-3-540-47679-5_2
10.1007/10510374_2 (Springer-Verlag Berlin Heidelberg © 1997) Accessed: 20-10-2018



----------------
heusler_magnetic
----------------
1153 Heusler alloys with DFT-calculated magnetic and electronic properties. The 1153 alloys include 576 full, 449 half and 128 inverse Heusler alloys. The data are extracted and cleaned (including de-duplicating) from Citrine.

**Number of entries:** 1153

=============== ====================================
Column          Description
=============== ====================================
e_form          Formation energy in eV/atom
formula         Chemical formula of the entry
heusler type    Full, Half, or Inverse Heusler
latt const      Lattice constant
mu_b            Magnetic moment
mu_b saturation Saturation magnetization in emu/cc
num_electron    Number of electrons per formula unit
pol fermi       Polarization at Fermi level in %
struct type     Structure type
tetragonality   Tetragonality, i.e. c/a
=============== ====================================



**Reference**

https://citrination.com/datasets/150561/



-------------
jarvis_dft_2d
-------------
Various properties of 636 2D materials computed with the OptB88vdW and TBmBJ functionals taken from the JARVIS DFT database.

**Number of entries:** 636

================= ===============================================================================
Column            Description
================= ===============================================================================
composition       A Pymatgen Composition descriptor of the composition of the material
e_form            formation energy per atom, in eV/atom
epsilon_x opt     Static dielectric function in x direction calculated with OptB88vDW functional.
epsilon_x tbmbj   Static dielectric function in x direction calculuated with TBMBJ functional.
epsilon_y opt     Static dielectric function in y direction calculated with OptB88vDW functional.
epsilon_y tbmbj   Static dielectric function in y direction calculuated with TBMBJ functional.
epsilon_z opt     Static dielectric function in z direction calculated with OptB88vDW functional.
epsilon_z tbmbj   Static dielectric function in z direction calculuated with TBMBJ functional.
exfoliation_en    Exfoliation energy (monolayer formation E) in eV
gap opt           Band gap calculated with OptB88vDW functional, in eV
gap tbmbj         Band gap calculated with TBMBJ functional, in eV
jid               JARVIS ID
mpid              Materials Project ID
structure         A description of the crystal structure of the material
structure initial Initial structure description of the crystal structure of the material
================= ===============================================================================



**Reference**

2D Dataset discussed in:
High-throughput Identification and Characterization of Two dimensional Materials using Density functional theory Kamal Choudhary, Irina Kalish, Ryan Beams & Francesca Tavazza Scientific Reports volume 7, Article number: 5179 (2017)
Original 2D Data file sourced from:
choudhary, kamal; https://orcid.org/0000-0001-9737-8074 (2018): jdft_2d-7-7-2018.json. figshare. Dataset.



-------------
jarvis_dft_3d
-------------
Various properties of 25,923 bulk materials computed with the OptB88vdW and TBmBJ functionals taken from the JARVIS DFT database.

**Number of entries:** 25923

================= ===============================================================================
Column            Description
================= ===============================================================================
bulk modulus      VRH average calculation of bulk modulus
composition       A Pymatgen Composition descriptor of the composition of the material
e_form            formation energy per atom, in eV/atom
epsilon_x opt     Static dielectric function in x direction calculated with OptB88vDW functional.
epsilon_x tbmbj   Static dielectric function in x direction calculuated with TBMBJ functional.
epsilon_y opt     Static dielectric function in y direction calculated with OptB88vDW functional.
epsilon_y tbmbj   Static dielectric function in y direction calculuated with TBMBJ functional.
epsilon_z opt     Static dielectric function in z direction calculated with OptB88vDW functional.
epsilon_z tbmbj   Static dielectric function in z direction calculuated with TBMBJ functional.
gap opt           Band gap calculated with OptB88vDW functional, in eV
gap tbmbj         Band gap calculated with TBMBJ functional, in eV
jid               JARVIS ID
mpid              Materials Project ID
shear modulus     VRH average calculation of shear modulus
structure         A description of the crystal structure of the material
structure initial Initial structure description of the crystal structure of the material
================= ===============================================================================



**Reference**

3D Dataset discussed in:
Elastic properties of bulk and low-dimensional materials using van der Waals density functional Kamal Choudhary, Gowoon Cheon, Evan Reed, and Francesca Tavazza Phys. Rev. B 98, 014107
Original 3D Data file sourced from:
choudhary, kamal; https://orcid.org/0000-0001-9737-8074 (2018): jdft_3d.json. figshare. Dataset.



----------------------
jarvis_ml_dft_training
----------------------
Various properties of 24,759 bulk and 2D materials computed with the OptB88vdW and TBmBJ functionals taken from the JARVIS DFT database.

**Number of entries:** 24759

=============== ===============================================================================
Column          Description
=============== ===============================================================================
bulk modulus    VRH average calculation of bulk modulus
composition     A descriptor of the composition of the material
e mass_x        Effective electron mass in x direction (BoltzTraP)
e mass_y        Effective electron mass in y direction (BoltzTraP)
e mass_z        Effective electron mass in z direction (BoltzTraP)
e_exfol         exfoliation energy per atom in eV/atom
e_form          formation energy per atom, in eV/atom
epsilon_x opt   Static dielectric function in x direction calculated with OptB88vDW functional.
epsilon_x tbmbj Static dielectric function in x direction calculated with TBMBJ functional.
epsilon_y opt   Static dielectric function in y direction calculated with OptB88vDW functional.
epsilon_y tbmbj Static dielectric function in y direction calculated with TBMBJ functional.
epsilon_z opt   Static dielectric function in z direction calculated with OptB88vDW functional.
epsilon_z tbmbj Static dielectric function in z direction calculated with TBMBJ functional.
gap opt         Band gap calculated with OptB88vDW functional, in eV
gap tbmbj       Band gap calculated with TBMBJ functional, in eV
hole mass_x     Effective hole mass in x direction (BoltzTraP)
hole mass_y     Effective hole mass in y direction (BoltzTraP)
hole mass_z     Effective hole mass in z direction (BoltzTraP)
jid             JARVIS ID
mpid            Materials Project ID
mu_b            Magnetic moment, in Bohr Magneton
shear modulus   VRH average calculation of shear modulus
structure       A Pymatgen Structure object describing the crystal structure of the material
=============== ===============================================================================



**Reference**

Dataset discussed in:
Machine learning with force-field-inspired descriptors for materials: Fast screening and mapping energy landscape Kamal Choudhary, Brian DeCost, and Francesca Tavazza Phys. Rev. Materials 2, 083801

Original Data file sourced from:
choudhary, kamal (2018): JARVIS-ML-CFID-descriptors and material properties. figshare. Dataset.



----
m2ax
----
Elastic properties of 223 stable M2AX compounds from "A comprehensive survey of M2AX phase elastic properties" by Cover et al. Calculations are PAW PW91.

**Number of entries:** 223

=============== ==================================================================================
Column          Description
=============== ==================================================================================
a               Lattice parameter a, in A (angstrom)
bulk modulus    In GPa
c               lattice parameter c, in A (angstrom)
c11             Elastic constants of the M2AX material. These are specific to hexagonal materials.
c12             Elastic constants of the M2AX material. These are specific to hexagonal materials.
c13             Elastic constants of the M2AX material. These are specific to hexagonal materials.
c33             Elastic constants of the M2AX material. These are specific to hexagonal materials.
c44             Elastic constants of the M2AX material. These are specific to hexagonal materials.
d_ma            distance from the M atom to the A atom
d_mx            distance from the M atom to the X atom
elastic modulus In GPa
formula         chemical formula
shear modulus   In GPa
=============== ==================================================================================



**Reference**

http://iopscience.iop.org/article/10.1088/0953-8984/21/30/305403/meta



------
mp_all
------
A complete copy of the Materials Project database as of 10/18/2018. Mp_all files contain structure data for each material while mp_nostruct does not.

**Number of entries:** 83989

================== =========================================================================================
Column             Description
================== =========================================================================================
bulk modulus       in GPa, average of Voight, Reuss, and Hill
e_form             Formation energy per atom (eV)
e_hull             The calculated energy above the convex hull, in eV per atom
elastic anisotropy The ratio of elastic anisotropy.
formula            The chemical formula of the MP entry
gap pbe            The band gap in eV calculated with PBE-DFT functional
initial structure  A Pymatgen Structure object describing the material crystal structure prior to relaxation
mpid               (input): The Materials Project mpid, as a string.
mu_b               The total magnetization of the unit cell.
shear modulus      in GPa, average of Voight, Reuss, and Hill
structure          A Pymatgen Structure object describing the material crystal structure
================== =========================================================================================



**Reference**

A. Jain*, S.P. Ong*, G. Hautier, W. Chen, W.D. Richards, S. Dacek, S. Cholia, D. Gunter, D. Skinner, G. Ceder, K.A. Persson (*=equal contributions)
The Materials Project: A materials genome approach to accelerating materials innovation
APL Materials, 2013, 1(1), 011002.
doi:10.1063/1.4812323



-----------
mp_nostruct
-----------
A complete copy of the Materials Project database as of 10/18/2018. Mp_all files contain structure data for each material while mp_nostruct does not.

**Number of entries:** 83989

================== ===========================================================
Column             Description
================== ===========================================================
bulk modulus       in GPa, average of Voight, Reuss, and Hill
e_form             Formation energy per atom (eV)
e_hull             The calculated energy above the convex hull, in eV per atom
elastic anisotropy The ratio of elastic anisotropy.
formula            The chemical formula of the MP entry
gap pbe            The band gap in eV calculated with PBE-DFT functional
mpid               (input): The Materials Project mpid, as a string.
mu_b               The total magnetization of the unit cell.
shear modulus      in GPa, average of Voight, Reuss, and Hill
================== ===========================================================



**Reference**

A. Jain*, S.P. Ong*, G. Hautier, W. Chen, W.D. Richards, S. Dacek, S. Cholia, D. Gunter, D. Skinner, G. Ceder, K.A. Persson (*=equal contributions)
The Materials Project: A materials genome approach to accelerating materials innovation
APL Materials, 2013, 1(1), 011002.
doi:10.1063/1.4812323



--------------------
phonon_dielectric_mp
--------------------
Phonon (lattice/atoms vibrations) and dielectric properties of 1296 compounds computed via ABINIT software package in the harmonic approximation based on density functional perturbation theory.

**Number of entries:** 1296

=============== ======================================================================================================================================================================================================
Column          Description
=============== ======================================================================================================================================================================================================
eps_electronic  A target variable of the dataset, electronic contribution to the calculated dielectric constant; unitless.
eps_total       A target variable of the dataset, total calculated dielectric constant. Unitless: it is a ratio over the dielectric constant at vacuum.
formula         The chemical formula of the material
last phdos peak A target variable of the dataset, the frequency of the last calculated phonon density of states in 1/cm; may be used as an estimation of dominant longitudinal optical phonon frequency, a descriptor.
mpid            The Materials Project identifier for the material
structure       A pymatgen Structure object describing the chemical strucutre of the material
=============== ======================================================================================================================================================================================================



**Reference**

Petretto, G. et al. High-throughput density functional perturbation theory phonons for inorganic materials. Sci. Data 5:180065 doi: 10.1038/sdata.2018.65 (2018).
Petretto, G. et al. High-throughput density functional perturbation theory phonons for inorganic materials. (2018). figshare. Collection.



--------------------
piezoelectric_tensor
--------------------
941 structures with piezoelectric properties, calculated with DFT-PBE.

**Number of entries:** 941

==================== ==================================================================================================================================
Column               Description
==================== ==================================================================================================================================
cif                  optional: Description string for structure
eij_max              Piezoelectric modulus
formula              Chemical formula of the material
material_id          Materials Project ID of the material
meta                 optional, metadata descriptor of the datapoint
nsites               The \# of atoms in the unit cell of the calculation.
piezoelectric_tensor Tensor describing the piezoelectric properties of the material
point_group          Descriptor of crystallographic structure of the material
poscar               optional: Poscar metadata
space_group          Integer specifying the crystallographic structure of the material
structure            pandas Series defining the structure of the material
v_max                Crystallographic direction
volume               Volume of the unit cell in cubic angstroms, For supercell calculations, this quantity refers to the volume of the full supercell. 
==================== ==================================================================================================================================



**Reference**

de Jong, M., Chen, W., Geerlings, H., Asta, M. & Persson, K. A.
A database to enable discovery and design of piezoelectric materials.
Sci. Data 2, 150053 (2015)



--------------
steel_strength
--------------
312 steels with experimental yield strength and ultimate tensile strength, extracted and cleaned (including de-duplicating) from Citrine.

**Number of entries:** 312

================ ================================
Column           Description
================ ================================
al               weight percent of Al
c                weight percent of C
co               weight percent of Co
cr               weight percent of Cr
elongation       elongation in %
formula          Chemical formula of the entry
mn               weight percent of Mn
mo               weight percent of Mo
n                weight percent of N
nb               weight percent of Nb
ni               weight percent of Ni
si               weight percent of Si
tensile strength ultimate tensile strength in GPa
ti               weight percent of Ti
v                weight percent of V
w                weight percent of W
yield strength   yield strength in GPa
================ ================================



**Reference**

https://citrination.com/datasets/153092/



----------------
wolverton_oxides
----------------
4,914 perovskite oxides containing composition data, lattice constants, and formation + vacancy formation energies. All perovskites are of the form ABO3. Adapted from a dataset presented by Emery and Wolverton.

**Number of entries:** 4914

================= =======================================================================================
Column            Description
================= =======================================================================================
a                 Lattice parameter a, in A (angstrom)
alpha             Lattice angle alpha, in degrees
atom a            The atom in the 'A' site of the pervoskite.
atom b            The atom in the 'B' site of the perovskite.
b                 Lattice parameter b, in A (angstrom)
beta              Lattice angle beta, in degrees
c                 Lattice parameter c, in A (angstrom)
e_form            Formation energy in eV
e_form oxygen     Formation energy of oxygen vacancy (eV)
e_hull            Energy above convex hull, wrt. OQMD db (eV)
formula           Chemical formula of the entry
gamma             Lattice angle gamma, in degrees
gap pbe           Bandgap in eV from PBE calculations
lowest distortion Local distortion crystal structure with lowest energy among all considered distortions.
mu_b              Magnetic moment
vpa               Volume per atom (A^3/atom)
================= =======================================================================================



**Reference**

Emery, A. A. & Wolverton, C. High-throughput DFT calculations of formation energy, stability and oxygen vacancy formation energy of ABO3 perovskites. Sci. Data 4:170153 doi: 10.1038/sdata.2017.153 (2017).
Emery, A. A., & Wolverton, C. Figshare http://dx.doi.org/10.6084/m9.figshare.5334142 (2017)



