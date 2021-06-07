"""
Site featurizers requiring external libraries for core functionality.
"""
from monty.dev import requires
from pymatgen.io.ase import AseAtomsAdaptor
from sklearn.exceptions import NotFittedError
from pymatgen.core import Structure

from matminer.featurizers.base import BaseFeaturizer

# SOAPFeaturizer
try:
    import dscribe
    from dscribe.descriptors import SOAP as SOAP_dscribe
except ImportError:
    dscribe, SOAP_dscribe = None, None


class SOAP(BaseFeaturizer):
    """
    Smooth overlap of atomic positions (interface via DScribe).

    Class for generating a partial power spectrum from Smooth Overlap of Atomic
    Orbitals (SOAP). This implementation uses real (tesseral) spherical
    harmonics as the angular basis set and provides two orthonormalized
    alternatives for the radial basis functions: spherical primitive gaussian
    type orbitals ("gto") or the polynomial basis set ("polynomial"). By
    default the faster gto-basis is used. Please see the DScribe SOAP
    documentation for more details.

    Note that SOAP is only featurized for elements identified by "fit" (see
    following), thus "fit" must be called before "featurize", or else an error
    will be raised.

    Based originally on the following publications:

    "On representing chemical environments, Albert P. Bartók, Risi
        Kondor, and Gábor Csányi, Phys. Rev. B 87, 184115, (2013),
        https://doi.org/10.1103/PhysRevB.87.184115

    "Comparing molecules and solids across structural and alchemical
        space", Sandip De, Albert P. Bartók, Gábor Csányi and Michele Ceriotti,
        Phys.  Chem. Chem. Phys. 18, 13754 (2016),
        https://doi.org/10.1039/c6cp00415f

    Implementation (and some documentation) originally based on DScribe:
    https://github.com/SINGROUP/dscribe.

    "DScribe: Library of descriptors for machine learning in materials science",
        Himanen, L., J{\"a}ger, M. O.J., Morooka, E. V., Federici
        Canova, F., Ranawat, Y. S., Gao, D. Z., Rinke, P. and Foster, A. S.
        Computer Physics Communications, 106949 (2019),
        https://doi.org/10.1016/j.cpc.2019.106949

    Args:
        rcut (float): A cutoff for local region in angstroms. Should be
            bigger than 1 angstrom.
        nmax (int): The number of radial basis functions.
        lmax (int): The maximum degree of spherical harmonics.
        sigma (float): The standard deviation of the gaussians used to expand the
            atomic density.
        rbf (str): The radial basis functions to use. The available options are:

            * "gto": Spherical gaussian type orbitals defined as :math:`g_{nl}(r) = \sum_{n'=1}^{n_\mathrm{max}}\,\\beta_{nn'l} r^l e^{-\\alpha_{n'l}r^2}`
            * "polynomial": Polynomial basis defined as :math:`g_{n}(r) = \sum_{n'=1}^{n_\mathrm{max}}\,\\beta_{nn'} (r-r_\mathrm{cut})^{n'+2}`

        periodic (bool): Determines whether the system is considered to be
            periodic.
        crossover (bool): Determines if crossover of atomic types should
            be included in the power spectrum. If enabled, the power
            spectrum is calculated over all unique species combinations Z
            and Z'. If disabled, the power spectrum does not contain
            cross-species information and is only run over each unique
            species Z. Turned on by default to correspond to the original
            definition
    """

    @requires(
        dscribe,
        "SOAPFeaturizer requires DScribe. Install from github.com/SINGROUP/dscribe",
    )
    def __init__(
        self,
        rcut,
        nmax,
        lmax,
        sigma,
        periodic,
        rbf="gto",
        crossover=True,
    ):
        self.rcut = rcut
        self.nmax = nmax
        self.lmax = lmax
        self.sigma = sigma
        self.rbf = rbf
        self.periodic = periodic
        self.crossover = crossover
        self.adaptor = AseAtomsAdaptor()
        self.length = None
        self.atomic_numbers = None
        self.soap = None
        self.n_elements = None

    @classmethod
    def from_preset(cls, preset):
        """
        Create a SOAP featurizer object from sensible or published presets.
        Args:
            preset (str): Choose from:
                "formation energy": Preset used for formation energy prediction
                    in the original Dscribe paper.
        Returns:
        """
        valid_presets = ["formation_energy"]
        if preset == "formation_energy":
            return cls(6, 8, 8, 0.4, True, "gto", True)
        else:
            raise ValueError(f"'{preset}' is not a valid preset. Choose from {valid_presets}")

    def _check_fitted(self):
        if not self.soap:
            raise NotFittedError("Please fit SOAP before featurizing.")

    def fit(self, X, y=None):
        """
        Fit the SOAP featurizer to a dataframe.

        Args:
            X ([SiteCollection]): For example, a list of pymatgen Structures.
            y : unused (added for consistency with overridden method signature)

        Returns:
            self
        """
        # Check that pymatgen.Structures are provided
        if not all([isinstance(struct, Structure) for struct in X]):
            raise TypeError("This fit requires an array-like input of Pymatgen " "Structures and sites!")

        elements = set()
        for s in X:
            c = s.composition.elements
            for e in c:
                if e.Z not in elements:
                    elements.add(e.Z)
        self.elements_sorted = sorted(list(elements))

        self.atomic_numbers = elements
        self.soap = SOAP_dscribe(
            species=self.atomic_numbers,
            rcut=self.rcut,
            nmax=self.nmax,
            lmax=self.lmax,
            sigma=self.sigma,
            rbf=self.rbf,
            periodic=self.periodic,
            crossover=self.crossover,
            average="off",
            sparse=False,
        )

        self.length = self.soap.get_number_of_features()
        return self

    def featurize(self, struct, idx):
        self._check_fitted()
        s_ase = self.adaptor.get_atoms(struct)
        return self.soap.create(s_ase, positions=[idx], n_jobs=self.n_jobs).tolist()[0]

    def feature_labels(self):
        self._check_fitted()
        return [f"SOAP_{i}" for i in range(self.length)]

    def citations(self):
        return [
            "@article{PhysRevB.87.184115,"
            "title = {On representing chemical environments},"
            "author = {Bart'ok, Albert P. and Kondor, Risi and Cs'anyi, "
            "G'abor},"
            "journal = {Phys. Rev. B},"
            "volume = {87},"
            "issue = {18},"
            "pages = {184115},"
            "numpages = {16},"
            "year = {2013},"
            "month = {May},"
            "publisher = {American Physical Society},"
            "doi = {10.1103/PhysRevB.87.184115},"
            "url = {https://link.aps.org/doi/10.1103/PhysRevB.87.184115}}",
            "@Article{C6CP00415F,"
            "author ={De, Sandip and BartÃ³k, Albert P. and CsÃ¡nyi, GÃ¡bor"
            " and Ceriotti, Michele},"
            "title  ={Comparing molecules and solids across structural and "
            "alchemical space},"
            "journal = {Phys. Chem. Chem. Phys.},"
            "year = {2016},"
            "volume = {18},"
            "issue = {20},"
            "pages = {13754-13769},"
            "publisher = {The Royal Society of Chemistry},"
            "doi = {10.1039/C6CP00415F},"
            "url = {http://dx.doi.org/10.1039/C6CP00415F},}",
            "@article{dscribe, "
            'author = {Himanen, Lauri and J{"a}ger, Marc O.~J. and '
            "Morooka, Eiaki V. and Federici Canova, Filippo and Ranawat, "
            "Yashasvi S. and Gao, David Z. and Rinke, Patrick and Foster, "
            "Adam S.}, "
            "title = {{DScribe: Library of descriptors for machine "
            "learning in materials science}}, "
            "journal = {Computer Physics Communications}, "
            "year = {2019}, pages = {106949}, "
            "doi = {https://doi.org/10.1016/j.cpc.2019.106949}}",
        ]

    def implementors(self):
        return ["Lauri Himanen and the DScribe team", "Alex Dunn"]
