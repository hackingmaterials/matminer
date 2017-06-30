import math
import sympy as sp

__author__ = 'Saurabh Bajaj <sbajaj@lbl.gov>'


class YM:
    """
    Calculate Young's modulus (E) from other moduli.
    """
    def __init__(self, nu, G=None, K=None):
        """
        Args:
            nu: (float) Poisson's ratio
            G: (float) shear or rigidity modulus (N/m^2)
            K: (float) bulk modulus (N/m^2)
        """
        self.nu = nu
        self.G = G
        self.K = K

    def equation(self):
        return 'E - 2 * G * (1+nu)'

    def calculate(self):
        """
        Returns: Young's modulus (E) (N/m^2)
        """
        if self.G and self.K is None:
            return 2 * self.G * (1+self.nu)
        elif self.G is None and self.K:
            return 3 * self.K * (1-2*self.nu)
        else:
            raise ValueError("Enter one of G or K")


class YoungsModulus:
    def __init__(self):
        pass

    def properties_involved(self):
        E, G, nu = sp.symbols('E G nu')
        return E, G, nu

    def equation(self):
        E, G, nu = sp.symbols('E G nu')
        # return 'E - (2 * G * (1+nu))'
        return E - (2 * G * (1+nu))

    def calculate(self, nu, G):
        return 2 * G * (1+nu)


class ShearModulus:
    def __init__(self):
        pass

    def properties_involved(self):
        G, E, nu = sp.symbols('G E nu')
        return G, E, nu

    def equation(self):
        G, E, nu = sp.symbols('G E nu')
        return G - (E/(2 * (1+nu)))

    def calculate(self, E, nu):
        return E/(2 * (1+nu))


def shear_modulus(nu, E):
    """
    Calculate shear/rigidity modulus (G) from Young's modulus.

    Args:
        nu: (float) Poisson's ratio
        E: (float) Young's modulus (N/m^2)

    Returns: shear/rigidity modulus (G) (N/m^2)

    """
    return E/(2 * (1+nu))


def bulk_modulus(nu, E):
    """
    Calculate bulk modulus (K) from Young's modulus.

    Args:
        nu: (float) Poisson's ratio
        E: (float) Young's modulus (N/m^2)

    Returns: bulk modulus (K) (N/m^2)

    """
    return E/(1 - 2*nu)


class BulkModulus:
    def __init__(self):
        pass

    def properties_involved(self):
        K, E, nu = sp.symbols('K E nu')
        return K, E, nu

    def equation(self):
        K, E, nu = sp.symbols('K E nu')
        return K - (E/(1 - 2*nu))

    def calculate(self, E, nu):
        return E/(1 - 2*nu)


def bulkmodulus_coordination(N_c, ionicity, d):
    """
    Calculate bulk modulus (K) from coordination number.

    References:
        - ISBN: 9780521523394 (Title: "Atomic and Electronic Structure of Solids")
        - DOI: 10.1103/PhysRevB.32.7988 (Title: "Calculation of bulk moduli of diamond and zinc-blende solids")

    Args:
        N_c: (float) average coordination number of the solid
        ionicity: (int) a dimensionless number which describes the ionicity a dimensionless number which describes the
         ionicity; ionicity=0 for the homopolar crystals C, Si, Ge, Sn; ionicity=1 for III-V compounds (where the
         valence of each element differs by 1 from the average valence of 4); ionicity=2 for II-VI compounds (where the
         valence of each element differs by 2 from the average valence of 4).
        d: (float) bond length

    Returns: bulk modulus (K) (N/m^2)

    """
    return (N_c/4) * (1971-220*ionicity) * d**(-3.5)


def vickers_hardness1(G, K):
    """
    Calculate Vickers hardness (H_v) for a material.

    References:
        DOI: (not available yet) (Title: "A Statistical Learning Framework for Materials Science: Application to
        Elastic Moduli of k-nary Inorganic Compounds")

    Args:
        G: (float) elastic bulk modulus (N/m^2)
        K: (float) elastic shear modulus (N/m^2)

    Returns: (float) Vickers hardness (H_v) (N/m^2)

    """
    return 2*(G**3.0/K**2.0)**0.585 - 3


class Lames1stParam:
    """
    In continuum mechanics, the Lames parameters (also called the Lame coefficients or Lame constants) are two
    material-dependent quantities denoted by 'lamda' and 'mu' that arise in strain-stress relationships. In general,
    'lamda' and 'mu' are individually referred to as Lames first parameter and Lames second parameter, respectively.

    Reference:
        https://en.wikipedia.org/wiki/Lam%C3%A9_parameters
    """
    def __init__(self):
        pass

    def properties_involved(self):
        lamda, K, nu = sp.symbols('lamda K nu')
        return lamda, K, nu

    def equation(self):
        lamda, K, nu = sp.symbols('lamda K nu')
        return lamda - ((3 * K * nu)/(1 + nu))

    def calculate(self, K, nu):
        return (3 * K * nu)/(1 + nu)


class PWaveModulus:
    """
    In linear elasticity, the P-wave modulus 'M', also known as the longitudinal modulus or the constrained modulus, is
    one of the elastic moduli available to describe isotropic homogeneous materials.

    Reference:
        https://en.wikipedia.org/wiki/P-wave_modulus
    """
    def __init__(self):
        pass

    def properties_involved(self):
        M, K, nu = sp.symbols('M K nu')
        return M, K, nu

    def equation(self):
        M, K, nu = sp.symbols('M K nu')
        return M - ((3 * K * (1-nu))/(1 + nu))

    def calculate(self, K, nu):
        return (3 * K * (1-nu))/(1 + nu)


def vickers_hardness2(F, a):
    """
    Calculate Vickers hardness (H) for a material.

    References:
        DOI: 10.1007/s10853-013-7569-1 (Title: "Room temperature mechanical properties of natural-mineral-based
        thermoelectrics")

    Args:
        F: (float) indentation load (N)
        a: (float) half of diagonal length of the indentation impression (m)

    Returns: (float) Vickers hardness (H) (N/m^2)

    """
    return (1.8544 * F)/((2*a)**2.0)


def thermal_stress(E_T, cte_T, delta_T, nu_T):
    """
    Calculate thermal stress (sigma_T) in a material.

    References:
        DOI: 10.1007/s10853-013-7569-1 (Title: "Room temperature mechanical properties of natural-mineral-based
        thermoelectrics")

    Args:
        E_T: (float) temperature-dependent Young's modulus (N/m^2)
        cte_T: (float) temperature-dependent thermal expansion coefficient (K^(-1))
        delta_T: (float) temperature difference (K)
        nu_T: (float) temperature-dependent Poisson's ratio

    Returns: (float) thermal stress (sigma_T) induced by thermal gradients or thermal transients

    """
    return (E_T * cte_T * delta_T)/(1 - nu_T)


def fracture_toughness(E, H, F, c):
    """
    Calculate fracture toughness (K_c) of a material.

    References:
        DOI: 10.1007/s10853-013-7569-1 (Title: "Room temperature mechanical properties of natural-mineral-based
        thermoelectrics")

    Args:
        E: (float) Young's modulus (N/m^2)
        H: (float) hardness (N/m^2)
        F: (float) indentation load (N)
        c: (float) hald of radial crack length (m)

    Returns: (float) fracture toughness (K_c) of a material (Pa.m^(0.5) or N/m^(1.5))

    """
    return (0.016 * (E/H)**0.5 * F)/(c**1.5)


def brittleness_index(H, E, K_c):
    """
    Calculate brittleness index (BI) of a material.

    Args:
        H: (float) hardness (N/m^2)
        E: (float) Young's modulus (N/m^2)
        K_c: (float) fracture toughness  (N/m^(1.5))

    Returns: (float) brittleness index (BI) of a material (m^(-1))

    """
    return (H * E)/(K_c**2)


def steadystate_heatflow(A, T2, T1, kappa, x):
    """
    Calculate steady state heat flow (Q)

    Args:
        A: (float) cross-sectional area (m^2)
        T2: (float) temperature at cold end (K)
        T1: (float) temperature at hot end (K)
        kappa: (float) thermal conductivity (W(mK)^(-1))
        x: (float) slab length at which heat flow is calculated (m)

    Returns: steady state heat flow (Q) (W or J/s)

    """
    return (-A * (T2-T1) * kappa)/x


def max_allowed_heatflow(nu, kappa, sigma, E, cte):
    """
    Calculate maximum allowable heat flow (QF)

    Args:
        nu: (float) Poisson's ratio
        kappa: (float) thermal conductivity (W(mK)^(-1))
        sigma: (float) stress as a result of temperature gradient (N/m^2)
        E: (float) Young's modulus (N/m^2)
        cte: (float) coefficient of thermal expansion (K^(-1))

    Returns: maximum allowable heat flow (QF) (W/m)

    """
    return ((1-nu) * kappa * sigma)/(E * cte)


def stress_from_tempgradient(T2, T1, E, cte, nu):
    """
    Calculate stress as a result of temperature gradient (sigma)

    Args:
        T2: (float) temperature at cold end (K)
        T1: (float) temperature at hot end (K)
        E: (float) Young's modulus (N/m^2)
        cte: (float) coefficient of thermal expansion (K^(-1))
        nu: (float) Poisson's ratio

    Returns: stress as a result of temperature gradient (sigma) (N/m^2)

    """
    return ((T2-T1) * E * cte)/(1-nu)


def thermal_shock(sigma_tens, nu, kappa, E, cte):
    """
    Calculate thermal shock resistance parameter (R_therm)

    Args:
        sigma_tens: (float) tensile stress or strength (N/m^2)
        nu: (float) Poisson's ratio
        kappa: (float) thermal conductivity (W(mK)^(-1))
        E: (float) Young's modulus (N/m^2)
        cte: (float) coefficient of thermal expansion (K^(-1))

    Returns: thermal shock resistance parameter (R_therm)

    """
    return (sigma_tens * (1-nu) * kappa)/(E * cte)


def critical_stress(E, gamma_s, a, gamma_p=0, nu=0):
    """
    Calculate critical stress needed for crack propagation (sigma_c) according to Griffith theory of brittle fracture.

    References:
        - http://www4.ncsu.edu/~murty/NE509/NOTES/Ch4b-Fracture.pdf
        - http://www.srmuniv.ac.in/sites/default/files/downloads/griffith_theory_of_brittle_fracture.pdf
        - https://en.wikipedia.org/wiki/Fracture_mechanics
        - https://www.fose1.plymouth.ac.uk/fatiguefracture/tutorials/FractureMechanics/Griffith/GriffTheory1.htm

    Args:
        E: (float) Young's modulus (N/m^2)
        gamma_s: (float) elastic strain energy released (N/m)
        a: (float) one half crack length for internal cracks or crack length for edge cracks (m)
        gamma_p: (float) plastic strain energy released (N/m)
        nu: (float) Poisson's ratio, used in plain strain condition, else default=0

    Returns: critical stress needed for crack propagation (sigma_c) (N/m^2)

    """
    return ((2 * E * (gamma_s+gamma_p))/((1-nu**2) * math.pi * a)) ** 0.5


def critical_fracture_toughness(sigma, a, Y=1):
    """
    Calculate critical fracture toughness (K_IC) according to Griffith theory of brittle fracture. Also known as stress
    intensity factor, and called fracture toughness nder conditions of:
        (i) brittle fracture
        (ii) in the presence of a sharp crack
        (iii) under critical tensile loading

    References:
        - http://www4.ncsu.edu/~murty/NE509/NOTES/Ch4b-Fracture.pdf
        - http://www.srmuniv.ac.in/sites/default/files/downloads/griffith_theory_of_brittle_fracture.pdf
        - https://en.wikipedia.org/wiki/Fracture_mechanics
        - https://www.fose1.plymouth.ac.uk/fatiguefracture/tutorials/FractureMechanics/Griffith/GriffTheory1.htm

    Args:
        sigma: (float) tensile stress or strength (N/m^2)
        a: (float) one half crack length for internal cracks or crack length for edge cracks (m)
        Y: (float) Crack shape factor

    Returns: critical fracture toughness (K_IC) (Pa.m^(0.5) or N/m^(1.5))

    """
    return Y * sigma * (math.pi * a)**0.5


def strain_energy_releaserate(K_I, E, nu=0):
    """
    Calculate strain energy release rate (G_I). Irwin was the first to observe that if the size of the plastic zone
    around a crack is small compared to the size of the crack, the energy required to grow the crack will not be
    critically dependent on the state of stress at the crack tip. Irwin showed that for a mode I crack (opening mode)
    the strain energy release rate and the stress intensity factor are related by the following relation.

    Args:
        K_I: (float) stress intensity factor in mode 1 (N/m^(1.5))
        E: (float) Young's modulus (N/m^2)
        nu: (float) Poisson's ratio, used in plain strain condition, else default=0

    Returns: strain energy release rate (G_I) (N/m)

    """
    return ((1-nu**2) * K_I**2)/E


if __name__ == "__main__":
    print vickers_hardness1(3, 2)
    print YoungsModulus(nu=1).properties_involved()