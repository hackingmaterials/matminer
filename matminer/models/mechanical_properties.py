import math

__author__ = 'Saurabh Bajaj <sbajaj@lbl.gov>'


def vickers_hardness1(G, K):
    """
    Calculate Vickers hardness (H_v) for a material.

    References:
        DOI: (not available yet) (Title: "A Statistical Learning Framework for Materials Science: Application to
        Elastic Moduli of k-nary Inorganic Compounds")

    Args:
        G: (float) elastic bulk moduli
        K: (float) elastic shear moduli

    Returns: (float) Vickers hardness (H_v) (N/m^2)

    """
    return 2*(G**3.0/K**2.0)**0.585 - 3


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


def critical_stress(E, gamma_s, a, gamma_p=0):
    """
    Calculate critical stress needed for crack propagation (sigma_c) according to Griffith theory of brittle fracture.

    Args:
        E: (float) Young's modulus (N/m^2)
        gamma_s: (float) elastic strain energy released (N/m)
        a: (float) one half crack length for internal cracks or crack length for edge cracks (m)
        gamma_p: (float) plastic strain energy released (N/m)

    Returns: critical stress needed for crack propagation (sigma_c) (N/m^2)

    """
    return ((2 * E * (gamma_s+gamma_p))/(math.pi * a)) ** 0.5

if __name__ == "__main__":
    print vickers_hardness1(3, 2)
