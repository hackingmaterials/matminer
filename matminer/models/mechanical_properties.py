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
        E_T: (float) temperature-dependent Young's modulus
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
        E: (float) Young's modulus
        H: (float) hardness
        F: (float) indentation load (N)
        c: (float) hald of radial crack length (m)

    Returns: (float) fracture toughness (K_c) of a material (Pa.m^(0.5) or N/m^(3/2))

    """
    return (0.016 * (E/H)**0.5 * F)/(c**1.5)


def brittleness_index(H, E, K_c):
    """
    Calculate brittleness index (BI) of a material.
    Args:
        H: (float) hardness
        E: (float) Young's modulus
        K_c: (float) fracture toughness

    Returns: (float) brittleness index (BI) of a material

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


def stress_from_tempgradient(T2, T1, E, cte, nu):
    """
    Calculate stress as a result of temperature gradient (sigma)

    Args:
        T2: (float) temperature at cold end (K)
        T1: (float) temperature at hot end (K)
        E: (float) Young's modulus
        cte: coefficient of thermal expansion (K^(-1))
        nu: Poisson's ratio

    Returns: stress as a result of temperature gradient (sigma) (N/m^2)

    """
    return ((T2-T1) * E * cte)/(1-nu)


if __name__ == "__main__":
    print vickers_hardness1(3, 2)
