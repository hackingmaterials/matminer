__author__ = 'Saurabh Bajaj <sbajaj@lbl.gov>'


def vickers_hardness1(G, K):
    """
    Calculate Vickers hardness for a material.

    References:
        DOI: (not available yet) (Title: "A Statistical Learning Framework for Materials Science: Application to
        Elastic Moduli of k-nary Inorganic Compounds")

    Args:
        G: (float) elastic bulk moduli
        K: (float) elastic shear moduli

    Returns: (float) Vickers hardness

    """
    return 2*(G**3.0/K**2.0)**0.585 - 3


def thermal_stress(E_T, alpha_T, delta_T, nu_T):
    """
    Calculate thermal stress.

    References:
        DOI: 10.1007/s10853-013-7569-1 (Title: "Room temperature mechanical properties of natural-mineral-based
        thermoelectrics")

    Args:
        E_T: (float) temperature-dependent Young's modulus
        alpha_T: (float) temperature-dependent thermal expansion coefficient
        delta_T: (float) temperature difference
        nu_T: (float) temperature-dependent Poisson's ratio

    Returns: (float) thermal stress induced by thermal gradients or thermal transients

    """
    return (E_T * alpha_T * delta_T)/(1 - nu_T)

if __name__ == "__main__":
    print vickers_hardness1(3, 2)