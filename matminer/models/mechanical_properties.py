__author__ = 'Saurabh Bajaj <sbajaj@lbl.gov>'


def vickers_hardness(G, K):
    """
    Calculate Vickers hardness for a material.

    Args:
        G: (float) elastic bulk moduli
        K: (float) elastic shear moduli

    Returns: (float) Vickers hardness

    """
    return 2*(G**3.0/K**2.0)**0.585 - 3

if __name__ == "__main__":
    print vickers_hardness(3, 2)