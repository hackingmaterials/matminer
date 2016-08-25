import math
from scipy.constants import *

__author__ = 'Saurabh Bajaj <sbajaj@lbl.gov>'


class ThermalConductivity:
    def __init__(self, vol):
        """
        Args:
            vol: (float) volume (in SI units, i.e. m^(-3))

        Returns: None

        """
        self.volume = vol

    def cahill_model(self, n, v_l, v_t1, v_t2):
        """
        Calculate Cahill thermal conductivity.

        References:
        # http://www.sciencedirect.com/science/article/pii/S0925838814021562
        # http://www.sciencedirect.com/science/article/pii/S0927025615004395
        # http://onlinelibrary.wiley.com/doi/10.1002/adma.201400515/epdf
        # http://www.nature.com/nature/journal/v508/n7496/pdf/nature13184.pdf (full formula)

        Args:
            n: (int) number of atoms
            v_l: (float) longitudinal sound velocity (in SI units, i.e. m(s)^(-1))
            v_t1: (float) transverse sound velocity in direction 1 (in SI units, i.e. m(s)^(-1))
            v_t2: (float) transverse sound velocity in direction 2 (in SI units, i.e. m(s)^(-1))

        Returns: (float) Cahill thermal conductivity (in SI units, i.e. W(mK)^(-1))
        """
        return (1.0/2) * ((math.pi/6)**(1.0/3)) * k * ((n/self.volume)**(2.0/3)) * (v_l + v_t1 + v_t2)

    def clarke_model(self, M, E, m):
        """
        Calculate Clarke thermal conductivity.

        References:
        # http://www.sciencedirect.com/science/article/pii/S0925838814021562
        # http://www.sciencedirect.com/science/article/pii/S0927025615004395

        Args:
            M: (float) molecular mass
            E: (float) Young's modules (in SI units, i.e. Kgm(s)^(-2)
            m: (float) total mass (in SI units, i.e. Kg)

        Returns: (float) Clarke thermal conductivity (in SI units, i.e. W(mK)^(-1))
        """
        return 0.87 * k * ((1/M)**(2.0/3)) * (E**(1.0/2)) * ((m/self.volume)**(1.0/6))

if __name__ == "__main__":
    print ThermalConductivity(1).cahill_model(1, 1, 1, 1)
    # print unit('Boltzmann constant')
    print ThermalConductivity(1).clarke_model(1, 1, 1)
