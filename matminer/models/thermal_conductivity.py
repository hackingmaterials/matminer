import math
from scipy.constants import *
from scipy.integrate import quad

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
        http://www.sciencedirect.com/science/article/pii/S0925838814021562
        http://www.sciencedirect.com/science/article/pii/S0927025615004395
        http://onlinelibrary.wiley.com/doi/10.1002/adma.201400515/epdf
        http://www.nature.com/nature/journal/v508/n7496/pdf/nature13184.pdf (full formula)

        Args:
            n: (int) number of atoms
            v_l: (float) longitudinal sound velocity (in SI units, i.e. m(s)^(-1))
            v_t1: (float) transverse sound velocity in direction 1 (in SI units, i.e. m(s)^(-1))
            v_t2: (float) transverse sound velocity in direction 2 (in SI units, i.e. m(s)^(-1))

        Returns: (float) Cahill thermal conductivity (in SI units, i.e. W(mK)^(-1))

        """
        return (1.0 / 2) * ((math.pi / 6) ** (1.0 / 3)) * k * ((n / self.volume) ** (2.0 / 3)) * (v_l + v_t1 + v_t2)

    def clarke_model(self, M, E, m):
        """
        Calculate Clarke thermal conductivity.

        References:
        http://www.sciencedirect.com/science/article/pii/S0925838814021562
        http://www.sciencedirect.com/science/article/pii/S0927025615004395

        Args:
            M: (float) molecular mass
            E: (float) Young's modules (in SI units, i.e. Kgm(s)^(-2))
            m: (float) total mass (in SI units, i.e. Kg)

        Returns: (float) Clarke thermal conductivity (in SI units, i.e. W(mK)^(-1))

        """
        return 0.87 * k * ((1 / M) ** (2.0 / 3)) * (E ** (1.0 / 2)) * ((m / self.volume) ** (1.0 / 6))

    def callaway_integrand(self, x, t_ph):
        """
        Integrand function to calculate Callaway thermal conductivity.

        Args:
            x: (hbar * omega)/(k * T)   # hbar: reduced Planck's constant, omega = phonon frequency
            t_ph: phonon relaxation time (in SI units, s^(-1))

        Returns: (float) integral value

        """
        return (x**4 * math.exp(x)) / (t_ph**(-1) * (math.exp(x) - 1)**2)

    def callaway_model(self, v_m, T, theta, t_ph):
        """
        Calculate Callaway thermal conductivity
        (In some circumstances, a second term may be required as seen in
        http://link.springer.com/chapter/10.1007%2F0-387-25100-6_2#page-1)

        # References:
        http://onlinelibrary.wiley.com/doi/10.1002/adfm.201600718/full
        http://scitation.aip.org/content/aip/journal/jap/117/3/10.1063/1.4906225
        http://journals.aps.org/pr/pdf/10.1103/PhysRev.134.A1058

        Args:
            v_m: speed of sound in the material (in SI units, i.e. m(s)^(-1))
            T: absolute temperature (in K)
            theta: Debye temperature (in K)
            t_ph: phonon relaxation time (in SI units, s^(-1))

        Returns: (float) Callaway thermal conductivity (in SI units, i.e. W(mK)^(-1))

        """
        return (k / (2 * math.pi ** 2 * v_m)) * ((k * T) / hbar) ** 3 * quad(ThermalConductivity(1).callaway_integrand,
                                                                             0, theta / T, args=(t_ph,))

    def slack_integrand(self, x, t_c):
        """
        Integrand function to calculate Callaway thermal conductivity.

        Args:
            x: (hbar * omega)/(k * T)   # hbar: reduced Planck's constant, omega = phonon frequency
            t_c: phonon relaxation time (in SI units, s^(-1))

        Returns: (float) integral value

        """
        return t_c * x**2

    def slack_integrand_model(self, v_m, T, theta, t_c):
        """
        Calculate Slack thermal conductivity using the integral model.
        (In high temperature regions, those higher than that of the Debye temperature of the material, the Callaway
        model is insufficient at predicting the lattice thermal conductivity. This shortfall must be addressed as many
        thermoelectric materials are designed to be used in conditions beyond the Debye temperature of the alloys and
        accurate predictions are required. At high temperatures, a modification suggested by Glassbrenner and Slack is
        made to model thermal conductivity as shown here.)

        # References:
        http://journals.aps.org/pr/pdf/10.1103/PhysRev.134.A1058
        http://scitation.aip.org/content/aip/journal/jap/117/3/10.1063/1.4906225

        Args:
            v_m: speed of sound in the material (in SI units, i.e. m(s)^(-1))
            T: absolute temperature (in K)
            theta: Debye temperature (in K)
            t_c: combined phonon relaxation time that includes higher-order processes (in SI units, s^(-1))
                (see Ref. http://journals.aps.org/pr/pdf/10.1103/PhysRev.134.A1058)

        Returns: (float) Slack thermal conductivity (in SI units, i.e. W(mK)^(-1))

        """
        return (k / (2 * math.pi ** 2 * v_m)) * ((k * T) / hbar) ** 3 * quad(ThermalConductivity(1).callaway_integrand,
                                                                             0, theta / T, args=(t_c,))

    def slack_simple_model(self, M, theta, v_a, gamma, n, T):
        """
        Calculate the simple Slack thermal conductivity

        # References
        http://link.springer.com/chapter/10.1007%2F0-387-25100-6_2#page-1
        http://onlinelibrary.wiley.com/doi/10.1002/adfm.201600718/full

        Args:
            M: average atomic mass
            theta: Debye temperature (in K)
            v_a: (v_a)**3 is the volume per atom (in Angstroms)
            gamma: Gruneisen parameter
            n: number of atoms in primitive cell
            T: absolute temperature (in K)

        Returns: (float) Slack thermal conductivity (in SI units, i.e. W(mK)^(-1))

        """
        A_0 = 3.1 * 10**(-8) # for v_a in Angstroms.
        # Taken from http://link.springer.com/chapter/10.1007%2F0-387-25100-6_2#page-1
        # This constant is 3.1 * 10**(-6) in http://onlinelibrary.wiley.com/doi/10.1002/adfm.201600718/full
        return (A_0 * M * theta**3 * v_a)/(gamma * n**(2.0/3) * T)

if __name__ == "__main__":
    print ThermalConductivity(1).cahill_model(1, 1, 1, 1)
    # print unit('Boltzmann constant')
    print ThermalConductivity(1).clarke_model(1, 1, 1)
