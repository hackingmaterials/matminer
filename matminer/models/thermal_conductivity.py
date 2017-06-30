import math
from scipy.constants import *
from scipy.integrate import quad
from matminer.models import ureg, Q_

__author__ = 'Saurabh Bajaj <sbajaj@lbl.gov>'


class CahillSimpleModel:
    """
    Calculate Cahill thermal conductivity.

    References:
        - DOI: 10.1016/j.jallcom.2014.09.022 (Title: "Electronic structure, elastic anisotropy,
        thermal conductivity and optical properties of calcium apatite Ca5(PO4)3X (X = F, Cl or Br)")
        - DOI: 10.1016/j.commatsci.2015.07.029 (Title: "Electronic structure, mechanical properties and
        anisotropy of thermal conductivity of Y-Si-O-N quaternary crystals")
        - DOI: 10.1002/adma.201400515 (Title: "High Thermoelectric Performance in Non-Toxic Earth-Abundant
        Copper Sulfide")
    """
    @ureg.wraps(None, (None, None, 'm**3', 'm/s', 'm/s', 'm/s'), True)
    def __init__(self, n, V, v_l, v_t1, v_t2):
        """
        Args:
            n: (int) number of atoms in unit cell
            V: (float) unit cell volume (in SI units, i.e. m^(3))
            v_l: (float) longitudinal sound velocity (in SI units, i.e. m/s)
            v_t1: (float) transverse sound velocity in direction 1 (in SI units, i.e. m/s)
            v_t2: (float) transverse sound velocity in direction 2 (in SI units, i.e. m/s)

        Returns: None

        """
        self.n = n
        self.V = V
        self.v_l = v_l
        self.v_t1 = v_t1
        self.v_t2 = v_t2

    @ureg.wraps('joule/m/s/kelvin', None, True)
    def calculate(self):
        """
        Returns: (float) Cahill thermal conductivity (in SI units, i.e. joule/(m*s*kelvin))
        """
        return (1./2.) * ((math.pi/6)**(1./3.)) * k * ((self.n/self.V)**(2./3.)) * (self.v_l + self.v_t1 + self.v_t2)


def cahill_integrand(x):
    """
    Integrand function to calculate Cahill thermal conductivity.

    Args:
        x: (hbar * omega)/(k * T)   # hbar: reduced Planck's constant, omega = phonon frequency

    Returns: (float) integral value

    """
    return (x**3 * math.exp(x)) / ((math.exp(x)-1) ** 2)


def cahill_integrand_summation(v_i, T, theta):
    """
    Calculate the summation term for the Cahill thermal conductivity integrand model.
    Use this function repeatedly to calculate the total sum over all acoustic modes.

    References:
        - DOI: 10.1002/adfm.201600718 (Title: "Minimum Thermal Conductivity in Weak Topological Insulators with
        Bismuth-Based Stack Structure")
        - DOI: 10.1038/nature13184 (Title: "Ultralow thermal conductivity and high thermoelectric figure of merit in
        SnSe crystals") (full formula)

    Args:
        v_i: (float) sound velocity for the acoustic mode i (in SI units, i.e. m(s)^(-1))
        T: (float) absolute temperature (in K)
        theta: (float) Debye temperature (in K)

    Returns: (float) summation term for only *one* acoustic mode i

    """
    return v_i * (T/theta)**2 * quad(cahill_integrand, 0, theta/T)


def cahill_integrand_model(n, V, cahill_integrand_sum):
    """
    Calculate Cahill thermal conductivity using the intergrand model.

    References:
        - DOI: 10.1002/adfm.201600718 (Title: "Minimum Thermal Conductivity in Weak Topological Insulators with
        Bismuth-Based Stack Structure")
        - DOI: 10.1038/nature13184 (Title: "Ultralow thermal conductivity and high thermoelectric figure of merit in
        SnSe crystals") (full formula)

    Args:
        n: (int) number of atoms in primitive cell
        V: (float) unit cell volume (in SI units, i.e. m^(3))
        cahill_integrand_sum: (float) *sum* of the term calculate using the above function "cahill_integrand_summation"

    Returns: (float) Cahill thermal conductivity (in SI units, i.e. W(mK)^(-1))

    """
    n_d = (6 * math.pi**2 * (n/V)) ** (1./3.)
    return (math.pi/6)**(1./3.) * k * n_d**(1./3.) * cahill_integrand_sum


def clarke_model(n, E, m, V):
    """
    Calculate Clarke thermal conductivity.

    References:
        - DOI: 10.1016/j.jallcom.2014.09.022 (Title: "Electronic structure, elastic anisotropy,
        thermal conductivity and optical properties of calcium apatite Ca5(PO4)3X (X = F, Cl or Br)")
        - DOI: 10.1016/j.commatsci.2015.07.029 (Title: "Electronic structure, mechanical properties and
        anisotropy of thermal conductivity of Y-Si-O-N quaternary crystals")

    Args:
        n: (int) number of atoms in primitive cell
        E: (float) Young's modules (in SI units, i.e. Kgm(s)^(-2))
        m: (float) total mass per unit cell (in SI units, i.e. Kg)
        V: (float) unit cell volume (in SI units, i.e. m^(3))

    Returns: (float) Clarke thermal conductivity (in SI units, i.e. W(mK)^(-1))

    """
    return 0.87 * k * (((n*6.023*10**23)/m)**(2./3.)) * (E**(1./2.)) * ((m/V)**(1./6.))


def callaway_integrand(x, t_ph):
    """
    Integrand function to calculate Callaway thermal conductivity.

    Args:
        x: (hbar * omega)/(k * T)   # hbar: reduced Planck's constant, omega = phonon frequency
        t_ph: (float) phonon relaxation time (in SI units, s^(-1))

    Returns: (float) integral value

    """
    return (x**4 * math.exp(x)) / (t_ph**(-1) * (math.exp(x)-1) ** 2)


def callaway_model(v_m, T, theta, t_ph):
    """
    Calculate Callaway thermal conductivity
    (In some circumstances, a second term may be required as seen in
    http://link.springer.com/chapter/10.1007%2F0-387-25100-6_2#page-1)

    References:
        - DOI: 10.1002/adfm.201600718 (Title: "Minimum Thermal Conductivity in Weak Topological Insulators with
        Bismuth-Based Stack Structure")
        - DOI: 10.1063/1.4906225 (Title: "Critical analysis of lattice thermal conductivity of half-Heusler alloys
        using variations of Callaway model")
        - DOI: 10.1103/PhysRev.134.A1058 (Title: "Thermal Conductivity of Silicon and Germanium from 3 K to the Melting
         Point")

    Args:
        v_m: (float) speed of sound in the material (in SI units, i.e. m(s)^(-1))
        T: (float) absolute temperature (in K)
        theta: (float) Debye temperature (in K)
        t_ph: (float) phonon relaxation time (in SI units, s^(-1))

    Returns: (float) Callaway thermal conductivity (in SI units, i.e. W(mK)^(-1))

    """
    return (k / (2 * math.pi**2 * v_m)) * ((k*T)/hbar)**3 * quad(callaway_integrand, 0, theta/T, args=(t_ph,))


def slack_simple_model(M, theta, v_a, gamma, n, T):
    """
    Calculate the simple Slack thermal conductivity

    References
        - DOI: 10.1007/0-387-25100-6_2 (Title: "High Lattice Thermal Conductivity Solids")
        - DOI: 10.1002/adfm.201600718 (Title: "Minimum Thermal Conductivity in Weak Topological Insulators with
        Bismuth-Based Stack Structure")

    Args:
        M: (float) average atomic mass
        theta: (float) Debye temperature (K)
        v_a: (float) (v_a)**3 is the volume per atom (Angstroms)
        gamma: (float) Gruneisen parameter
        n: (int) number of atoms in primitive cell
        T: (float) absolute temperature (K)

    Returns: (float) Slack thermal conductivity (in SI units, i.e. W(mK)^(-1))

    """
    A_0 = 3.1 * 10**(-8)  # for v_a in Angstroms.
    # Taken from http://link.springer.com/chapter/10.1007%2F0-387-25100-6_2#page-1
    # This constant is 3.1 * 10**(-6) in Ref:- DOI: 10.1002/adfm.201600718
    return (A_0 * M * theta**3 * v_a)/(gamma * n**(2./3.) * T)


def slack_integrand(x, t_c):
    """
    Integrand function to calculate Callaway thermal conductivity.

    Args:
        x: (hbar * omega)/(k * T)   # hbar: reduced Planck's constant, omega = phonon frequency
        t_c: (float) phonon relaxation time (in SI units, s^(-1))

    Returns: (float) integral value

    """
    return t_c * x**2


def slack_integrand_model(v_m, T, theta, t_c):
    """
    Calculate Slack thermal conductivity using the integral model.
    (In high temperature regions, those higher than that of the Debye temperature of the material, the Callaway
    model is insufficient at predicting the lattice thermal conductivity. This shortfall must be addressed as many
    thermoelectric materials are designed to be used in conditions beyond the Debye temperature of the alloys and
    accurate predictions are required. At high temperatures, a modification suggested by Glassbrenner and Slack is
    made to model thermal conductivity as shown here.)

    References:
        - DOI: 10.1103/PhysRev.134.A1058 (Title: "Thermal Conductivity of Silicon and Germanium from 3 K to the Melting
         Point")
        - DOI: 10.1063/1.4906225 (Title: "Critical analysis of lattice thermal conductivity of half-Heusler alloys
        using variations of Callaway model")

    Args:
        v_m: (float) speed of sound in the material (in SI units, i.e. m(s)^(-1))
        T: (float) absolute temperature (in K)
        theta: (float) Debye temperature (in K)
        t_c: (float) combined phonon relaxation time that includes higher-order processes (in SI units, s^(-1))
            (see Ref:- DOI: 10.1103/PhysRev.134.A1058)

    Returns: (float) Slack thermal conductivity (in SI units, i.e. W(mK)^(-1))

    """
    return (k / (2 * math.pi**2 * v_m)) * ((k*T)/hbar)**3 * quad(callaway_integrand, 0, theta/T, args=(t_c,))


def keyes_model(gamma, e_m, T_m, m, V, T, A):
    """
    Calculate Keyes thermal conductivity

    References:
        - DOI: 10.1103/PhysRev.115.564 (Title: "High-Temperature Thermal Conductivity of Insulating Crystals:
        Relationship to the Melting Point")

    Args:
        gamma: (float) Gruneisen parameter
        e_m: (float) amplitude of atomic vibrations as fraction of lattice constant at which melting takes place
        T_m: (float) melting temperature (K)
        m: (float) total mass (in SI units, i.e. Kg)
        V: (float) unit cell volume (in SI units, i.e. m^(3))
        T: (float) absolute temperature (in K)
        A: (float) average atomic weight

    Returns: (float) Keyes thermal conductivity (in SI units, i.e. W(mK)^(-1))

    """
    B = (R**(3./2.))/(3 * gamma**2 * e_m**3 * (6.023*10**23)**(1.0/3))
    return (B * T_m**(3./2.) * (m/V)**(2./3.))/(T * A**(7./6.))


def debye_model(M, E, m, V):
    """
    Calculate Debye thermal conductivity.

    Args:
        M: (float) molecular mass
        E: (float) Young's modules (in SI units, i.e. Kgm(s)^(-2))
        m: (float) total mass (in SI units, i.e. Kg)
        V: (float) unit cell volume (in SI units, i.e. m^(3))

    Returns: (float) Debye thermal conductivity (in SI units, i.e. W(mK)^(-1))

    """
    return 2.489e-11 * ((1/M)**(1./3.)) * (E**0.5) * ((m/V)**(-1./6.))
