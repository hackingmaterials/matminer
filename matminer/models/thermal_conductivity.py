import math
from scipy.constants import *
from scipy.integrate import quad

__author__ = 'Saurabh Bajaj <sbajaj@lbl.gov>'


def cahill_simple_model(n, V, v_l, v_t1, v_t2):
    """
    Calculate Cahill thermal conductivity.

    References:
        http://www.sciencedirect.com/science/article/pii/S0925838814021562
        http://www.sciencedirect.com/science/article/pii/S0927025615004395
        http://onlinelibrary.wiley.com/doi/10.1002/adma.201400515/epdf
        http://www.nature.com/nature/journal/v508/n7496/pdf/nature13184.pdf (full formula)

    Args:
        n: (int) number of atoms in unit cell
        V: (float) unit cell volume (in SI units, i.e. m^(-3))
        v_l: (float) longitudinal sound velocity (in SI units, i.e. m(s)^(-1))
        v_t1: (float) transverse sound velocity in direction 1 (in SI units, i.e. m(s)^(-1))
        v_t2: (float) transverse sound velocity in direction 2 (in SI units, i.e. m(s)^(-1))

    Returns: (float) Cahill thermal conductivity (in SI units, i.e. W(mK)^(-1))

    """
    return (1.0 / 2) * ((math.pi / 6) ** (1.0 / 3)) * k * ((n/V) ** (2.0 / 3)) * (v_l + v_t1 + v_t2)


def cahill_integrand(x):
    """
    Integrand function to calculate Cahill thermal conductivity.

    Args:
        x: (hbar * omega)/(k * T)   # hbar: reduced Planck's constant, omega = phonon frequency

    Returns: (float) integral value

    """
    return (x**3 * math.exp(x)) / ((math.exp(x) - 1)**2)


def cahill_integrand_summation(v_i, T, theta):
    """
    Calculate the summation term for the Cahill thermal conductivity integrand model.
    Use this function repeatedly to calculate the total sum over all acoustic modes.

    References:
        http://onlinelibrary.wiley.com/doi/10.1002/adfm.201600718/full
        http://www.nature.com/nature/journal/v508/n7496/pdf/nature13184.pdf (full formula)

    Args:
        v_i: (float) sound velocity for the acoustic mode i (in SI units, i.e. m(s)^(-1))
        T: (float) absolute temperature (in K)
        theta: (float) Debye temperature (in K)

    Returns: (float) summation term for only *one* acoustic mode i

    """
    return v_i * (T/theta)**2 * quad(cahill_integrand, 0, theta/T)


def cahill_integrand_model(N, V, cahill_integrand_sum):
    """
    Calculate Cahill thermal conductivity using the intergrand model.

    References:
        http://onlinelibrary.wiley.com/doi/10.1002/adfm.201600718/full
        http://www.nature.com/nature/journal/v508/n7496/pdf/nature13184.pdf (full formula)

    Args:
        N: (int) number of atoms in primitive cell
        V: (float) unit cell volume (in SI units, i.e. m^(-3))
        cahill_integrand_sum: (float) *sum* of the term calculate using the above function "cahill_integrand_summation"

    Returns: (float) Cahill thermal conductivity (in SI units, i.e. W(mK)^(-1))

    """
    n_d = (6 * math.pi**2 * (N/V)) ** (1.0/3)
    return (math.pi/6)**(1.0/3) * k * (n_d)**(1.0/3) * cahill_integrand_sum


def clarke_model(M, E, m, V):
    """
    Calculate Clarke thermal conductivity.

    References:
        http://www.sciencedirect.com/science/article/pii/S0925838814021562
        http://www.sciencedirect.com/science/article/pii/S0927025615004395

    Args:
        M: (float) molecular mass
        E: (float) Young's modules (in SI units, i.e. Kgm(s)^(-2))
        m: (float) total mass (in SI units, i.e. Kg)
        V: (float) unit cell volume (in SI units, i.e. m^(-3))

    Returns: (float) Clarke thermal conductivity (in SI units, i.e. W(mK)^(-1))

    """
    return 0.87 * k * ((1 / M) ** (2.0 / 3)) * (E ** (1.0 / 2)) * ((m/V) ** (1.0 / 6))


def callaway_integrand(x, t_ph):
    """
    Integrand function to calculate Callaway thermal conductivity.

    Args:
        x: (hbar * omega)/(k * T)   # hbar: reduced Planck's constant, omega = phonon frequency
        t_ph: phonon relaxation time (in SI units, s^(-1))

    Returns: (float) integral value

    """
    return (x**4 * math.exp(x)) / (t_ph**(-1) * (math.exp(x) - 1)**2)


def callaway_model(v_m, T, theta, t_ph):
    """
    Calculate Callaway thermal conductivity
    (In some circumstances, a second term may be required as seen in
    http://link.springer.com/chapter/10.1007%2F0-387-25100-6_2#page-1)

    References:
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
    return (k / (2 * math.pi ** 2 * v_m)) * ((k * T) / hbar) ** 3 * quad(callaway_integrand, 0, theta/T, args=(t_ph,))


def slack_simple_model(M, theta, v_a, gamma, n, T):
    """
    Calculate the simple Slack thermal conductivity

    References
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
    A_0 = 3.1 * 10**(-8)  # for v_a in Angstroms.
    # Taken from http://link.springer.com/chapter/10.1007%2F0-387-25100-6_2#page-1
    # This constant is 3.1 * 10**(-6) in http://onlinelibrary.wiley.com/doi/10.1002/adfm.201600718/full
    return (A_0 * M * theta**3 * v_a)/(gamma * n**(2.0/3) * T)


def slack_integrand(x, t_c):
    """
    Integrand function to calculate Callaway thermal conductivity.

    Args:
        x: (hbar * omega)/(k * T)   # hbar: reduced Planck's constant, omega = phonon frequency
        t_c: phonon relaxation time (in SI units, s^(-1))

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
    return (k / (2 * math.pi ** 2 * v_m)) * ((k * T) / hbar) ** 3 * quad(callaway_integrand, 0, theta/T, args=(t_c,))

if __name__ == "__main__":
    print cahill_simple_model(1, 1, 1, 1, 1)
    # print unit('Boltzmann constant')
    print clarke_model(1, 1, 1, 1)
