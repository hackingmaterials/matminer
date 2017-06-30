from pymatgen.core.units import unitized, FloatWithUnit

__author__ = 'Anubhav Jain <ajain@lbl.gov>'

"""
This is a skeleton of how a model might look like
"""

# TODO: base class useful?
class SoundVelocityModel():
    """
    This is an example model
    """

    def __init__(self):
        # any global options for the model set here
        # e.g. data mining model parameters or coefficients for terms
        pass

    @unitized(unit="kg")
    def compute(self, input_data):
        return FloatWithUnit(input_data, "g")

    def citation(self):
        return "I have no idea where this model came from"

    # maybe a helper method if you need to process an entire column of data, e.g. process Pandas dataframe


def effective_cubic_elasticconstant(direction, mode, C_11=None, C_12=None, C_44=None):
    """
    Calculate effective elastic constant (C_eff) for a cubic material.

    References:
        http://unlcms.unl.edu/cas/physics/tsymbal/teaching/SSP-927/Section%2004_Elastic_Properties.pdf

    Args:
        direction: (str) direction of sound velocity. Choose from Choose from "100", "110", and "111".
        mode: (str) wave mode. Choose from "longitudinal", "transverse1", and "transverse2".
        C_11: elastic constant C_xxxx (N/m^2)
        C_12: elastic constant C_xxyy (N/m^2)
        C_44: elastic constant C_yzyz (N/m^2)

    Returns: (float) effective elastic constant (C_eff) for a cubic material (N/m^2).

    """
    if direction == '100':
        if mode == 'longitudinal':
            return C_11
        elif mode in ['transverse1', 'transverse2']:
            return C_44
        else:
            raise ValueError('Invalid "mode". Choose from "longitudinal", "transverse1", and "transverse2".')
    elif direction == '110':
        if mode == 'longitudinal':
            return (C_11 + C_12 + 2*C_44)/2
        elif mode == 'transverse1':
            return C_44
        elif mode == 'transverse2':
            return (C_11 - C_12)/2
        else:
            raise ValueError('Invalid "mode". Choose from "longitudinal", "transverse1", and "transverse2".')
    elif direction == '111':
        if mode == 'longitudinal':
            return (C_11 + 2*C_12 + 4*C_44)/3
        elif mode in ['transverse1', 'transverse2']:
            return (C_11 - C_12 + C_44)/3
        else:
            raise ValueError('Invalid "mode". Choose from "longitudinal", "transverse1", and "transverse2".')
    else:
        raise ValueError('Invalid "direction". Choose from Choose from "100", "110", and "111".')


def sound_velocity(C_eff, rho):
    """
    Calculate sound velocity (v) in a material from elastic constant.

    Args:
        C_eff: (float) effective elastic constant (N/m^2)
        rho: (float) material density (kg/m^3)

    Returns: (float) velocity of sound (m/s) in the material

    """
    return C_eff/rho


if __name__ == "__main__":

    svm = SoundVelocityModel()
    print(svm.compute(5))
