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
