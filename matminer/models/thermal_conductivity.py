import math
from scipy.constants import *

__author__ = 'Saurabh Bajaj <sbajaj@lbl.gov>'


class ThermalConductivity:
    def __init__(self):
        pass

    def cahill_model(self, n, V, v_l, v_t1, v_t2):
        # http://www.sciencedirect.com/science/article/pii/S0925838814021562
        # http://www.sciencedirect.com/science/article/pii/S0927025615004395
        # http://onlinelibrary.wiley.com/doi/10.1002/adma.201400515/epdf
        # http://www.nature.com/nature/journal/v508/n7496/pdf/nature13184.pdf (full formula)
        return (1.0/2) * ((math.pi/6) ** (1.0/3)) * k * ((n/V) ** (2.0/3)) * (v_l + v_t1 + v_t2)


if __name__ == "__main__":
    print ThermalConductivity().cahill_model(1, 1, 1, 1, 1)
    # print unit('Boltzmann constant')
