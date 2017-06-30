from pint import UnitRegistry, set_application_registry

__author__ = 'Anubhav Jain <ajain@lbl.gov>, Saurabh Bajaj <sbajaj@lbl.gov>'

ureg = UnitRegistry()
Q_ = ureg.Quantity
set_application_registry(ureg)