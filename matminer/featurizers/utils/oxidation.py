# Utility operations
def has_oxidation_states(comp):
    """Check if a composition object has oxidation states for each element
    TODO: Does this make sense to add to pymatgen? -wardlt
    Args:
        comp (Composition): Composition to check
    Returns:
        (boolean) Whether this composition object contains oxidation states
    """
    for el in comp.elements:
        if not hasattr(el, "oxi_state") or el.oxi_state is None:
            return False
    return True


def is_ionic(comp):
    """Determines whether a compound is an ionic compound.

    Looks at the oxidation states of each site and checks if both anions
    and cations exist

    Args:
        comp (Composition): Composition to check
    Returns:
        (bool) Whether the composition describes an ionic compound
    """

    has_cations = False
    has_anions = False

    for el in comp.elements:
        if el.oxi_state < 0:
            has_anions = True
        if el.oxi_state > 0:
            has_cations = True
        if has_anions and has_cations:
            return True
    return False
