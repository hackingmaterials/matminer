# Utility operations
def has_oxidation_states(comp):
    """Check if a composition object has oxidation states for each element

    Args:
        comp (Composition): Composition to check
    Returns:
        (boolean) Whether this composition object contains oxidation states
    """
    for el in comp.elements:
        if not hasattr(el, "oxi_state") or el.oxi_state is None:
            return False
    return True
