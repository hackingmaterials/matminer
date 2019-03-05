

def avg_structure_stats(structures):
    preallocated = [[None] * len(structures)]
    stats = {"n_sites": preallocated,
             "is_ordered": preallocated}
    for i, s in enumerate(structures):
        stats["is_ordered"][i] = s.is_ordered
        stats["n_sites"][i] = len(s.sites)
    return {"n_sites": len(structure.sites),
            "is_ordered": structure.is_ordered}

