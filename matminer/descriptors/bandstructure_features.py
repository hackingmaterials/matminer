from __future__ import division

__author__ = 'Anubhav Jain <ajain@lbl.gov>'


def branch_point_energy(bs, n_vb=1, n_cb=1):
    """
    Get the branch point energy as defined by:
    Schleife, Fuchs, Rodi, Furthmuller, Bechstedt, APL 94, 012104 (2009)

    TODO: incorporate kpoint weights!!

    Args:
        bs: (BandStructure) - uniform mesh bandstructure object
        n_vb: number of valence bands to include
        n_cb: number of conduction bands to include

    Returns: (int) branch point energy on same energy scale as BS eigenvalues

    """
    if bs.is_metal():
        raise ValueError("Cannot define a branch point energy for metals!")

    total_sum_energies = 0
    num_points = 0

    for spin in bs.bands:
        for kpt_idx in range(len(bs.kpoints)):
            vb_energies = []
            cb_energies = []
            for band_idx in range(bs.nb_bands):
                e = bs.bands[spin][band_idx][kpt_idx]
                if e > bs.efermi:
                    cb_energies.append(e)
                else:
                    vb_energies.append(e)
            vb_energies.sort(reverse=True)
            cb_energies.sort()
            total_sum_energies += (sum(vb_energies[0:n_vb])/n_vb +
                                   sum(cb_energies[0:n_cb])/n_cb) / 2  # TODO: multiply by kpoint weight
            num_points += 1  # TODO: set to kpoint weight

    return total_sum_energies/num_points


def absolute_band_positions(bs, **kwargs):
    """
    Absolute VBM and CBM positions with respect to branch point energy

    Args:
        bs: Bandstructure object
        **kwargs: arguments to feed into branch point energy code

    Returns:
        (vbm, cbm) - tuple of floats

    """
    bpe = branch_point_energy(bs, **kwargs)
    return (bs.get_vbm()["energy"] - bpe), (bs.get_cbm()["energy"] - bpe)

