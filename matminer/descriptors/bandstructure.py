from __future__ import division, unicode_literals, print_function

from pymatgen import Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

__author__ = 'Anubhav Jain <ajain@lbl.gov>'


def branch_point_energy(bs, n_vb=1, n_cb=1):
    """
    Get the branch point energy as defined by:
    Schleife, Fuchs, Rodi, Furthmuller, Bechstedt, APL 94, 012104 (2009)

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
    kpt_wts = SpacegroupAnalyzer(bs.structure).get_kpoint_weights(
        [k.frac_coords for k in bs.kpoints])
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
            total_sum_energies += (sum(vb_energies[0:n_vb]) / n_vb +
                                   sum(cb_energies[0:n_cb]) / n_cb) * kpt_wts[kpt_idx] / 2

            num_points += kpt_wts[kpt_idx]

    return total_sum_energies / num_points


def absolute_band_positions_bpe(bs, target_gap=None, **kwargs):
    """
    Absolute VBM and CBM positions with respect to branch point energy

    Args:
        bs: Bandstructure object
        target_gap: if a better band gap is known, shift band positions by this gap
        **kwargs: arguments to feed into branch point energy code

    Returns:
        (vbm, cbm) - tuple of floats

    """
    vbm = bs.get_vbm()["energy"]
    cbm = bs.get_cbm()["energy"]
    shift = 0
    if target_gap:
        # for now, equal shift to VBM / CBM
        shift = (target_gap - (cbm - vbm)) / 2

    bpe = branch_point_energy(bs, **kwargs)
    return (vbm - bpe - shift), (cbm - bpe + shift)


def band_center(comp):
    """
    Estimate absolution position of band center using geometric mean of electronegativity
    Ref: Butler, M. a. & Ginley, D. S. Prediction of Flatband Potentials at
    Semiconductor-Electrolyte Interfaces from Atomic Electronegativities.
    J. Electrochem. Soc. 125, 228 (1978).

    Args:
        comp: (Composition)

    Returns: (float) band center

    """
    prod = 1.0
    for el, amt in comp.get_el_amt_dict().iteritems():
        prod = prod * (Element(el).X ** amt)

    return -prod ** (1 / sum(comp.get_el_amt_dict().values()))
