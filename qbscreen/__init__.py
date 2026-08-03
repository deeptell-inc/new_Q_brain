"""qbscreen — radical-pair spin dynamics as a quantum reservoir.

Exact Liouville-space Haberkorn master equation with electronic decoherence,
driven as a real chemical turnover does: the nuclei are retained and a new
spin-correlated electron pair is born each cycle, its singlet character
encoding the input.  Provides the out-of-sample information-processing and
memory capacity estimators, the biologically accessible readout routes
(kinetics, both exit channels, heterogeneous lifetimes, CIDNP, accumulated
pool), the classical echo-state baselines, and the nuclear-register reuse
criterion.

This package is the code base of

    H. Wakaura and T. Tanimae,
    "What a radical-pair quantum reservoir would require: readable capacity,
     no quantum advantage, and a nuclear-register criterion".

See manuscript/ for the paper and simulation_results/ for the archived
numerical outputs.
"""

__version__ = "0.2.0"
__author__ = "Hikaru Wakaura, Taiki Tanimae"
__email__ = "h.wakaura@qiri.co.jp"
