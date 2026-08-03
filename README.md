# new_Q_brain

Code, data and manuscript for

> **What a radical-pair quantum reservoir would require: readable capacity, no
> quantum advantage, and a nuclear-register criterion**
> Hikaru Wakaura and Taiki Tanimae, QIRI (Quantum Integrated Research Institute Inc.)

Can the spin chemistry of a cryptochrome-type radical pair *process* information?
Posed — as it must be for a warm, noisy system — as reservoir computing rather
than gate-model computation, the answer turns out to be neither the optimistic
nor the pessimistic one usually assumed.

## The three results

| | Result | Where |
|---|---|---|
| 1 | The reservoir is **readable by ordinary chemistry**. Time-resolved product yield gives out-of-sample IPC $= 2.0$; the nuclear polarisation carried away by the product raises it to $4.7$. No spectroscopy required. | `readout_routes.py` |
| 2 | There is **no quantum advantage**. A classical echo-state network with the same number of readout features matches or exceeds the five-spin reservoir ($9.4$ vs $5.6$). | `final_numbers.py`, `qrc_benchmarks.py` |
| 3 | The memory horizon is set **not** by the microsecond pair lifetime but by the turnover interval, because the nuclear register survives recombination in the diamagnetic product. A physiological $1$–$100$ ms turnover changes MC by $<0.2\%$ and puts the horizon at $3$–$280$ ms. The binding constraint is chemical: the **same nuclear register must be reused** between turnovers, or the capacity collapses to a memoryless read-back. | `readout_routes.py` (`clock_scan`, `register_reuse`) |

A mechanistic point falls out of (1): at $J=0$ the reduced state of *either*
electron of a newly born pair is maximally mixed for every input — the input
lives entirely in the two-electron correlation. Spin-selective recombination is
therefore what *writes* the input into the nuclear register. In a separated
pair, CIDNP is not an optional extra readout channel; it is the write mechanism.

## What makes this repository different from its predecessor

This work supersedes an earlier analysis (withdrawn from *PCCP* as
CP-ART-06-2026-002404) in which the reservoir was driven by resetting a single
electron. That abstraction destroys the electron–electron correlation and, for a
separated pair, pins the singlet probability at exactly $1/4$ regardless of
input — the "readout collapse" it produced was an artifact, not chemistry. The
correct drive, implemented here, retains the nuclei and creates a new
spin-correlated pair each cycle:

$$\rho_{\rm e}(s) = s\,|S\rangle\langle S| + (1-s)\,P_{\rm T}/3$$

`qbscreen/corrected_injection.py` runs both side by side: IPC $0.000$ for the old
single-electron reset, $2.94$ for the corrected S/T birth, $3.12$ for its
S–T$_0$ coherent variant.

## Layout

```
qbscreen/                    the package
  spin_dynamics.py           spin operators, hyperfine/Zeeman Hamiltonians, singlet projector
  master_equation.py         Liouville-space Haberkorn solver + electronic dephasing
  reservoir.py               reservoir Hamiltonian, propagator, out-of-sample IPC/MC estimators
  corrected_injection.py     correlated-pair birth channel and the single-electron-reset control
  readout_routes.py          seven biologically accessible readouts, turnover clock, register reuse
  qrc_benchmarks.py          classical echo-state baselines, NARMA2, coherence-time scan
  final_numbers.py           protocol-consistent recomputation of every headline number
  reanalysis.py              the R1–R6 corrective analyses
  ensemble.py                heterogeneous-ensemble / spatial-resolution analysis
  quantum_vs_classical.py    quantum-vs-classical dephasing sweep
  honest_mfe.py              5-spin magnetic-field-effect reference used by the solver tests
  tests/                     28 tests covering the solver, the capacity bound, and the estimators
manuscript/                  main.tex, supplementary.tex (ESI), cover letter, DAS, figures
simulation_results/          archived JSON outputs behind every figure and table
```

## Reproduce

```bash
pip install -e .
python -m pytest qbscreen/tests/ -q
```

Then, from the repository root (the scripts write into `simulation_results/`):

```bash
python -m qbscreen.final_numbers
```

```bash
python -m qbscreen.readout_routes
```

`readout_routes` accepts a stage name — `survey`, `clock`, `reuse`, `grid` — and
runs all four when given none. The remaining generators are
`python -m qbscreen.corrected_injection`, `qbscreen.reanalysis`,
`qbscreen.qrc_benchmarks`, `qbscreen.ensemble` and
`qbscreen.quantum_vs_classical`.

Figures are rebuilt from the JSON with `python manuscript/make_schematic.py` and
`python manuscript/make_fig_criterion.py`.

Every capacity in the paper is **out of sample** — trained on the first half of
each run, scored on the held-out second half — and is quoted with its
shuffled-input null floor and its convergence in sample length. Protocol
constants (input length, seed count) are the defaults in each function's
signature; capacities are means over 4–10 input realisations.

## Requirements

Python ≥ 3.10, NumPy ≥ 1.24, SciPy ≥ 1.10, Matplotlib ≥ 3.7. Nothing
proprietary, no external quantum-chemistry binaries.

## Licence

MIT. See [LICENSE](LICENSE).
