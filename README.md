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
| 1 | The reservoir is **readable by ordinary chemistry**. Time-resolved product yield gives out-of-sample IPC $= 2.0$; the nuclear polarisation carried away by the product raises it to $4.7$. No spectroscopy required. **Caveat added in revision:** once the register is restricted to protons (the flavin ¹⁴N is quadrupolar and relaxes in µs) and the ensemble is allowed to desynchronise, the time-resolved kinetic readout dies and CIDNP survives at ~29% of the excess capacity. Only these two of the seven routes were re-evaluated under those restrictions. | `readout_routes.py` |
| 2 | There is **no quantum advantage**. A classical echo-state network with the same number of readout features exceeds the five-spin reservoir ($9.4$ vs $5.6$); at the cryptochrome point it wins on both channel-matched accountings (ESN 8-node $6.9$ vs quantum 8-channel $4.7$; ESN 5-node $4.5$ vs quantum 5-channel $2.0$), and the physical-unit accounting separates them by less than one standard deviation. | `final_numbers.py`, `qrc_benchmarks.py` |
| 3 | The memory horizon is set **not** by the microsecond pair lifetime but by the turnover interval, because the nuclear register survives recombination in the diamagnetic product. Across five decades of turnover interval MC falls by $1.6\%$ (12 seeds, paired), putting the horizon at $19$–$188$ ms for a $10$–$100$ ms turnover — but only if the register outlives the pause. At the geomagnetic field nuclear relaxation is in the extreme-narrowing regime, so a proton on the intact protein relaxes in $2.4$ ms and caps the horizon at $5.5$ ms; the register must reorient $\gtrsim3\times$ faster than the protein. The binding constraint is chemical: the **same nuclear register must be reused** between turnovers, or the capacity collapses to a memoryless read-back. | `readout_routes.py` (`clock_scan`, `register_reuse`) |

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
  ensemble.py                SUPERSEDED (single-electron reset) - control only, no reported result
  quantum_vs_classical.py    SUPERSEDED (single-electron reset) - control only, no reported result
  honest_mfe.py              5-spin magnetic-field-effect reference used by the solver tests
  panel_response.py          referee controls: product register, route floors, delay kernel,
                             12-seed clock + paired test, MC grid, T1/T2 channel, CRY baselines,
                             ridge and shot-noise scans, electron-only coherence control
  semiclassical.py           classical-spin reference model for the coherence fraction (S8)
  nuclide_register.py        per-nucleus register relaxation: can 1H alone hold it? (S9)
  ensemble_pooled.py         pooled ensembles: coupling heterogeneity, turnover desynchronisation,
                             and heterogeneous relaxation with its mean-field control (S10)
  general_spin.py            anisotropic hyperfine tensors and spin-1 14N (S11)
  relaxation_estimate.py     predicted nuclear T1/T2, validated against measured systems (S12)
  turnover_estimate.py       turnover interval from photophysics/catalysis; feasible region (S12)
  product_carrier_audit.py   adverse results re-run with the product register (S4)
  tests/                     258 tests: solver, capacity bound, estimators, claim-binding and table-row regressions (capacity and SD cells, input-parameter table, cross-document table references; derived-time and raw-IPC columns not yet bound)
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

```bash
python -m qbscreen.corrected_injection
python -m qbscreen.reanalysis
python -m qbscreen.qrc_benchmarks
python -m qbscreen.panel_response          # referee controls (writes simulation_results/panel/)
python -m qbscreen.semiclassical           # S8  classical-spin reference
python -m qbscreen.semiclassical floor     # S8  its own memoryless floor
python -m qbscreen.semiclassical conv      # S8  integration-step convergence
python -m qbscreen.nuclide_register        # S9  per-nucleus register relaxation
python -m qbscreen.ensemble_pooled         # S10 heterogeneity, desync, q=1 floor, heterogeneous tau_c
python -m qbscreen.general_spin            # S11 anisotropy and spin-1 14N
python -m qbscreen.relaxation_estimate     # S12 predicted nuclear T1/T2  (run before turnover)
python -m qbscreen.turnover_estimate       # S12 turnover interval and feasible region
python -m qbscreen.product_carrier_audit   # adverse results re-run with the product register
```

`relaxation_estimate` must be run before `turnover_estimate`, which consumes its
`tau_c_scan` output. `open1_ensemble_desync_floor.json` — the register-wiped control that supplies the
entire `excess` column of the SI jitter table — was orphaned until the set-5 audit
and is now generated by `python -m qbscreen.ensemble_pooled floor`, which
reproduces the shipped file exactly. One shipped file is deliberately outside the reproducible set:
`open5_feasible_region.json`, which duplicates the
`feasible_region` block already inside `open5_turnover_estimate.json` and from
which nothing is quoted independently. It is a snapshot predating the `feasible`
field the current generator emits, so it is excluded from the reproducible set by
name in `scripts/regenerate_all.sh` rather than silently counted as reproduced.

`manuscript/figures/fig_baseline.pdf` is produced by `qbscreen.qrc_benchmarks`
and is frozen with the rest of the outputs, but no manuscript figure uses it —
the three figures in the paper are `fig_schematic`, `fig_readout` and
`fig_criterion`.

`simulation_results/panel_before_regen/` holds 26 JSON files (plus nine `.log`
files, 35 entries in all) archived from an earlier
revision round. No command writes them, no test reads them, and they are not in
`FREEZE_MANIFEST.txt`; they are kept only so the pre-revision values remain
inspectable. They are excluded from the reproducible set by
`scripts/regenerate_all.sh`.

The legacy root-level outputs `coherence_tradeoff.json`, `cryptochrome_reality.json`,
`ensemble_results.json`, `quantum_vs_classical.json`, `reservoir_readout_realism.json`
and `reservoir_results.json` come from `qbscreen.qrc_benchmarks tradeoff`,
`qbscreen.qrc_benchmarks cryptochrome`, `qbscreen.ensemble`,
`qbscreen.quantum_vs_classical` and `qbscreen.reservoir`. All five commands are in
the reproduce list above, so a clean checkout recreates these files too; until
set 9 they were shipped and frozen but produced by no documented command.
`qbscreen.ensemble`, `qbscreen.quantum_vs_classical` and the `run_reservoir`
driver in `qbscreen.reservoir` implement the **superseded** single-electron
reset. They are retained as controls and no reported result derives from them.

Figures are rebuilt from the JSON with `python manuscript/make_schematic.py` and
`python manuscript/make_fig_criterion.py`.

Every capacity in the paper is **out of sample** — trained on the first half of
each run, scored on the held-out second half — and is quoted with its
shuffled-input null floor and its convergence in sample length. Protocol
constants (input length, seed count) are the defaults in each function's
signature; capacities are means over 3–12 input realisations (the ridge, shot-noise and semiclassical step-size diagnostics, and the superseded clock scan kept in ESI Table S7, are single-realisation) (clock 12, register-reuse / nuclear-channel / semiclassical 4, ensemble/anisotropy 3, the rest 6–8).

## Requirements

Python ≥ 3.10, NumPy ≥ 1.24, SciPy ≥ 1.10, Matplotlib ≥ 3.7. Nothing
proprietary, no external quantum-chemistry binaries.

## Licence

MIT. See [LICENSE](LICENSE).
