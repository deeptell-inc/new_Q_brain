# PCCP submission bundle — new manuscript

Upload at <https://mc.manuscriptcentral.com/pccp>.

This is a **new submission**, not a revision. It supersedes
**CP-ART-06-2026-002404**, which the authors asked to withdraw during review
after finding that one of its central results was an artifact of their own
simulation protocol. `cover_letter.pdf` says so in its second paragraph, and the
editors invited a fresh submission. There is therefore no point-by-point
response and no marked-up copy — a `latexdiff` against the withdrawn version
would be misleading, since the conclusions differ rather than the wording.

| PCCP requirement | File |
|---|---|
| Manuscript, TeX + final PDF | `main.tex` + `main.pdf` (14 pp; `figures/`; bibliography is inline `thebibliography`, so no `.bib`/`.bbl` is needed) |
| Electronic Supplementary Information | `supplementary.tex` + `supplementary.pdf` (9 pp) |
| Cover letter | `cover_letter.pdf` |
| TOC entry graphic (max 8 × 4 cm) | `toc_entry.pdf` / `toc_entry.tif` — exactly 8.00 × 4.00 cm, vector / 600 dpi |
| TOC entry text (max 250 characters) | `toc_entry.txt` — 248 characters |
| Data availability statement | `\section{Data and code availability}` in `main.tex`; the standalone `data_availability.pdf` is the same statement with the module-by-module detail |
| Conflicts of interest | `\section*{Conflicts of interest}` in `main.tex` |
| Author contributions | `\section*{Author contributions}` in `main.tex` |
| High-quality figures (≥600 dpi, .tif/.eps/.pdf) | `figures/fig_schematic.pdf`, `fig_readout.pdf`, `fig_criterion.pdf` — all vector |
| CheckCIF reports | not applicable (no crystallographic data) |

`FREEZE_MANIFEST.txt` is included so that every file in the bundle can be hashed
against the state the tests were run in. `scripts/make_bundle.sh` does exactly
that check before writing the bundle, and refuses to build unless
`scripts/release_check.sh` passes.

## Blocking — must be done before upload

- **The repository the data availability statement names does not exist yet.**
  `main.tex` and `data_availability.tex` both cite
  `https://github.com/deeptell-inc/new_Q_brain`, which currently returns 404,
  and this working tree has no git remote configured. Push the repository
  (public, or public at acceptance with a private link for referees) or change
  the statement. A DAS pointing at a 404 is the kind of thing an editor checks.

## Still to do in ScholarOne (cannot be done from files)

- Link the submitting author's ORCID (H. Wakaura: 0000-0001-8381-8323).
- Decide whether to opt in to transparent peer review.
- Declare the relationship to CP-ART-06-2026-002404 in the "previous
  submission" metadata field as well as in the cover letter.
- Suggest referees, if the journal asks.

## Known deviation from the RSC template

`main.tex` is `revtex4-2` (`aps,prx,reprint`), not the RSC article template that
the withdrawn version used. The content is journal-ready — ESI wording,
conflicts, author contributions and the DAS are all in RSC form — but the
typesetting is APS. PCCP accepts a PDF in any reasonable format at initial
submission and asks for its own template at revision, so this is publishable as
it stands; converting now is a presentation choice, not a requirement. The RSC
sources to convert against are in the sibling repository at
`brain_protein_screening/manuscript/reservoir_pccp/` (`rsc.bst`, `head_foot/`,
and this paper's own predecessor in the template).

## Rebuilding

```
bash scripts/make_bundle.sh
```

Writes `manuscript/submission/` and `manuscript/pccp_submission.tar.gz`. The
bundle is deliberately not committed: it is derivable from the sources, and a
committed copy is one more thing that can go stale. Run it immediately before
uploading.
