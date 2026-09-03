#!/usr/bin/env bash
# Marked-up copies against the withdrawn submission (CP-ART-06-2026-002404).
#
# The editor asked for "all changes to the manuscript clearly marked" relative to
# the version the referee saw. That version is manuscript/withdrawn/, extracted
# byte-for-byte from the tarball that was uploaded on 2026-06-24. Almost every
# sentence differs, so the diff is mostly colour; that is the truthful picture.
#
# Two things latexdiff cannot do on its own:
#   * the ESI changed document class (article -> revtex4-2), and diffing across
#     classes leaves revtex's \author machinery choking on the old title block.
#     So the old ESI body is spliced under the NEW preamble first, and only the
#     bodies are diffed.
#   * floats deleted wholesale lose their \label, so \ref{...} in struck-out
#     text prints "??". Those refs are replaced by the number the withdrawn
#     version printed, read from the order of its figure/table environments.
set -eu
cd "$(dirname "$0")/../manuscript"
export PATH="/Library/TeX/texbin:$PATH"
CFG='PICTUREENV=(?:picture|DIFnomarkup|tabular)[\w\d*@]*'

latexdiff --config="$CFG" withdrawn/main.tex main.tex > main_diff.tex 2>/dev/null

python3 - <<'PY'
import pathlib
new = pathlib.Path("supplementary.tex").read_text()
old = pathlib.Path("withdrawn/esi.tex").read_text()
head = new[:new.index("\\maketitle") + len("\\maketitle")]
body = old[old.index("\\maketitle") + len("\\maketitle"):]
pathlib.Path("withdrawn/_esi_spliced.tex").write_text(head + body)
PY
latexdiff --config="$CFG" withdrawn/_esi_spliced.tex supplementary.tex > supplementary_diff.tex 2>/dev/null
rm -f withdrawn/_esi_spliced.tex

python3 - <<'PY'
import pathlib, re

def fix(old_f, new_f, diff_f, prefix):
    old = pathlib.Path(old_f).read_text()
    num = {}
    for env in ("figure", "table"):
        bodies = re.findall(r"\\begin\{%s\*?\}(.*?)\\end\{%s\*?\}" % (env, env), old, re.S)
        for i, body in enumerate(bodies, 1):
            for lab in re.findall(r"\\label\{([^}]+)\}", body):
                num[lab] = prefix + str(i)
    d = pathlib.Path(diff_f); s = d.read_text()
    s = s.replace(r"\graphicspath{{figures/}{../figures/}}",
                  r"\graphicspath{{figures/}{../figures/}{withdrawn/}}")
    # a \label inside struck-out text is never executed, so "live" is read from
    # the new document, not from the diff
    live = set(re.findall(r"\\label\{([^}]+)\}", pathlib.Path(new_f).read_text()))
    for lab, n in num.items():
        if lab not in live:
            s = s.replace(r"\ref{%s}" % lab, n)
    # the withdrawn ESI used mhchem's \ce; the revtex supplement does not load it
    if "mhchem" not in s:
        s = s.replace("\\begin{document}",
                      "\\usepackage[version=3]{mhchem}\n\\begin{document}", 1)
    d.write_text(s)

fix("withdrawn/main.tex", "main.tex", "main_diff.tex", "")
fix("withdrawn/esi.tex", "supplementary.tex", "supplementary_diff.tex", "S")
PY

for d in main_diff supplementary_diff; do
  for _ in 1 2 3; do pdflatex -interaction=nonstopmode "$d.tex" > /dev/null 2>&1 || true; done
  printf "%-20s pages=%-3s errors=%-2s undefined=%s\n" "$d:" \
    "$(pdfinfo "$d.pdf" 2>/dev/null | awk '/^Pages/{print $2}')" \
    "$(grep -c '^!' "$d.log")" \
    "$(grep -c 'Reference .* undefined\|Citation .* undefined' "$d.log")"
done
