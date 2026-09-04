"""The data-availability statement must list exactly the commands that reproduce.

data_availability.tex tells a referee which commands to run. scripts/regenerate_all.sh
is what we actually verify against the shipped bytes. When those two lists drift,
the paper hands the referee a recipe nobody has tested -- which happened: the DAS
was missing corrected_injection, reanalysis and both semiclassical sub-commands
while claiming every reported number could be regenerated.
"""

import pathlib
import re
from qbscreen.tests._manuscript import requires_manuscript

pytestmark = requires_manuscript

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _das_commands():
    t = (ROOT / "manuscript" / "data_availability.tex").read_text().replace("\\_", "_")
    return [m.group(1).strip()
            for m in re.finditer(r"python -m qbscreen\.([a-z_]+(?: [a-z]+)?)", t)]


def _script_commands():
    t = (ROOT / "scripts" / "regenerate_all.sh").read_text()
    return [m.group(1).strip()
            for m in re.finditer(r"^run qbscreen\.([a-z_]+(?: [a-z]+)?)", t, re.M)]


def test_das_lists_every_command_the_reproduce_script_runs():
    missing = [c for c in _script_commands() if c not in _das_commands()]
    assert not missing, (
        "regenerate_all.sh runs these but the data-availability statement does "
        f"not list them: {missing}")


def test_das_lists_no_command_the_reproduce_script_omits():
    """A command in the DAS that the script never runs is unverified: nobody has
    checked it produces what the paper says it does."""
    extra = [c for c in _das_commands() if c not in _script_commands()]
    assert not extra, (
        "the data-availability statement lists these but regenerate_all.sh does "
        f"not run them, so they are unverified: {extra}")


def test_every_listed_command_names_a_real_module():
    import importlib.util
    for c in set(_das_commands()) | set(_script_commands()):
        mod = c.split()[0]
        assert importlib.util.find_spec(f"qbscreen.{mod}") is not None, \
            f"qbscreen.{mod} does not exist but is listed as a reproduce command"
