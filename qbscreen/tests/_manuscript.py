"""The manuscript sources are supplied to the journal, not distributed here.

Every test that reads a .tex/.aux/.md under manuscript/ binds the printed paper
to the data. Those files stay out of the public repository by decision of the
authors (the repository carries code, results and figures), so on a clone the
binding tests cannot run. They skip with this reason rather than fail, and the
count the paper quotes is the collected count, which is unchanged.
"""
import pathlib
import pytest

MAN = pathlib.Path(__file__).resolve().parents[2] / "manuscript"
requires_manuscript = pytest.mark.skipif(
    not (MAN / "main.tex").exists(),
    reason="manuscript sources are supplied with the submission, not distributed "
           "in the public repository; the binding tests run on the authors' tree")
