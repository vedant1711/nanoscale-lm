"""Runnable scripts: figure generators, ablation runners and report writers.

This is a package so that the scripts are importable (and therefore testable) as
``scripts.<name>`` rather than only executable as files. ``tests/unit/test_ablation.py``
imports the shipped ablation suites to check they build valid configs for every arm,
which is the kind of thing that otherwise only breaks when you run the eight-minute job.
"""
