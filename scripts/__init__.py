"""Operational and build scripts.

A package rather than a bare directory so these are importable as
`scripts.<name>` — which is how they are run (`python -m scripts.<name>`)
and how their tests exercise them, rather than each test re-implementing a
path-munging import shim.
"""
