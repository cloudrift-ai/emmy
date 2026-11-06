"""Tests of how the statement module is built: its inner normalization steps, value numbering, body assembly.

They import the module's submodules and private names, which nothing else may. The tests beside this directory state
what goes into the module and what comes out, and hold for any implementation of it; these leave with this one.
"""
