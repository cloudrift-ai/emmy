"""Tests of how the statement module is built: the normal form every stored kernel settles to, the body's own
summaries, the memo.

They read the module's private names, which nothing else may. The tests beside this directory state what goes into
the module and what comes out, and hold for any implementation of it; these leave with this one. The inner steps of
normalization are tested beside their Rust, in ``crates/emmy-ir``.
"""
