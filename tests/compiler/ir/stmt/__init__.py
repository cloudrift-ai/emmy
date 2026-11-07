"""Tests of the statement module that hold for any implementation of it: a body goes in as text and comes out as text."""

from textwrap import dedent


def listing(text: str) -> str:
    """A body's text as a test writes it in an indented string: the common indentation and the edge newlines removed."""
    return dedent(text).strip("\n")
