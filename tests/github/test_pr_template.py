"""The tracked pull-request template holds instructions, never a PR's own body."""

import re
from pathlib import Path

TEMPLATE = Path(__file__).parents[2] / ".github" / "PULL_REQUEST_TEMPLATE.md"


def test_template_is_only_the_abstract_heading_and_the_rule_outside_its_instruction_comments():
    outside_comments = re.sub(r"<!--.*?-->", "", TEMPLATE.read_text(), flags=re.DOTALL)

    assert outside_comments.split() == ["##", "Abstract", "---"]
