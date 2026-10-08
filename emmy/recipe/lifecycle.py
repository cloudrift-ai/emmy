"""Recipe lifecycle tag validation."""

import re

MAINTAINED_TAG = "maintained"
BEST_EFFORT_TAG = "best-effort"
OBSOLETE_TAG = "obsolete"
ONBOARDING_TAG = "onboarding"
UNTESTED_TAG = "untested"
#: Not a lifecycle state: the last onboarding or verification attempt failed. Nightly work skips the recipe until an
#: explicit manual retry succeeds and removes the tag.
ONBOARDING_FAILED_TAG = "onboarding-failed"
#: Not a lifecycle state: the shipped priors do not yet reproduce the recipe's golden. The golden stays evidence and
#: training data; the prior reproduction gate skips it until a refit reproduces it and that refit removes the tag.
PRIOR_PENDING_TAG = "prior-pending"
#: Not a lifecycle state: a person decided the recipe's lifecycle. The discovery workflow leaves the recipe out — it is
#: never scored or reclassified, and does not count against the maintained set.
LIFECYCLE_LOCKED_TAG = "lifecycle-locked"

LIFECYCLE_TAGS = frozenset({MAINTAINED_TAG, BEST_EFFORT_TAG, OBSOLETE_TAG, ONBOARDING_TAG})
DISABLED_TAGS = frozenset({OBSOLETE_TAG, ONBOARDING_TAG})
_TAG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def validate_recipe_tags(value: object) -> tuple[str, ...]:
    """Return validated, unique recipe tags while preserving their order."""
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(tag, str) for tag in value):
        raise ValueError("recipe tags must be a list of strings")
    if any(not _TAG.fullmatch(tag) for tag in value):
        raise ValueError("recipe tags must use lowercase kebab-case")
    if len(value) != len(set(value)):
        raise ValueError("recipe tags must be unique")

    lifecycle = LIFECYCLE_TAGS.intersection(value)
    if len(lifecycle) > 1:
        raise ValueError(f"recipe must have at most one lifecycle tag, got {', '.join(sorted(lifecycle))}")
    if LIFECYCLE_LOCKED_TAG in value and not lifecycle:
        raise ValueError("recipe tag 'lifecycle-locked' needs a lifecycle tag to lock")
    if (ONBOARDING_TAG in value) != (UNTESTED_TAG in value):
        raise ValueError("recipe tags 'onboarding' and 'untested' must appear together")
    return tuple(value)


def recipe_lifecycle(config: dict) -> str | None:
    """Return the recipe lifecycle tag, or None for a legacy untagged recipe."""
    tags = validate_recipe_tags(config.get("tags"))
    return next((tag for tag in tags if tag in LIFECYCLE_TAGS), None)


def recipe_is_runnable(config: dict) -> bool:
    """Whether a recipe can be deployed, benchmarked, published, and bundled."""
    return recipe_lifecycle(config) not in DISABLED_TAGS
