"""The serving API key of a protected deployment: the token every model request must carry."""

import argparse
import re

from emmy import config
from emmy.redact import register_secret

# Safe as one word in the compose file and in a shell command, and long enough to be redacted from logs.
_KEY = re.compile(r"[A-Za-z0-9._~-]{16,}")


def serving_api_key(text: str) -> str:
    """argparse type: a serving API key, registered so no log line shows it. The error never repeats the value."""
    if not _KEY.fullmatch(text):
        raise argparse.ArgumentTypeError("expected at least 16 characters from letters, digits and . _ ~ -")
    register_secret(text)
    return text


def add_serving_api_key_argument(parser) -> None:
    """Register ``--serving-api-key KEY`` on a deploy target; ``EMMY_SERVING_API_KEY`` is the fallback."""
    parser.add_argument(
        "--serving-api-key",
        type=serving_api_key,  # argparse runs the type on a string default too, so the fallback is checked alike
        default=config.serving_api_key() or None,
        metavar="KEY",
        help="Deploy a protected deployment: every model request must send this key as 'Authorization: Bearer KEY' "
        "(default: $EMMY_SERVING_API_KEY; unset deploys an open endpoint). vLLM and SGLang only",
    )
