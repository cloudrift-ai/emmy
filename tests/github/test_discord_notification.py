import importlib.util
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError

MODULE_PATH = Path(__file__).parents[2] / ".github" / "scripts" / "discord_notification.py"
SPEC = importlib.util.spec_from_file_location("discord_notification", MODULE_PATH)
discord_notification = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(discord_notification)

BASE_ENVIRONMENT = {
    "GITHUB_REPOSITORY": "cloudrift-ai/emmy",
    "GITHUB_RUN_ID": "12345",
    "GITHUB_SERVER_URL": "https://github.com",
}


def test_onboarding_success_payload_has_target_pr_and_no_mentions():
    environment = {
        **BASE_ENVIRONMENT,
        "WORKFLOW_KIND": "onboard",
        "WORKFLOW_RESULT": "success",
        "SELECTED": "true",
        "MODEL_ID": "Qwen/Qwen3-Embedding-8B",
        "TARGET_GPU": "NVIDIA GeForce RTX 4090",
        "TARGET_GPU_COUNT": "1",
        "ONBOARD_MODE": "onboarding",
        "DEPLOYMENT_SUMMARY": "vLLM 0.22.1, 32K context, concurrency 8",
        "PERFORMANCE_SUMMARY": "100 requests, 2,400 output tok/s, p50 TTFT 42 ms, 0 failures",
        "PR_NUMBER": "487",
    }

    payload = discord_notification.build_payload(environment, now=datetime(2026, 8, 16, tzinfo=UTC))
    embed = payload["embeds"][0]

    assert payload["allowed_mentions"] == {"parse": []}
    assert embed["title"] == "Model onboarding completed"
    assert embed["url"] == "https://github.com/cloudrift-ai/emmy/actions/runs/12345"
    assert embed["timestamp"] == "2026-08-16T00:00:00+00:00"
    assert embed["fields"] == [
        {"name": "Model", "value": "`Qwen/Qwen3-Embedding-8B`", "inline": False},
        {"name": "Target", "value": "`NVIDIA GeForce RTX 4090 x1`", "inline": False},
        {"name": "Deployment", "value": "vLLM 0.22.1, 32K context, concurrency 8", "inline": False},
        {"name": "Performance", "value": "100 requests, 2,400 output tok/s, p50 TTFT 42 ms, 0 failures", "inline": False},
        {"name": "Mode", "value": "`onboarding`", "inline": True},
        {
            "name": "Rolling PR",
            "value": "[#487](https://github.com/cloudrift-ai/emmy/pull/487)",
            "inline": True,
        },
    ]


NIGHTLY_ENVIRONMENT = {
    **BASE_ENVIRONMENT,
    "WORKFLOW_KIND": "nightly",
    "DURATIONS_RESULT": "success",
    "DURATIONS_UPDATED": "true",
    "PRIOR_RESULT": "success",
    "SCHEDULE_PRIOR": "candidate rejected: median rank rose in 1 of 3 cells",
    "PLACEMENT_PRIOR": "Updated the weights on main; candidate qualifies: 2 of 3 cells improved by at least 5%, none regressed",
    "DISCOVER_RESULT": "success",
}


def test_nightly_success_payload_reports_every_job_and_groups_modified_models():
    environment = {
        **NIGHTLY_ENVIRONMENT,
        "MODIFIED_MODELS": json.dumps(
            [
                {"model_id": "org/maintained", "lifecycle": "maintained", "heat": 70},
                {"model_id": "org/best-effort", "lifecycle": "best-effort", "heat": 40},
                {"model_id": "org/new", "lifecycle": "onboarding", "heat": 95},
            ]
        ),
    }

    payload = discord_notification.build_payload(environment)
    embed = payload["embeds"][0]

    assert embed["title"] == "Nightly refresh completed"
    assert embed["color"] == discord_notification.SUCCESS_COLOR
    assert embed["fields"] == [
        {"name": "CPU test durations", "value": "Updated on main.", "inline": False},
        {"name": "Schedule prior", "value": "candidate rejected: median rank rose in 1 of 3 cells", "inline": False},
        {
            "name": "Placement prior",
            "value": "Updated the weights on main; candidate qualifies: 2 of 3 cells improved by at least 5%, none regressed",
            "inline": False,
        },
        {"name": "Maintained", "value": "• `org/maintained` · heat **70**", "inline": False},
        {"name": "Best effort", "value": "• `org/best-effort` · heat **40**", "inline": False},
        {"name": "Onboarding", "value": "• `org/new` · heat **95**", "inline": False},
    ]


def test_nightly_payload_reports_no_recipe_changes_and_no_duration_push():
    environment = {**NIGHTLY_ENVIRONMENT, "DURATIONS_UPDATED": "", "MODIFIED_MODELS": "[]"}

    payload = discord_notification.build_payload(environment)
    fields = payload["embeds"][0]["fields"]

    assert fields[0] == {"name": "CPU test durations", "value": "No change on main.", "inline": False}
    assert {"name": "Modified models", "value": "None; the lifecycle review produced no recipe changes.", "inline": False} in fields


def test_nightly_failure_payload_names_the_failed_job_and_keeps_the_other_results():
    environment = {
        **NIGHTLY_ENVIRONMENT,
        "PRIOR_RESULT": "failure",
        "PLACEMENT_PRIOR": "",
        "DISCOVER_RESULT": "failure",
    }

    payload = discord_notification.build_payload(environment)
    embed = payload["embeds"][0]

    assert payload["allowed_mentions"] == {"parse": []}
    assert embed["title"] == "Nightly refresh failed"
    assert embed["color"] == discord_notification.FAILURE_COLOR
    assert embed["fields"] == [
        {"name": "CPU test durations", "value": "Updated on main.", "inline": False},
        {"name": "Schedule prior", "value": "candidate rejected: median rank rose in 1 of 3 cells", "inline": False},
        {"name": "Placement prior", "value": "Failed; open the run for the failing step and logs.", "inline": False},
        {"name": "Modified models", "value": "Failed; open the run for the failing step and logs.", "inline": False},
    ]


def test_nightly_cancellation_is_not_a_failure():
    environment = {**NIGHTLY_ENVIRONMENT, "DISCOVER_RESULT": "cancelled", "MODIFIED_MODELS": ""}

    embed = discord_notification.build_payload(environment)["embeds"][0]

    assert embed["title"] == "Nightly refresh cancelled"
    assert embed["color"] == discord_notification.CANCELLED_COLOR
    assert embed["fields"][-1] == {"name": "Modified models", "value": "Cancelled.", "inline": False}


def test_no_eligible_onboarding_is_a_neutral_summary():
    environment = {
        **BASE_ENVIRONMENT,
        "WORKFLOW_KIND": "onboard",
        "WORKFLOW_RESULT": "success",
        "SELECTED": "false",
    }

    payload = discord_notification.build_payload(environment)
    embed = payload["embeds"][0]

    assert embed["title"] == "No eligible model deployment"
    assert embed["color"] == discord_notification.NEUTRAL_COLOR


def test_unresolved_onboarding_regression_is_prominent_and_non_pinging():
    environment = {
        **BASE_ENVIRONMENT,
        "WORKFLOW_KIND": "onboard",
        "WORKFLOW_RESULT": "failure",
        "SELECTED": "true",
        "MODEL_ID": "Qwen/Qwen3-Embedding-8B",
        "TARGET_GPU": "NVIDIA GeForce RTX 4090",
        "TARGET_GPU_COUNT": "1",
        "ONBOARD_MODE": "verification",
        "FAILURE_KIND": "regression",
        "FAILURE_SUMMARY": "serving: the official image replacement still fails its health check",
    }

    payload = discord_notification.build_payload(environment)
    embed = payload["embeds"][0]

    assert payload["content"] == "🚨 **Model regression needs attention**"
    assert payload["allowed_mentions"] == {"parse": []}
    assert embed["title"] == "Model regression needs attention"
    assert embed["color"] == discord_notification.FAILURE_COLOR
    assert {
        "name": "Regression",
        "value": "serving: the official image replacement still fails its health check",
        "inline": False,
    } in embed["fields"]


def test_delivery_retries_and_requests_a_confirmed_discord_response():
    calls = []
    sleeps = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b"{}"

    def opener(request, *, timeout):
        calls.append((request, timeout))
        if len(calls) < 3:
            raise URLError("temporary failure")
        return Response()

    payload = {"allowed_mentions": {"parse": []}}

    delivered = discord_notification.send_notification(
        "https://discord.com/api/webhooks/id/token?thread_id=7",
        payload,
        opener=opener,
        sleeper=sleeps.append,
    )

    assert delivered is True
    assert len(calls) == 3
    assert calls[-1][0].full_url.endswith("?thread_id=7&wait=true")
    assert calls[-1][1] == 15
    assert json.loads(calls[-1][0].data) == payload
    assert sleeps == [1, 2]


def test_missing_webhook_is_non_fatal(caplog):
    with caplog.at_level(logging.WARNING, logger="discord_notification"):
        result = discord_notification.main({})

    assert result == 0
    assert "DISCORD_EMMY_ROBOTS_WEBHOOK_URL is not configured" in caplog.text
