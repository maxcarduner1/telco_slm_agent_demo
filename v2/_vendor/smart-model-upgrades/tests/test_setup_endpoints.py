"""Tests for smu.setup_endpoints."""
import pytest

from smart_model_upgrades import setup_endpoints


def test_creates_when_missing(mocker):
    mocker.patch(
        "smart_model_upgrades.optimization.gw.get_endpoint",
        side_effect=Exception("404"),
    )
    create = mocker.patch("smart_model_upgrades.optimization.gw.create_endpoint")
    statuses = setup_endpoints({"ep1": "databricks-claude-sonnet-4"})
    assert statuses == {"ep1": "created"}
    create.assert_called_once()
    name = create.call_args.kwargs["name"]
    dest = create.call_args.kwargs["destinations"][0]["name"]
    assert name == "ep1"
    assert dest == "system.ai.databricks-claude-sonnet-4"


def test_skips_when_destination_matches(mocker):
    mocker.patch(
        "smart_model_upgrades.optimization.gw.get_endpoint",
        return_value={"config": {"destinations": [{"name": "system.ai.databricks-claude-sonnet-4"}]}},
    )
    create = mocker.patch("smart_model_upgrades.optimization.gw.create_endpoint")
    update = mocker.patch("smart_model_upgrades.optimization.gw.update_endpoint")
    statuses = setup_endpoints({"ep1": "databricks-claude-sonnet-4"})
    assert statuses == {"ep1": "exists:system.ai.databricks-claude-sonnet-4"}
    create.assert_not_called()
    update.assert_not_called()


def test_updates_when_destination_drifts(mocker):
    mocker.patch(
        "smart_model_upgrades.optimization.gw.get_endpoint",
        return_value={"config": {"destinations": [{"name": "system.ai.databricks-old"}]}},
    )
    update = mocker.patch("smart_model_upgrades.optimization.gw.update_endpoint")
    statuses = setup_endpoints({"ep1": "databricks-new"})
    assert statuses == {"ep1": "updated:system.ai.databricks-old->system.ai.databricks-new"}
    update.assert_called_once()


def test_update_failure_propagates_does_not_fall_through_to_create(mocker):
    """The narrowed try block: an update_endpoint failure must surface, not
    silently fall through to create_endpoint (which would then complain
    'already exists' and mask the real error)."""
    mocker.patch(
        "smart_model_upgrades.optimization.gw.get_endpoint",
        return_value={"config": {"destinations": [{"name": "system.ai.databricks-old"}]}},
    )
    mocker.patch(
        "smart_model_upgrades.optimization.gw.update_endpoint",
        side_effect=PermissionError("not allowed"),
    )
    create = mocker.patch("smart_model_upgrades.optimization.gw.create_endpoint")
    with pytest.raises(PermissionError, match="not allowed"):
        setup_endpoints({"ep1": "databricks-new"})
    create.assert_not_called()


def test_tags_propagated_to_create(mocker):
    mocker.patch(
        "smart_model_upgrades.optimization.gw.get_endpoint",
        side_effect=Exception("404"),
    )
    create = mocker.patch("smart_model_upgrades.optimization.gw.create_endpoint")
    setup_endpoints(
        {"ep1": "databricks-claude-sonnet-4"},
        agent_tag="myagent",
        extra_tags=[("env", "dev")],
    )
    tags = create.call_args.kwargs["tags"]
    keys = {t["key"]: t["value"] for t in tags}
    assert keys.get("managed_by") == "smart-model-upgrades"
    assert keys.get("agent") == "myagent"
    assert keys.get("env") == "dev"


def test_multiple_endpoints_processed_independently(mocker):
    """One call should process several endpoints, mixing created/exists/updated."""
    def fake_get(name):
        if name == "ep_match":
            return {"config": {"destinations": [{"name": "system.ai.databricks-match"}]}}
        if name == "ep_drift":
            return {"config": {"destinations": [{"name": "system.ai.databricks-stale"}]}}
        raise Exception("404")

    mocker.patch("smart_model_upgrades.optimization.gw.get_endpoint", side_effect=fake_get)
    create = mocker.patch("smart_model_upgrades.optimization.gw.create_endpoint")
    update = mocker.patch("smart_model_upgrades.optimization.gw.update_endpoint")

    statuses = setup_endpoints({
        "ep_match": "databricks-match",
        "ep_drift": "databricks-fresh",
        "ep_new": "databricks-claude-sonnet-4",
    })
    assert statuses["ep_match"] == "exists:system.ai.databricks-match"
    assert statuses["ep_drift"] == "updated:system.ai.databricks-stale->system.ai.databricks-fresh"
    assert statuses["ep_new"] == "created"
    create.assert_called_once()
    update.assert_called_once()
