# Databricks notebook source
# MAGIC %md
# MAGIC # V2.0 - Setup AI Gateway Endpoints
# MAGIC
# MAGIC Creates V2-only AI Gateway endpoint(s) used by the app and optimizer.
# MAGIC This notebook must not create, modify, or repoint V1 endpoints.

# COMMAND ----------

import os
import requests

dbutils.widgets.text("supervisor_gateway", "telcogpt-v2-supervisor", "V2 supervisor gateway endpoint")
dbutils.widgets.text("initial_model", "databricks-claude-sonnet-4", "Initial production model")
dbutils.widgets.text("candidate_model", "databricks-claude-haiku-4-5", "Conservative alternate candidate")

supervisor_gateway = dbutils.widgets.get("supervisor_gateway")
initial_model = dbutils.widgets.get("initial_model")
candidate_model = dbutils.widgets.get("candidate_model")

if not supervisor_gateway.startswith("telcogpt-v2-"):
    raise ValueError("V2 gateway endpoint names must start with 'telcogpt-v2-'")

os.environ["DATABRICKS_HOST"] = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
)
os.environ["DATABRICKS_TOKEN"] = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
)

host = os.environ["DATABRICKS_HOST"].rstrip("/")
token = os.environ["DATABRICKS_TOKEN"]
base_url = f"{host}/api/ai-gateway/v2/endpoints"
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _destination(model_name: str, traffic_percentage: int = 100) -> dict:
    return {
        "name": f"system.ai.{model_name}",
        "type": "PAY_PER_TOKEN_FOUNDATION_MODEL",
        "traffic_percentage": traffic_percentage,
    }


def _raise_for_status(resp):
    if not resp.ok:
        raise RuntimeError(f"{resp.status_code} {resp.reason}: {resp.text}")


def _get_endpoint(name: str) -> dict | None:
    resp = requests.get(f"{base_url}/{name}", headers=headers)
    if resp.status_code == 404:
        return None
    _raise_for_status(resp)
    return resp.json()


def _create_endpoint(name: str, model_name: str) -> dict:
    body = {
        "name": name,
        "task_type": "llm/v1/chat",
        "config": {
            "destinations": [_destination(model_name)],
            "routing_strategy": "REQUEST_BASED_TRAFFIC_SPLIT",
            "tags": [
                {"key": "project", "value": "telcogpt-v2"},
                {"key": "mode", "value": "evaluate-only"},
            ],
            "usage_tracking": {"enabled": True},
        },
    }
    resp = requests.post(base_url, headers=headers, json=body)
    _raise_for_status(resp)
    return resp.json()

print("Gateway endpoint:")
print(f"  {supervisor_gateway} -> {initial_model}")
print("Candidate model pool for optimizer:")
print(f"  - {initial_model}")
print(f"  - {candidate_model}")

existing = _get_endpoint(supervisor_gateway)
if existing:
    print(f"Gateway already exists: {supervisor_gateway}")
    print(existing)
else:
    created = _create_endpoint(supervisor_gateway, initial_model)
    print(f"Created gateway endpoint: {supervisor_gateway}")
    print(created)

# COMMAND ----------

print("V2 gateway setup complete.")
print("Use this endpoint in V2 app/eval config:")
print(f"  LLM_ENDPOINT={supervisor_gateway}")
