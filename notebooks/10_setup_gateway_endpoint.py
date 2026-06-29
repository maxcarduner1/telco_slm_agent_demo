# Databricks notebook source
# MAGIC %md
# MAGIC # 10 - Setup AI Gateway Endpoint
# MAGIC
# MAGIC Creates the TelcoGPT supervisor AI Gateway endpoint used by the app.

# COMMAND ----------

import os

import requests

dbutils.widgets.text("supervisor_gateway", "telcogpt-supervisor", "Supervisor gateway endpoint")
dbutils.widgets.text("initial_model", "databricks-claude-sonnet-4", "Initial production model")

supervisor_gateway = dbutils.widgets.get("supervisor_gateway")
initial_model = dbutils.widgets.get("initial_model")

if not supervisor_gateway.startswith("telcogpt-"):
    raise ValueError("Gateway endpoint names must start with 'telcogpt-'")

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
            "tags": [{"key": "project", "value": "telcogpt"}],
            "usage_tracking": {"enabled": True},
        },
    }
    resp = requests.post(base_url, headers=headers, json=body)
    _raise_for_status(resp)
    return resp.json()


print("Gateway endpoint:")
print(f"  {supervisor_gateway} -> {initial_model}")

existing = _get_endpoint(supervisor_gateway)
if existing:
    print(f"Gateway already exists: {supervisor_gateway}")
    print(existing)
else:
    created = _create_endpoint(supervisor_gateway, initial_model)
    print(f"Created gateway endpoint: {supervisor_gateway}")
    print(created)

print("Gateway setup complete.")
print(f"Use this endpoint in app config: LLM_ENDPOINT={supervisor_gateway}")
