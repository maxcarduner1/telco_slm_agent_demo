# Databricks notebook source
# MAGIC %md
# MAGIC # ChatDatabricks AI Gateway V2 - End-to-End Test
# MAGIC
# MAGIC Validates the `use_ai_gateway` / `use_ai_gateway_native_api` params added to
# MAGIC `ChatDatabricks` in branch `max/chat-databricks-ai-gateway-v2` of
# MAGIC `maxmarcussen-db/databricks-ai-bridge`.
# MAGIC
# MAGIC Installs both `databricks-langchain` and `databricks-openai` from the PR branch
# MAGIC (the langchain change depends on gateway support that lives in the openai package).

# COMMAND ----------

# MAGIC %pip install "databricks-langchain @ git+https://github.com/maxmarcussen-db/databricks-ai-bridge.git@max/chat-databricks-ai-gateway-v2#subdirectory=integrations/langchain" "databricks-openai @ git+https://github.com/maxmarcussen-db/databricks-ai-bridge.git@max/chat-databricks-ai-gateway-v2#subdirectory=integrations/openai" -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import inspect
from databricks_langchain import ChatDatabricks
from databricks_openai import DatabricksOpenAI

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Sanity check: params are present

# COMMAND ----------

params = list(inspect.signature(DatabricksOpenAI.__init__).parameters.keys())
print(f"DatabricksOpenAI init params: {params}")
assert "use_ai_gateway" in params, "DatabricksOpenAI missing use_ai_gateway"
assert "use_ai_gateway_native_api" in params, "DatabricksOpenAI missing use_ai_gateway_native_api"

fields = ChatDatabricks.model_fields
print(f"ChatDatabricks has use_ai_gateway: {'use_ai_gateway' in fields}")
print(f"ChatDatabricks has use_ai_gateway_native_api: {'use_ai_gateway_native_api' in fields}")
assert "use_ai_gateway" in fields, "ChatDatabricks missing use_ai_gateway field"
assert "use_ai_gateway_native_api" in fields, "ChatDatabricks missing use_ai_gateway_native_api field"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Pick a V2 gateway endpoint
# MAGIC
# MAGIC Lists gateway endpoints via the V2 REST API. Pick one that's `READY` with a chat task type.

# COMMAND ----------

import requests
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
host = w.config.host.rstrip("/")
token = w.config.authenticate()["Authorization"].replace("Bearer ", "")

resp = requests.get(
    f"{host}/api/ai-gateway/v2/endpoints",
    headers={"Authorization": f"Bearer {token}"},
    params={"page_size": 50},
)
resp.raise_for_status()
endpoints = resp.json().get("endpoints", [])
for e in endpoints:
    print(f"{e['name']:55s} {e.get('task_type','-'):25s} {e.get('status','-')}")

# COMMAND ----------

# Set this to the name of a V2 gateway endpoint from the list above
GATEWAY_ENDPOINT = "smart-model-upgrades-supervisor-exp"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Test: ChatDatabricks with use_ai_gateway=True (MLflow API route)

# COMMAND ----------

llm = ChatDatabricks(model=GATEWAY_ENDPOINT, use_ai_gateway=True)
print(f"base_url: {llm.client.base_url}")
assert "/ai-gateway/mlflow/v1" in str(llm.client.base_url), "Wrong base URL for use_ai_gateway=True"

response = llm.invoke("Say hello in exactly three words.")
print(f"Response: {response.content}")
assert response.content, "Empty response"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Test: ChatDatabricks with use_ai_gateway_native_api=True (OpenAI API route)
# MAGIC
# MAGIC This only invokes successfully if the gateway endpoint's destination supports
# MAGIC the `openai/v1` API. If the endpoint only supports `mlflow/v1`, we expect a
# MAGIC 400 from the gateway — that still proves the base_url routing works.

# COMMAND ----------

from openai import BadRequestError

llm_native = ChatDatabricks(model=GATEWAY_ENDPOINT, use_ai_gateway_native_api=True)
print(f"base_url: {llm_native.client.base_url}")
assert "/ai-gateway/openai/v1" in str(llm_native.client.base_url), "Wrong base URL for use_ai_gateway_native_api=True"

try:
    response_native = llm_native.invoke("Say hello in exactly three words.")
    print(f"Response: {response_native.content}")
    assert response_native.content, "Empty response"
except BadRequestError as e:
    if "not supported by AI Gateway endpoint" in str(e):
        print(f"Expected 400 from gateway: endpoint doesn't expose openai/v1 (routing works).")
    else:
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Test: default (no gateway flags) still routes to serving endpoints

# COMMAND ----------

llm_default = ChatDatabricks(model="databricks-claude-sonnet-4-6")
print(f"base_url: {llm_default.client.base_url}")
assert "/serving-endpoints" in str(llm_default.client.base_url), "Wrong base URL for default path"
assert "/ai-gateway" not in str(llm_default.client.base_url), "Default path should not hit ai-gateway"

response_default = llm_default.invoke("Say hello in exactly three words.")
print(f"Response: {response_default.content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Streaming through the gateway

# COMMAND ----------

print("Streaming chunks:")
for chunk in llm.stream("Count from one to five."):
    print(chunk.content, end="", flush=True)
print()

# COMMAND ----------

print("All tests passed.")
