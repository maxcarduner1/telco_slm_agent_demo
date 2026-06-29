# Databricks notebook source
# MAGIC %md
# MAGIC # AI Gateway V2 — Demo Notebook
# MAGIC
# MAGIC Covers the full CRUD lifecycle and every Quick Reference pattern from the API spec.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Set `DATABRICKS_HOST` and `DATABRICKS_TOKEN` as environment variables,
# MAGIC   or pass them explicitly to each function call.

# COMMAND ----------

# MAGIC %pip install -e .. openai typing_extensions -qU
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from smart_model_upgrades.ai_gateway import (
    # CRUD
    create_endpoint,
    get_endpoint,
    list_endpoints,
    update_endpoint,
    delete_endpoint,
    # Builders
    destination,
    rate_limit,
    fallback_config,
    tag,
    inference_table,
)

# COMMAND ----------

import os

# Gets current Databricks notebook’s Host and API token as a string, from inside notebook. Replace as needed.
os.environ["DATABRICKS_HOST"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
os.environ["DATABRICKS_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## List existing endpoints

# COMMAND ----------

result = list_endpoints()
for ep in result.get("endpoints", []):
    print(ep["name"], ep.get("status"))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Pattern 1: Create a single PPT endpoint

# COMMAND ----------

operation = create_endpoint(
    name="my-claude-endpoint",
    destinations=[
        destination(
            name="system.ai.databricks-claude-3-7-sonnet",
            destination_type="PAY_PER_TOKEN_FOUNDATION_MODEL",
            traffic_percentage=100,
        )
    ],
)
print(operation)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sample Endpoint Call

# COMMAND ----------

from openai import OpenAI
import os

# To get a DATABRICKS_TOKEN, click the "Generate Access Token" button or follow https://docs.databricks.com/en/dev-tools/auth/pat.html
DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')

client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url="https://6051921418418893.ai-gateway.staging.cloud.databricks.com/mlflow/v1"
)

chat_completion = client.chat.completions.create(
  messages=[
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hello! How can I assist you today?"},
    {"role": "user", "content": "What is Databricks?"},
  ],
  model="my-claude-endpoint",
  max_tokens=1024
)

print(chat_completion.choices[0].message.content)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Pattern 2: Create with traffic split + fallback

# COMMAND ----------

operation = create_endpoint(
    name="my-split-endpoint",
    destinations=[
        destination(
            name="system.ai.databricks-claude-3-7-sonnet",
            destination_type="PAY_PER_TOKEN_FOUNDATION_MODEL",
            traffic_percentage=70,
        ),
        destination(
            name="system.ai.databricks-gpt-5",
            destination_type="PAY_PER_TOKEN_FOUNDATION_MODEL",
            traffic_percentage=30,
        ),
    ],
    fallback=fallback_config(
        destinations=[
            destination(
                name="system.ai.databricks-gpt-5-mini",
                destination_type="PAY_PER_TOKEN_FOUNDATION_MODEL",
                # no traffic_percentage on fallback destinations
            )
        ],
        strategy="ROUND_ROBIN",
        max_attempts=2,
    ),
)
print(operation)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sample Endpoint Call

# COMMAND ----------

from openai import OpenAI
import os

# To get a DATABRICKS_TOKEN, click the "Generate Access Token" button or follow https://docs.databricks.com/en/dev-tools/auth/pat.html
DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')

client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url="https://6051921418418893.ai-gateway.staging.cloud.databricks.com/mlflow/v1"
)

chat_completion = client.chat.completions.create(
  messages=[
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hello! How can I assist you today?"},
    {"role": "user", "content": "What is Databricks?"},
  ],
  model="my-split-endpoint",
  max_tokens=1024
)

print(chat_completion.choices[0].message.content)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Get an endpoint

# COMMAND ----------

endpoint = get_endpoint("my-claude-endpoint")
print(endpoint)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Pattern 3: Add rate limits to an existing endpoint

# COMMAND ----------

operation = update_endpoint(
    name="my-claude-endpoint",
    rate_limits=[
        rate_limit(key="ENDPOINT", requests=500),
        rate_limit(key="USER", tokens=10000, principal="user@example.com"),
    ],
)
print(operation)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Pattern 4: Add tags

# COMMAND ----------

operation = update_endpoint(
    name="my-claude-endpoint",
    tags=[
        tag("team", "ml-platform"),
        tag("environment", "production"),
    ],
)
print(operation)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Pattern 5: Enable inference table logging

# COMMAND ----------

operation = update_endpoint(
    name="my-claude-endpoint",
    inference_table_config=inference_table(
        catalog_name="main",
        schema_name="mohamad_aboufoul",
        table_name_prefix="gateway_logs",
        enabled=True,
    ),
)
print(operation)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Pattern 6: Update multiple fields at once

# COMMAND ----------

operation = update_endpoint(
    name="my-claude-endpoint",
    rate_limits=[
        rate_limit(key="ENDPOINT", requests=1000),
    ],
    tags=[
        tag("env", "prod"),
    ],
)
print(operation)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Delete endpoints

# COMMAND ----------

delete_endpoint("my-claude-endpoint")
print("Deleted my-claude-endpoint.")

delete_endpoint("my-split-endpoint")
print("Deleted my-split-endpoint.")
