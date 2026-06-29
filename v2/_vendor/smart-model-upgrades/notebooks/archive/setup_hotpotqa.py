# Databricks notebook source
# MAGIC %md
# MAGIC # HotpotQA Agent Setup -- Prompt + Gateway Endpoint
# MAGIC
# MAGIC Registers the HotpotQA QA prompt in the MLflow Prompt Registry and
# MAGIC creates a single production AI Gateway V2 endpoint
# MAGIC (`hotpotqa-smart-endpoint`) seeded with `databricks-gpt-5-4-nano`.
# MAGIC
# MAGIC Idempotent -- safe to re-run. Existing prompts get a new version;
# MAGIC existing endpoint is left as-is.

# COMMAND ----------

# MAGIC %pip install -e .. -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import yaml

import mlflow
from smart_model_upgrades import ai_gateway as gw
from smart_model_upgrades import register_prompts_from_config

os.environ["DATABRICKS_HOST"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
os.environ["DATABRICKS_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config
# MAGIC
# MAGIC All prompt names, endpoint names, and initial models come from the agent
# MAGIC config YAML -- this notebook only owns the prompt templates.

# COMMAND ----------

AGENT_CONFIG_PATH = os.path.join(os.getcwd(), "..", "configs", "hotpotqa_config.yaml")

with open(AGENT_CONFIG_PATH) as f:
    agent_cfg = yaml.safe_load(f)

# Baseline prompt from the MLflow blog:
# https://mlflow.org/blog/mlflow-prompt-optimization
TEMPLATES = {
    "endpoint": """You are a question answering assistant. Answer questions based ONLY on the provided context.

IMPORTANT INSTRUCTIONS:
- For yes/no questions, answer ONLY "yes" or "no"
- Do NOT include phrases like "based on the context" or "according to the documents"

Context:
{{ context }}

Question: {{ question }}

Answer:""",
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register Prompt

# COMMAND ----------

register_prompts_from_config(AGENT_CONFIG_PATH, TEMPLATES)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Production Gateway Endpoint

# COMMAND ----------

gw_section = agent_cfg["gateway_endpoints"]
for comp, cfg in gw_section.items():
    ep_name = cfg["smart_endpoint"]
    initial_model = cfg["initial_model"]
    try:
        existing = gw.get_endpoint(ep_name)
        dests = existing.get("config", {}).get("destinations", [])
        current = dests[0]["name"] if dests else "unknown"
        print(f"{ep_name}: already exists ({current})")
    except Exception:
        gw.create_endpoint(
            name=ep_name,
            destinations=[gw.destination(
                f"system.ai.{initial_model}",
                "PAY_PER_TOKEN_FOUNDATION_MODEL",
                100,
            )],
            task_type="llm/v1/chat",
            tags=[
                gw.tag("managed_by", "smart-model-upgrades"),
                gw.tag("agent", "hotpotqa"),
            ],
        )
        print(f"{ep_name}: created with system.ai.{initial_model}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify

# COMMAND ----------

for comp, full_name in agent_cfg["prompt_registry"].items():
    pv = mlflow.genai.load_prompt(f"prompts:/{full_name}@production")
    print(f"Prompt: {full_name}@production (v{pv.version}, {len(pv.template)} chars)")

for comp, cfg in gw_section.items():
    ep_name = cfg["smart_endpoint"]
    ep = gw.get_endpoint(ep_name)
    dests = ep.get("config", {}).get("destinations", [])
    print(f"Endpoint: {ep_name} -> {dests[0]['name'] if dests else 'none'}")
