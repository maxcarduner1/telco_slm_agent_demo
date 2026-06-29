# Databricks notebook source
# MAGIC %md
# MAGIC # V2.0 - Setup Prompt Registry
# MAGIC
# MAGIC Registers the TelcoGPT V2 supervisor prompt under an isolated V2 name.
# MAGIC This is idempotent in the sense that re-running creates a new prompt version
# MAGIC and repoints the `@production` alias for the V2 prompt only.

# COMMAND ----------

# MAGIC %pip install pyyaml -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
from pathlib import Path

import mlflow
import yaml

dbutils.widgets.text("catalog", "cmegdemos_catalog", "UC Catalog")
dbutils.widgets.text("schema", "network_analytics_enablement", "UC Schema")
dbutils.widgets.text("prompt_name", "telcogpt_v2_supervisor", "Prompt short name")
dbutils.widgets.text("prompt_file", "v2/prompts/supervisor.yaml", "Prompt YAML path")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
prompt_name = dbutils.widgets.get("prompt_name")
prompt_file = dbutils.widgets.get("prompt_file")

full_prompt_name = f"{catalog}.{schema}.{prompt_name}"
cwd = Path(os.getcwd())
repo_root = cwd if (cwd / "v2").exists() else cwd.parents[1]
prompt_path = repo_root / prompt_file

if not prompt_path.exists():
    raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

with prompt_path.open() as f:
    prompt_cfg = yaml.safe_load(f)

template = prompt_cfg["template"]
description = prompt_cfg.get("description", "TelcoGPT V2 supervisor prompt")

print(f"Registering prompt: {full_prompt_name}")
print(f"Template chars: {len(template)}")

# COMMAND ----------

pv = mlflow.genai.register_prompt(
    name=full_prompt_name,
    template=template,
    commit_message=f"V2.0 seed prompt: {description}",
)
mlflow.genai.set_prompt_alias(
    name=full_prompt_name,
    alias="production",
    version=pv.version,
)

print(f"Registered {full_prompt_name}@production = v{pv.version}")
print(f"Prompt URI: prompts:/{full_prompt_name}@production")
