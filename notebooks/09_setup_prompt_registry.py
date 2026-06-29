# Databricks notebook source
# MAGIC %md
# MAGIC # 09 - Setup Prompt Registry
# MAGIC
# MAGIC Registers the TelcoGPT supervisor prompt under a configurable Unity Catalog
# MAGIC name. Re-running creates a new prompt version and repoints `@production`.

# COMMAND ----------

# MAGIC %pip install pyyaml -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
from pathlib import Path

import mlflow
import yaml
from mlflow.exceptions import RestException

mlflow.set_registry_uri("databricks-uc")

dbutils.widgets.text("catalog", "cmegdemos_catalog", "UC Catalog")
dbutils.widgets.text("schema", "network_analytics_enablement", "UC Schema")
dbutils.widgets.text("prompt_name", "telcogpt_supervisor", "Prompt short name")
dbutils.widgets.text("prompt_file", "prompts/supervisor.yaml", "Prompt YAML path")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
prompt_name = dbutils.widgets.get("prompt_name")
prompt_file = dbutils.widgets.get("prompt_file")

full_prompt_name = f"{catalog}.{schema}.{prompt_name}"
cwd = Path(os.getcwd())
repo_root = next(
    (
        path
        for path in (cwd, *cwd.parents)
        if (path / "prompts").exists() and (path / "notebooks").exists()
    ),
    cwd,
)
prompt_path = repo_root / prompt_file

if not prompt_path.exists():
    raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

with prompt_path.open() as f:
    prompt_cfg = yaml.safe_load(f)

template = prompt_cfg["template"]
description = prompt_cfg.get("description", "TelcoGPT supervisor prompt")

print(f"Registering prompt: {full_prompt_name}")
print(f"Template chars: {len(template)}")

# COMMAND ----------

try:
    pv = mlflow.genai.register_prompt(
        name=full_prompt_name,
        template=template,
        commit_message=f"Seed prompt: {description}",
    )
    mlflow.genai.set_prompt_alias(
        name=full_prompt_name,
        alias="production",
        version=pv.version,
    )

    print(f"Registered {full_prompt_name}@production = v{pv.version}")
    print(f"Prompt URI: prompts:/{full_prompt_name}@production")
except RestException as exc:
    if "FEATURE_DISABLED" not in str(exc):
        raise
    print(
        "WARNING: Prompt Registry is not enabled in this workspace. "
        "Skipping prompt registration; the app will use its local prompt fallback."
    )
