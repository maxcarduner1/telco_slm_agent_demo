# Databricks notebook source
# MAGIC %md
# MAGIC # V2.0 - Run Smart Model Upgrade (Evaluate Only)
# MAGIC
# MAGIC Runs prompt/model optimization for the V2 supervisor. This notebook is
# MAGIC intentionally evaluate-only and does not call `promote_to_prod`.

# COMMAND ----------

# MAGIC %pip install pyyaml -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys
from pathlib import Path

import mlflow
import yaml
from mlflow.genai.scorers import Correctness

dbutils.widgets.text("catalog", "cmegdemos_catalog", "UC Catalog")
dbutils.widgets.text("schema", "network_analytics_enablement", "UC Schema")
dbutils.widgets.text("warehouse_id", "", "SQL Warehouse ID")
dbutils.widgets.text("supervisor_gateway", "telcogpt-v2-supervisor", "V2 supervisor gateway endpoint")
dbutils.widgets.text("candidate_model", "databricks-claude-haiku-4-5", "Conservative alternate candidate")
dbutils.widgets.text("eval_set", "v2/eval_sets/telcogpt_supervisor_eval.yaml", "Eval YAML path")
dbutils.widgets.text("max_metric_calls", "30", "Max metric calls")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
warehouse_id = dbutils.widgets.get("warehouse_id")
supervisor_gateway = dbutils.widgets.get("supervisor_gateway")
candidate_model = dbutils.widgets.get("candidate_model")
eval_set_path = dbutils.widgets.get("eval_set")
max_metric_calls = int(dbutils.widgets.get("max_metric_calls"))

if not supervisor_gateway.startswith("telcogpt-v2-"):
    raise ValueError("V2 gateway endpoint names must start with 'telcogpt-v2-'")
if not warehouse_id:
    raise ValueError("warehouse_id widget is required")

cwd = Path(os.getcwd())
repo_root = cwd if (cwd / "v2").exists() else cwd.parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "v2" / "_vendor" / "smart-model-upgrades"))

import smart_model_upgrades as smu

os.environ["DATABRICKS_HOST"] = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
)
os.environ["DATABRICKS_TOKEN"] = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
)
os.environ["UC_CATALOG"] = catalog
os.environ["UC_SCHEMA"] = schema
os.environ["DATABRICKS_WAREHOUSE_ID"] = warehouse_id
os.environ["LLM_ENDPOINT"] = supervisor_gateway
os.environ["SUPERVISOR_PROMPT_URI"] = f"prompts:/{catalog}.{schema}.telcogpt_v2_supervisor@production"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "/Shared/telcogpt-v2-smart-model-upgrades"

from v2.predict import predict

# COMMAND ----------

with (repo_root / eval_set_path).open() as f:
    rows = yaml.safe_load(f)

records = []
for row in rows:
    records.append(
        {
            "inputs": row["inputs"],
            "expectations": {
                "expected_response": row["expectations"]["expected_response"],
                "guidelines": row["expectations"].get("guidelines", []),
            },
        }
    )

split = max(1, int(len(records) * 0.6))
train_data = records[:split]
val_data = records[split:] or records[-1:]

prompt_uri = os.environ["SUPERVISOR_PROMPT_URI"]
gateway_endpoints = {
    supervisor_gateway: [
        "databricks-claude-sonnet-4",
        candidate_model,
    ]
}

print(f"Prompt URI: {prompt_uri}")
print(f"Gateway endpoint: {supervisor_gateway}")
print(f"Train rows: {len(train_data)}, validation rows: {len(val_data)}")
print("Promotion: disabled for V2.0")

# COMMAND ----------

mlflow.set_experiment("/Shared/telcogpt-v2-smart-model-upgrades")

scorers = [Correctness(model="databricks:/databricks-gpt-5-4-nano")]

with mlflow.start_run(run_name="telcogpt-v2-evaluate-only") as run:
    mlflow.log_params(
        {
            "mode": "evaluate_only",
            "prompt_uri": prompt_uri,
            "supervisor_gateway": supervisor_gateway,
            "candidate_model": candidate_model,
            "max_metric_calls": max_metric_calls,
            "train_rows": len(train_data),
            "val_rows": len(val_data),
        }
    )

    result = smu.optimize_prompts_and_models(
        predict_fn=predict,
        train_data=train_data,
        val_data=val_data,
        prompt_uris=[prompt_uri],
        gateway_endpoints=gateway_endpoints,
        scorers=scorers,
        max_metric_calls=max_metric_calls,
    )

    mlflow.log_metric("baseline_score", result.baseline_score)
    mlflow.log_metric("best_score", result.best_score)
    mlflow.log_metric("score_delta", result.best_score - result.baseline_score)

print(
    f"Evaluate-only result: baseline={result.baseline_score:.3f}, "
    f"best={result.best_score:.3f}, delta={result.best_score - result.baseline_score:+.3f}"
)
print("V2.0 does not promote. Review this run before enabling V2.1 promotion.")
