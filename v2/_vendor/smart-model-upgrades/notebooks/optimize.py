# Databricks notebook source
# MAGIC %md
# MAGIC # Optimize -- Joint Model + Prompt via AI Gateway
# MAGIC
# MAGIC Pick an agent from the widget, then run all cells. To plug in your own agent,
# MAGIC drop it under `examples/<name>/` (see `examples/minimal_agent.py`) with an
# MAGIC `agent_config.yaml` that lists `candidate_models` per gateway endpoint.
# MAGIC
# MAGIC The library:
# MAGIC - creates `<endpoint>-exp` clones for the optimization loop
# MAGIC - scores the seed candidate, runs GEPA, scores the winner
# MAGIC - cleans up the exp endpoints when done
# MAGIC - registers any rewritten prompts and updates prod endpoints when you call `promote_to_prod`

# COMMAND ----------

# MAGIC %pip install -e '..[demos]' -qU

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import importlib
import os
import sys

import mlflow
import yaml

import smart_model_upgrades as smu

# Make the repo root importable so `from examples.<name>.agent import predict` works.
sys.path.insert(0, os.path.normpath(os.path.join(os.getcwd(), "..")))

os.environ["DATABRICKS_HOST"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
os.environ["DATABRICKS_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"

EXAMPLES_DIR = os.path.normpath(os.path.join(os.getcwd(), "..", "examples"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pick an agent
# MAGIC
# MAGIC Every entry under `examples/` with an `agent_config.yaml` is selectable.
# MAGIC Prompt URIs, endpoint names, and the candidate model pool per endpoint all
# MAGIC come from that config (`gateway_endpoints.<comp>.candidate_models`).

# COMMAND ----------

AGENT_CHOICES = sorted(
    name for name in os.listdir(EXAMPLES_DIR)
    if os.path.isfile(os.path.join(EXAMPLES_DIR, name, "agent_config.yaml"))
)

dbutils.widgets.dropdown(
    name="agent",
    defaultValue="wanderbricks" if "wanderbricks" in AGENT_CHOICES else AGENT_CHOICES[0],
    choices=AGENT_CHOICES,
    label="Agent",
)
AGENT_NAME = dbutils.widgets.get("agent")
AGENT_DIR = os.path.join(EXAMPLES_DIR, AGENT_NAME)

with open(os.path.join(AGENT_DIR, "agent_config.yaml")) as f:
    agent_cfg = yaml.safe_load(f)

PROMPT_URIS = [f"prompts:/{name}@production" for name in agent_cfg["prompt_registry"].values()]
GATEWAY_ENDPOINTS = {
    cfg["smart_endpoint"]: cfg["candidate_models"]
    for cfg in agent_cfg["gateway_endpoints"].values()
}
print(f"Using agent: {AGENT_NAME}")
print(f"  prompts: {len(PROMPT_URIS)}, endpoints: {len(GATEWAY_ENDPOINTS)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import the agent's `predict`
# MAGIC
# MAGIC The library transparently routes the agent's calls through `<endpoint>-exp`
# MAGIC clones during optimization, so the agent code doesn't need any special
# MAGIC handling for optimization vs. production.

# COMMAND ----------

predict = importlib.import_module(f"examples.{AGENT_NAME}.agent").predict

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scorers
# MAGIC
# MAGIC Default: `Correctness` (LLM-judged). Edit per agent. MLflow `Scorer` objects
# MAGIC and plain `(inputs, expectations, answer) -> float` callables both work.

# COMMAND ----------

from mlflow.genai.scorers import Correctness

SCORERS = [Correctness(model="databricks:/databricks-gpt-5-4-nano")]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Eval data
# MAGIC
# MAGIC Loads `examples/<agent>/eval_set.yaml`. Rows follow the MLflow evaluation
# MAGIC schema: `{"inputs": dict, "expectations": dict}` per row, where `expectations`
# MAGIC typically carries `{"expected_response": ...}` for the `Correctness` judge
# MAGIC (or `guidelines`, `expected_facts`, etc. for other judges). Swap this cell
# MAGIC for JSONL / parquet / Delta as needed -- the library only cares about the row shape.

# COMMAND ----------

EVAL_SET = os.path.join(AGENT_DIR, "eval_set.yaml")
if not os.path.exists(EVAL_SET):
    if AGENT_NAME == "hotpotqa":
        raise FileNotFoundError(
            f"{EVAL_SET} is missing. Run `examples/hotpotqa/prepare_data.py` once "
            "to download a slice of the HotpotQA distractor split."
        )
    raise FileNotFoundError(f"No eval_set.yaml at {EVAL_SET}")

with open(EVAL_SET) as f:
    rows = yaml.safe_load(f)

INPUT_KEYS = tuple(k for k in rows[0] if k != "expected_answer")
def _to_record(r):
    return {
        "inputs": {k: r[k] for k in INPUT_KEYS},
        "expectations": {"expected_response": r["expected_answer"]},
    }

VAL_SIZE = 5
train_data = [_to_record(r) for r in rows[:-VAL_SIZE]]
val_data   = [_to_record(r) for r in rows[-VAL_SIZE:]]
print(f"train={len(train_data)}, val={len(val_data)}")

MAX_METRIC_CALLS = 50

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run

# COMMAND ----------

mlflow.set_experiment("/Users/{}/smart-model-upgrades-runs".format(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
))

with mlflow.start_run() as run:
    mlflow.log_params({
        "agent": AGENT_NAME,
        "max_metric_calls": MAX_METRIC_CALLS,
        "n_prompts": len(PROMPT_URIS),
        "n_endpoints": len(GATEWAY_ENDPOINTS),
    })

    result = smu.optimize_prompts_and_models(
        predict_fn=predict,
        train_data=train_data,
        val_data=val_data,
        prompt_uris=PROMPT_URIS,
        gateway_endpoints=GATEWAY_ENDPOINTS,
        scorers=SCORERS,
        max_metric_calls=MAX_METRIC_CALLS,
    )

    mlflow.log_metric("baseline_score", result.baseline_score)
    mlflow.log_metric("best_score", result.best_score)

print(f"baseline {result.baseline_score:.3f} -> best {result.best_score:.3f} "
      f"(delta {result.best_score - result.baseline_score:+.3f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspect the winner

# COMMAND ----------

for ep_name in GATEWAY_ENDPOINTS:
    print(f"  model:{ep_name} -> {result.best_candidate.get(f'model:{ep_name}')}")
for uri in PROMPT_URIS:
    short = uri.split("@", 1)[0].rsplit(".", 1)[-1]
    new = result.best_candidate.get(f"prompt:{short}")
    print(f"  prompt:{short} -> {len(new) if new else 0} chars")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Promote winner to production
# MAGIC
# MAGIC Updates production gateway endpoints to the winning models, registers any
# MAGIC rewritten prompts as new versions, and points `@production_previous` at
# MAGIC the prior version for rollback.

# COMMAND ----------

smu.promote_to_prod(result)
