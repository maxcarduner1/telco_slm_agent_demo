# Databricks notebook source
# MAGIC %md
# MAGIC # Toy Agent Tester
# MAGIC
# MAGIC Parameterized end-to-end test of the smart_model_upgrades library against a
# MAGIC toy agent in `examples/<toy>/`. Each toy is a self-contained directory
# MAGIC with `agent.py`, `agent_config.yaml`, `optimization_config.yaml`,
# MAGIC `seed_prompts.yaml`, and `eval_set.yaml`.
# MAGIC
# MAGIC Set `TOY_DIR` below to one of:
# MAGIC
# MAGIC - `toy_translator`         (1 component, multi-key input)
# MAGIC - `toy_qa_critic`          (2 sequential components)
# MAGIC - `toy_email_writer`       (2 fan-out components)
# MAGIC - `toy_3step_research`     (3 sequential components)
# MAGIC
# MAGIC Then run all cells. The notebook registers prompts, creates prod + exp
# MAGIC gateway endpoints, runs GEPA, promotes the winner, and cleans up.

# COMMAND ----------

# MAGIC %pip install -e .. databricks-openai openai -qU

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import importlib.util
import os
import re

import gepa
import mlflow
import yaml

import smart_model_upgrades as smu
from smart_model_upgrades import ai_gateway as gw

os.environ["DATABRICKS_HOST"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
os.environ["DATABRICKS_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pick a toy

# COMMAND ----------

TOY_DIR = "toy_translator"
MAX_METRIC_CALLS = 50
SPLIT_AT = 15

TOY_PATH = os.path.join(os.getcwd(), "..", "examples", TOY_DIR)
AGENT_CONFIG_PATH = os.path.join(TOY_PATH, "agent_config.yaml")
OPT_CONFIG_PATH = os.path.join(TOY_PATH, "optimization_config.yaml")
SEED_PATH = os.path.join(TOY_PATH, "seed_prompts.yaml")
EVAL_PATH = os.path.join(TOY_PATH, "eval_set.yaml")
AGENT_PY = os.path.join(TOY_PATH, "agent.py")

for p in [AGENT_CONFIG_PATH, OPT_CONFIG_PATH, SEED_PATH, EVAL_PATH, AGENT_PY]:
    assert os.path.exists(p), f"Missing: {p}"
print(f"Toy: {TOY_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register prompts at @production

# COMMAND ----------

with open(SEED_PATH) as f:
    TEMPLATES = yaml.safe_load(f)

smu.register_prompts_from_config(AGENT_CONFIG_PATH, TEMPLATES)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create production gateway endpoints (idempotent)

# COMMAND ----------

with open(AGENT_CONFIG_PATH) as f:
    agent_cfg = yaml.safe_load(f)

for comp, ep_cfg in agent_cfg["gateway_endpoints"].items():
    name = ep_cfg["smart_endpoint"]
    initial_model = ep_cfg["initial_model"]
    try:
        existing = gw.get_endpoint(name)
        dests = existing.get("config", {}).get("destinations", [])
        current = dests[0]["name"] if dests else "unknown"
        print(f"{name}: already exists ({current})")
    except Exception:
        gw.create_endpoint(
            name=name,
            destinations=[gw.destination(
                f"system.ai.{initial_model}",
                "PAY_PER_TOKEN_FOUNDATION_MODEL",
                100,
            )],
            task_type="llm/v1/chat",
            tags=[
                gw.tag("managed_by", "smart-model-upgrades"),
                gw.tag("agent", TOY_DIR),
            ],
        )
        print(f"{name}: created with system.ai.{initial_model}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build OptConfig + scorers

# COMMAND ----------

from mlflow.genai.scorers import Correctness

config = smu.OptConfig.from_yaml(
    OPT_CONFIG_PATH,
    agent_config_path=AGENT_CONFIG_PATH,
)
config.scorers = [Correctness(model=config.scorer_model)]

print("Components:", list(config.components.keys()))
print("Initial models:")
for k, v in config.initial_models.items():
    print(f"  {k}: {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create experimental gateway endpoints

# COMMAND ----------

smu.ensure_exp_endpoints(config)
smu.verify_gateway_endpoints(
    [*config.gateway_endpoints.values(), *config.gateway_exp_endpoints.values()]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import toy agent (experimental mode)

# COMMAND ----------

os.environ["ENDPOINT_MODE"] = "experimental"
os.environ["AGENT_CONFIG_PATH"] = AGENT_CONFIG_PATH

spec = importlib.util.spec_from_file_location(f"toy_module_{TOY_DIR}", AGENT_PY)
toy_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(toy_module)
predict_fn = toy_module.predict

run_fn, apply_fn = smu.build_optimization_fns(config, predict_fn)
adapter = smu.AgentAdapter(config, run_fn=run_fn, apply_fn=apply_fn)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load eval data
# MAGIC Input keys are inferred from the first row of `eval_set.yaml` (all keys
# MAGIC except `expected_answer`).

# COMMAND ----------

with open(EVAL_PATH) as f:
    sample_row = yaml.safe_load(f)[0]
input_keys = tuple(k for k in sample_row.keys() if k != "expected_answer")

train_data, val_data = smu.load_eval_data(EVAL_PATH, input_keys=input_keys, split_at=SPLIT_AT)
print(f"input_keys={input_keys}, train={len(train_data)}, val={len(val_data)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Smoke test (one prediction at the seed candidate)

# COMMAND ----------

prompt_seeds = smu.load_seed_prompts(config.prompt_names)
seed_candidate = smu.build_seed_candidate(prompt_seeds, config)

sample = val_data[0]
out, latency, _ = run_fn(sample["inputs"])
print(f"Inputs: {sample['inputs']}")
print(f"Expected: {sample['outputs']}")
print(f"Got ({latency:.1f}s):\n{out}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Baseline eval

# COMMAND ----------

baseline_batch = adapter.evaluate(val_data, seed_candidate)
baseline_score = sum(baseline_batch.scores) / len(baseline_batch.scores)
print(f"Baseline score: {baseline_score:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## GEPA optimization

# COMMAND ----------

mlflow.set_experiment("/Users/{}/toy-{}".format(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get(),
    TOY_DIR.replace("_", "-"),
))

reflection_templates = {
    **smu.build_model_reflection_templates(config),
    **smu.build_prompt_reflection_templates(config),
}

with mlflow.start_run(run_name=f"optimize_{TOY_DIR}") as run:
    mlflow.log_params({
        "toy_dir": TOY_DIR,
        "max_metric_calls": MAX_METRIC_CALLS,
        "reflection_endpoint": config.reflection_model,
        "num_components": len(config.components),
    })
    for comp, model in config.initial_models.items():
        mlflow.log_param(f"initial_model/{comp}", model)

    mlflow.log_metric("initial_eval_score", baseline_score)

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=train_data,
        valset=val_data,
        adapter=adapter,
        reflection_lm=f"databricks/{config.reflection_model}",
        reflection_prompt_template=reflection_templates,
        frontier_type="hybrid",
        max_metric_calls=MAX_METRIC_CALLS,
        display_progress_bar=True,
        use_mlflow=True,
    )
    optimized = result.best_candidate

    opt_batch = adapter.evaluate(val_data, optimized)
    optimized_score = sum(opt_batch.scores) / len(opt_batch.scores)
    mlflow.log_metric("final_eval_score", optimized_score)

    for comp in config.components:
        mlflow.log_param(f"final_model/{comp}", optimized[f"{comp}_model"])
        changed = optimized[f"{comp}_model"] != config.initial_models[comp]
        mlflow.log_param(f"model_changed/{comp}", changed)

    print(f"\nBaseline: {baseline_score:.3f} -> Optimized: {optimized_score:.3f} "
          f"(delta: {optimized_score - baseline_score:+.3f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspect winner

# COMMAND ----------

for comp in config.components:
    model = optimized[f"{comp}_model"]
    prompt = optimized[f"{comp}_prompt"]
    changed = " (CHANGED)" if model != config.initial_models[comp] else ""
    print(f"--- {comp} ---")
    print(f"  Model:  {model}{changed}")
    print(f"  Prompt: {prompt[:300]}{'...' if len(prompt) > 300 else ''}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Promote winner to production + cleanup

# COMMAND ----------

smu.promote_to_prod(optimized, config)
smu.verify_gateway_endpoints(config.gateway_endpoints.values())

# COMMAND ----------

run_id = run.info.run_id
for comp in config.components:
    full_name = config.prompt_names[comp]
    seed_template = (
        prompt_seeds[comp].template
        if hasattr(prompt_seeds[comp], "template")
        else str(prompt_seeds[comp])
    )
    new_prompt = optimized[f"{comp}_prompt"]
    if new_prompt != seed_template:
        prev = mlflow.genai.load_prompt(f"prompts:/{full_name}@production")
        mlflow.genai.set_prompt_alias(
            name=full_name, alias="production-previous", version=prev.version,
        )
        version = mlflow.genai.register_prompt(
            name=full_name,
            template=new_prompt,
            commit_message=f"GEPA-optimized on {TOY_DIR}, model={optimized[f'{comp}_model']}",
        )
        mlflow.genai.set_prompt_alias(
            name=full_name, alias="production", version=version.version,
        )
        smu.client.link_prompt_version_to_run(run_id=run_id, prompt=version)
        print(f"{comp}: @production -> v{version.version} (previous v{prev.version})")
    else:
        print(f"{comp}: prompt unchanged, keeping current @production")

# COMMAND ----------

smu.cleanup_exp_endpoints(config)
print("Cleaned up -exp endpoints.")
