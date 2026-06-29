# Databricks notebook source
# MAGIC %md
# MAGIC # HotpotQA: Joint Model + Prompt Optimization (Gateway)
# MAGIC
# MAGIC Drop-in demonstration that the optimization loop is agent-agnostic.
# MAGIC Follows the MLflow blog example
# MAGIC (https://mlflow.org/blog/mlflow-prompt-optimization) but goes through
# MAGIC AI Gateway V2 and jointly optimizes the model endpoint alongside the
# MAGIC prompt.
# MAGIC
# MAGIC 1. **Setup** -- Create the experimental gateway endpoint (cloned from prod)
# MAGIC 2. **Optimize** -- GEPA loop updates the experimental endpoint in-loop
# MAGIC 3. **Promote to prod** -- Update `hotpotqa-smart-endpoint` to the winner
# MAGIC 4. **Cleanup** -- Delete the experimental endpoint

# COMMAND ----------

# MAGIC %pip install -e .. databricks-openai openai pyarrow -qU

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import time

import gepa
import mlflow
import yaml
import smart_model_upgrades as smu

os.environ["DATABRICKS_HOST"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
os.environ["DATABRICKS_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config

# COMMAND ----------

MAX_METRIC_CALLS = 100
NUM_SAMPLES = 30  # HotpotQA items to use (split 25 train / 5 val by default)

OPT_CONFIG_PATH = os.path.join(os.getcwd(), "..", "configs", "hotpotqa_optimization_config.yaml")
AGENT_CONFIG_PATH = os.path.join(os.getcwd(), "..", "configs", "hotpotqa_config.yaml")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare HotpotQA eval data
# MAGIC
# MAGIC Pull from the `hotpot_qa` dataset (distractor split), shape into
# MAGIC `{context, question, expected_answer}` rows.

# COMMAND ----------

# Download HotpotQA validation parquet directly from HuggingFace. Avoids
# the datasets/huggingface_hub/fsspec version tangle on Databricks runtime.
import io
import requests
import pyarrow.parquet as pq

HOTPOT_URL = (
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/"
    "refs%2Fconvert%2Fparquet/distractor/validation/0000.parquet"
)

resp = requests.get(HOTPOT_URL, timeout=60)
resp.raise_for_status()
table = pq.read_table(io.BytesIO(resp.content))
raw = table.slice(0, NUM_SAMPLES).to_pylist()

eval_rows = []
for ex in raw:
    context_text = "\n\n".join(
        f"Document {i+1}: {title}\n{' '.join(sentences)}"
        for i, (title, sentences) in enumerate(
            zip(ex["context"]["title"], ex["context"]["sentences"])
        )
    )
    eval_rows.append({
        "context": context_text,
        "question": ex["question"],
        "expected_answer": ex["answer"],
    })

EVAL_PATH = "/tmp/hotpotqa_eval_set.yaml"
with open(EVAL_PATH, "w") as f:
    yaml.safe_dump(eval_rows, f)

print(f"Prepared {len(eval_rows)} samples -> {EVAL_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up scorers
# MAGIC
# MAGIC Primary scorer is exact-match equivalence (same as the blog). MLflow's
# MAGIC `Correctness` provides a softer judged score as a secondary signal.

# COMMAND ----------

from mlflow.entities import Feedback
from mlflow.genai.judges import CategoricalRating
from mlflow.genai.scorers import Correctness, scorer

config = smu.OptConfig.from_yaml(
    OPT_CONFIG_PATH,
    agent_config_path=AGENT_CONFIG_PATH,
)

@scorer
def equivalence(outputs, expectations):
    return Feedback(
        name="equivalence",
        value=CategoricalRating.YES
        if (outputs or "").strip() == expectations["expected_response"].strip()
        else CategoricalRating.NO,
    )

SCORERS = [
    equivalence,
    Correctness(model=config.scorer_model),
]
config.scorers = SCORERS

for s in SCORERS:
    name = getattr(s, "name", getattr(s, "__name__", str(s)))
    print(f"  - {name}")

print("Initial models:")
for k, v in config.initial_models.items():
    print(f"  {k}: {v}")
print("\nGateway endpoints:")
for k, v in config.gateway_endpoints.items():
    print(f"  {k}: {v} (exp: {config.gateway_exp_endpoints[k]})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Experimental Gateway Endpoint

# COMMAND ----------

prod_models = smu.ensure_exp_endpoints(config)

# COMMAND ----------

smu.verify_gateway_endpoints(
    [*config.gateway_endpoints.values(), *config.gateway_exp_endpoints.values()]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build agent + optimization adapter
# MAGIC
# MAGIC Import the HotpotQA agent with `ENDPOINT_MODE=experimental` so its
# MAGIC DatabricksOpenAI client points at the `-exp` endpoint. Prompt injection
# MAGIC and gateway destination swaps are handled generically by
# MAGIC `build_optimization_fns`.

# COMMAND ----------

os.environ["ENDPOINT_MODE"] = "experimental"
from agent.hotpotqa_agent import predict as predict_fn

run_fn, apply_fn = smu.build_optimization_fns(config, predict_fn)
adapter = smu.AgentAdapter(config, run_fn=run_fn, apply_fn=apply_fn)

for comp, ep in config.gateway_exp_endpoints.items():
    print(f"  {comp} -> {ep}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Start MLflow run

# COMMAND ----------

mlflow.set_experiment("/Users/{}/hotpotqa-smart-upgrades".format(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
))

run = mlflow.start_run(run_name="optimize_hotpotqa")

mlflow.log_params({
    "optimizer": "gepa",
    "max_metric_calls": MAX_METRIC_CALLS,
    "reflection_endpoint": config.reflection_model,
    "num_scorers": len(SCORERS),
    "num_samples": NUM_SAMPLES,
    "opt/weight_quality": config.weight_quality,
    "opt/weight_latency": config.weight_latency,
    "opt/weight_cost": config.weight_cost,
    "opt/latency_hard_gate": config.latency_hard_gate,
    "deploy_mode": "gateway",
})
for comp, sla in config.latency_slas.items():
    mlflow.log_param(f"opt/latency_sla.{comp}", sla)
for comp, model in config.initial_models.items():
    mlflow.log_param(f"initial_model/{comp}", model)

print(f"MLflow run started: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load eval set + seed prompts

# COMMAND ----------

train_data, val_data = smu.load_eval_data(
    EVAL_PATH, input_keys=("context", "question"), split_at=25,
)
print(f"Train: {len(train_data)}, Val: {len(val_data)}")

# COMMAND ----------

prompt_seeds = smu.load_seed_prompts(config.prompt_names)

# COMMAND ----------

seed_candidate = smu.build_seed_candidate(prompt_seeds, config)
print(f"Seed candidate: {len(seed_candidate)} parameters")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Selection Template

# COMMAND ----------

reflection_templates = {
    **smu.build_model_reflection_templates(config),
    **smu.build_prompt_reflection_templates(config),
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Baseline Eval

# COMMAND ----------

baseline_batch = adapter.evaluate(val_data, seed_candidate)
baseline_score = sum(baseline_batch.scores) / len(baseline_batch.scores)
print(f"Baseline score: {baseline_score:.3f}")

# COMMAND ----------

mlflow.log_metric("initial_eval_score", baseline_score)

run_id = mlflow.active_run().info.run_id
for comp in config.components:
    name = config.prompt_names[comp]
    pv = mlflow.genai.load_prompt(f"prompts:/{name}@production")
    mlflow.log_param(f"initial_prompt/{comp}", f"prompts:/{name}/{pv.version}")
    smu.client.link_prompt_version_to_run(run_id=run_id, prompt=pv)

print(f"Logged initial_eval_score={baseline_score:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## GEPA Optimization

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspect Results

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
# MAGIC ## Post-Optimization Eval

# COMMAND ----------

opt_batch = adapter.evaluate(val_data, optimized)
optimized_score = sum(opt_batch.scores) / len(opt_batch.scores)

print(f"Baseline: {baseline_score:.3f} -> Optimized: {optimized_score:.3f} "
      f"(delta: {optimized_score - baseline_score:+.3f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Promote to Production

# COMMAND ----------

smu.promote_to_prod(optimized, config)

# COMMAND ----------

smu.verify_gateway_endpoints(config.gateway_endpoints.values())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy Prompt to Registry

# COMMAND ----------

mlflow.log_metric("final_eval_score", optimized_score)

run_id = mlflow.active_run().info.run_id
for comp in config.components:
    mlflow.log_param(f"final_model/{comp}", optimized[f"{comp}_model"])
    model_changed = optimized[f"{comp}_model"] != config.initial_models[comp]
    mlflow.log_param(f"model_changed/{comp}", model_changed)

    prompt = optimized[f"{comp}_prompt"]
    full_name = config.prompt_names[comp]
    seed_template = prompt_seeds[comp].template if hasattr(prompt_seeds[comp], "template") else str(prompt_seeds[comp])
    if prompt != seed_template:
        prev = mlflow.genai.load_prompt(f"prompts:/{full_name}@production")
        mlflow.genai.set_prompt_alias(name=full_name, alias="production-previous", version=prev.version)
        version = mlflow.genai.register_prompt(
            name=full_name,
            template=prompt,
            commit_message=f"GEPA-optimized on HotpotQA, model={optimized[f'{comp}_model']}",
        )
        mlflow.genai.set_prompt_alias(name=full_name, alias="production", version=version.version)
        mlflow.log_param(f"final_prompt/{comp}", f"prompts:/{full_name}/{version.version}")
        smu.client.link_prompt_version_to_run(run_id=run_id, prompt=version)
        print(f"{comp}: @production -> v{version.version} (previous: v{prev.version})")
    else:
        pv = mlflow.genai.load_prompt(f"prompts:/{full_name}@production")
        mlflow.log_param(f"final_prompt/{comp}", f"prompts:/{full_name}/{pv.version}")
        print(f"{comp}: prompt unchanged, keeping current @production")

print(f"\nLogged final_eval_score={optimized_score:.3f}")
print(f"Delta: {optimized_score - baseline_score:+.3f}")

# COMMAND ----------

mlflow.end_run()
print("MLflow run ended.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate Through Production Endpoint

# COMMAND ----------

def ask(context, question):
    answer, latency, _ = run_fn({"context": context, "question": question})
    print(f"({latency:.1f}s) {answer}")

# COMMAND ----------

sample = eval_rows[0]
print(f"Q: {sample['question']}")
print(f"Expected: {sample['expected_answer']}")
ask(sample["context"], sample["question"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleanup Experimental Endpoint

# COMMAND ----------

smu.cleanup_exp_endpoints(config)
