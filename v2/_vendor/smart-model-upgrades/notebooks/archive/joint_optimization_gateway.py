# Databricks notebook source
# MAGIC %md
# MAGIC # Joint Model + Prompt Optimization (Gateway)
# MAGIC
# MAGIC Same GEPA optimization as `joint_optimization_direct.py`, but the agent
# MAGIC routes through AI Gateway V2 experimental endpoints during optimization.
# MAGIC When GEPA proposes a new model, the experimental endpoint is updated
# MAGIC in-loop so the agent always calls through the gateway.
# MAGIC
# MAGIC 1. **Setup** -- Create experimental gateway endpoints (cloned from prod)
# MAGIC 2. **Optimize** -- GEPA loop updates experimental endpoints in-loop
# MAGIC 3. **Promote to prod** -- Update production endpoints to winners
# MAGIC 4. **Cleanup** -- Delete experimental endpoints

# COMMAND ----------

# MAGIC %pip install -e .. databricks-langchain 'langgraph>=0.4' langchain-core langchain-openai -qU

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys
import time
sys.path.insert(0, os.path.join(os.getcwd(), ".."))

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

MAX_METRIC_CALLS = 500

OPT_CONFIG_PATH = os.path.join(os.getcwd(), "..", "configs", "optimization_config.yaml")
AGENT_CONFIG_PATH = os.path.join(os.getcwd(), "..", "configs", "config.yaml")

config = smu.OptConfig.from_yaml(
    OPT_CONFIG_PATH,
    agent_config_path=AGENT_CONFIG_PATH,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up scorers

# COMMAND ----------

from mlflow.genai.scorers import RelevanceToQuery, Correctness
from mlflow.entities import Feedback
from databricks_langchain import ChatDatabricks
from langchain_core.messages import HumanMessage

JUDGE_PROMPT = """Question: {question}
Expected answer: {expected_answer}
Actual answer: {answer}

Score the actual answer from 0.0 to 1.0 based on correctness and relevance.
Return ONLY a number, nothing else."""

_judge_llm = ChatDatabricks(endpoint=config.judge_model)

def score_quality(inputs, expected_answer, answer):
    """Custom LLM judge score, 0.0-1.0."""
    prompt = JUDGE_PROMPT.format(
        question=inputs.get("question", ""), expected_answer=expected_answer, answer=answer,
    )
    try:
        response = _judge_llm.invoke([HumanMessage(content=prompt)])
        return max(0.0, min(1.0, float(response.content.strip())))
    except Exception:
        return 0.0

SCORERS = [
    score_quality,
    RelevanceToQuery(model=config.scorer_model),
    Correctness(model=config.scorer_model),
]
config.scorers = SCORERS

print(f"Quality scorers: {len(SCORERS)}")
for s in SCORERS:
    name = getattr(s, "name", getattr(s, "__name__", str(s)))
    print(f"  - {name}")

print("Initial models:")
for k, v in config.initial_models.items():
    print(f"  {k}: {v}")
print(f"\nGateway mode: routing through experimental endpoints")
for k, v in config.gateway_exp_endpoints.items():
    print(f"  {k}: {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Experimental Gateway Endpoints
# MAGIC
# MAGIC Reads each production endpoint's current model and creates a matching
# MAGIC `-exp` endpoint. Idempotent -- skips if they already exist.

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
# MAGIC The agent is imported as-is (same code path as serving). Experimental
# MAGIC endpoints are selected via env var. Prompt injection and gateway
# MAGIC destination swaps are handled generically by `build_optimization_fns`.

# COMMAND ----------

os.environ["ENDPOINT_MODE"] = "experimental"
from agent.agent import AGENT
from mlflow.types.responses import ResponsesAgentRequest

mlflow.langchain.autolog()

def predict_fn(inputs):
    request = ResponsesAgentRequest(input=[{"role": "user", "content": inputs["question"]}])
    response = AGENT.predict(request)
    if not response.output:
        return ""
    block = response.output[-1].content[0]
    return block["text"] if isinstance(block, dict) else block.text

run_fn, apply_fn = smu.build_optimization_fns(config, predict_fn)
adapter = smu.AgentAdapter(config, run_fn=run_fn, apply_fn=apply_fn)

for comp, ep in config.gateway_exp_endpoints.items():
    print(f"  {comp} -> {ep}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Start MLflow run

# COMMAND ----------

run = mlflow.start_run(run_name="joint_optimization_gateway")

mlflow.log_params({
    "optimizer": "gepa",
    "max_metric_calls": MAX_METRIC_CALLS,
    "reflection_endpoint": config.reflection_model,
    "num_scorers": len(SCORERS),
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

eval_path = os.path.join(os.getcwd(), "..", "configs", "optimization_eval_set.yaml")
train_data, val_data = smu.load_eval_data(eval_path, split_at=75, random_shuffle=True)
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

traces = mlflow.search_traces(
    run_id=mlflow.active_run().info.run_id,
    max_results=100,
)
predict_traces = traces[traces["tags"].apply(lambda t: t.get("mlflow.traceName") == "predict")]
latencies_ms = predict_traces["execution_duration"].tolist()

print(f"Baseline eval end-to-end latencies ({len(latencies_ms)} calls):")
for i, ms in enumerate(latencies_ms):
    print(f"  [{i+1}] {ms / 1000:.1f}s")
print(f"\nAverage: {sum(latencies_ms) / len(latencies_ms) / 1000:.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## GEPA Optimization
# MAGIC
# MAGIC Each time GEPA proposes a candidate with a new model, the experimental
# MAGIC gateway endpoint is updated in-loop and the agent evaluates through it.

# COMMAND ----------

result = gepa.optimize(
    seed_candidate=seed_candidate,
    trainset=train_data,
    valset=val_data,
    adapter=adapter,
    reflection_lm=f"databricks/{config.reflection_model}",
    reflection_prompt_template=reflection_templates,
    reflection_minibatch_size=5,
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

print("=== Optimized Configuration ===\n")

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

_opt_eval_start_ms = int(time.time() * 1000)
opt_batch = adapter.evaluate(val_data, optimized)
optimized_score = sum(opt_batch.scores) / len(opt_batch.scores)

print(f"Baseline: {baseline_score:.3f} -> Optimized: {optimized_score:.3f} "
      f"(delta: {optimized_score - baseline_score:+.3f})")

# COMMAND ----------

# MAGIC %md
# MAGIC Optimized eval latencies:

# COMMAND ----------

# DBTITLE 1,Optimized eval latencies
traces = mlflow.search_traces(
    run_id=mlflow.active_run().info.run_id,
    max_results=200,
)
opt_predict = traces[
    (traces["tags"].apply(lambda t: t.get("mlflow.traceName") == "predict"))
    & (traces["request_time"] >= _opt_eval_start_ms)
]
opt_latencies_ms = opt_predict["execution_duration"].tolist()

print(f"Optimized eval end-to-end latencies ({len(opt_latencies_ms)} calls):")
for i, ms in enumerate(opt_latencies_ms):
    print(f"  [{i+1}] {ms / 1000:.1f}s")
opt_avg = sum(opt_latencies_ms) / len(opt_latencies_ms) / 1000
baseline_avg = sum(latencies_ms) / len(latencies_ms) / 1000
print(f"\nOptimized avg: {opt_avg:.1f}s  |  Baseline avg: {baseline_avg:.1f}s  |  Delta: {opt_avg - baseline_avg:+.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Experimental Endpoints

# COMMAND ----------

smu.verify_gateway_endpoints(config.gateway_exp_endpoints.values())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Promote to Production

# COMMAND ----------

smu.promote_to_prod(optimized, config)

# COMMAND ----------

smu.verify_gateway_endpoints(config.gateway_endpoints.values())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy Prompts to Registry

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
        version = smu.client.create_prompt_version(name=full_name, template=prompt)
        smu.client.set_prompt_alias(name=full_name, alias="production", version=version.version)
        mlflow.log_param(f"final_prompt/{comp}", f"prompts:/{full_name}/{version.version}")
        smu.client.link_prompt_version_to_run(run_id=run_id, prompt=version)
        print(f"{comp}: @production -> version {version.version}")
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
# MAGIC ## Validate Through Production Endpoints

# COMMAND ----------

print("Validating through production endpoints...")
print("(Experimental endpoints mirror prod after promote_to_prod)")

def ask(question):
    answer, latency, _ = run_fn({"question": question})
    print(f"({latency:.1f}s) {answer}")

# COMMAND ----------

print("--- Prod validation ---")
ask("Find me a place in Paris for 2 people, under $150/night, in August 2026")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleanup Experimental Endpoints

# COMMAND ----------

smu.cleanup_exp_endpoints(config)
