# Databricks notebook source
# MAGIC %md
# MAGIC # Joint Model + Prompt Optimization
# MAGIC
# MAGIC Finds the best (prompt, LLM) pair for each of the agent's 3 components
# MAGIC in a single GEPA run. The agent has 6 optimizable parameters (3 prompts +
# MAGIC 3 models). GEPA evolves prompts via reflection and selects models from a
# MAGIC fixed candidate list.

# COMMAND ----------

# MAGIC %pip install -e .. databricks-langchain 'langgraph>=0.4' langchain-core -qU

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
# MAGIC
# MAGIC Quality = average of all scorers below. Accepts MLflow built-in scorers
# MAGIC and plain callables with signature (question, expected_answer, answer) -> float.

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build agent graph for optimization

# COMMAND ----------

from agent.agent import build_graph, LangGraphResponsesAgent
from databricks_langchain.genie import GenieAgent
from databricks.sdk import WorkspaceClient
from mlflow.types.responses import ResponsesAgentRequest

mlflow.langchain.autolog()

with open(AGENT_CONFIG_PATH) as f:
    agent_cfg = yaml.safe_load(f)

# Mutable prompt store (raw Jinja templates, wrapped in JinjaPrompt for .format())
_prompt_store = {}
_current_models = dict(config.initial_models)

def load_prompts_fn():
    return {k: smu.JinjaPrompt(v) for k, v in _prompt_store.items()}

llms = {comp: ChatDatabricks(endpoint=config.initial_models[comp])
        for comp in config.components}
genie_agent = GenieAgent(
    genie_space_id=agent_cfg["databricks_resources"]["genie_space_id"],
    genie_agent_name="WanderBricks",
    client=WorkspaceClient(),
)
graph = build_graph(
    llms, load_prompts_fn, genie_agent,
    max_worker_rounds=agent_cfg.get("max_worker_rounds", 4),
    enrichment_recursion_limit=agent_cfg.get("enrichment_recursion_limit", 10),
)
agent = LangGraphResponsesAgent(graph)

def run_fn(inputs):
    """Run agent through the ResponsesAgent wrapper (same path as serving)."""
    start = time.perf_counter()
    try:
        request = ResponsesAgentRequest(input=[{"role": "user", "content": inputs["question"]}])
        response = agent.predict(request)
        final = response.output[-1].content[0]["text"] if response.output else ""
        trace = smu.extract_trace_summary(config.components)
    except Exception as e:
        final = f"ERROR: {e}"
        trace = smu.extract_trace_summary(config.components)
    return final, time.perf_counter() - start, trace

def apply_fn(candidate):
    global graph, agent
    for comp in config.components:
        _prompt_store[comp] = candidate[f"{comp}_prompt"]
    # Rebuild graph if any model changed
    models_changed = any(
        candidate[f"{comp}_model"] != _current_models[comp]
        for comp in config.components
    )
    if models_changed:
        new_llms = {comp: ChatDatabricks(endpoint=candidate[f"{comp}_model"])
                    for comp in config.components}
        graph = build_graph(
            new_llms, load_prompts_fn, genie_agent,
            max_worker_rounds=agent_cfg.get("max_worker_rounds", 4),
            enrichment_recursion_limit=agent_cfg.get("enrichment_recursion_limit", 10),
        )
        agent = LangGraphResponsesAgent(graph)
        for comp in config.components:
            _current_models[comp] = candidate[f"{comp}_model"]

adapter = smu.AgentAdapter(config, run_fn=run_fn, apply_fn=apply_fn)

print("Agent graph built. Adapter ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Start MLflow run

# COMMAND ----------

run = mlflow.start_run(run_name="joint_optimization")

mlflow.log_params({
    "optimizer": "gepa",
    "max_metric_calls": MAX_METRIC_CALLS,
    "reflection_endpoint": config.reflection_model,
    "num_scorers": len(SCORERS),
    "opt/weight_quality": config.weight_quality,
    "opt/weight_latency": config.weight_latency,
    "opt/weight_cost": config.weight_cost,
    "opt/latency_hard_gate": config.latency_hard_gate,
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
train_data, val_data = smu.load_eval_data(eval_path, split_at=75)
print(f"Train: {len(train_data)}, Val: {len(val_data)}")

# COMMAND ----------

prompt_seeds = smu.load_seed_prompts(config.prompt_names)

# COMMAND ----------

seed_candidate = smu.build_seed_candidate(prompt_seeds, config)
print(f"Seed candidate: {len(seed_candidate)} parameters")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Selection Template
# MAGIC
# MAGIC GEPA handles prompt reflection natively. For model components, we provide
# MAGIC a template that constrains selection to valid endpoint names.

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

opt_batch = adapter.evaluate(val_data, optimized)
optimized_score = sum(opt_batch.scores) / len(opt_batch.scores)

print(f"Baseline: {baseline_score:.3f} -> Optimized: {optimized_score:.3f} "
      f"(delta: {optimized_score - baseline_score:+.3f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy Winners to Registry

# COMMAND ----------

mlflow.log_metric("final_eval_score", optimized_score)

run_id = mlflow.active_run().info.run_id
for comp in config.components:
    mlflow.log_param(f"final_model/{comp}", optimized[f"{comp}_model"])
    model_changed = optimized[f"{comp}_model"] != config.initial_models[comp]
    mlflow.log_param(f"model_changed/{comp}", model_changed)

    prompt = optimized[f"{comp}_prompt"]
    full_name = config.prompt_names[comp]
    if prompt != prompt_seeds[comp]:
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
# MAGIC ## End-to-End Validation

# COMMAND ----------

adapter.apply_fn(optimized)

def ask(question):
    answer, latency, _ = adapter.run_fn({"question": question})
    print(f"({latency:.1f}s) {answer}")

# COMMAND ----------

print("--- Test 1: Property + Weather ---")
ask("Find me a place in Paris for 2 people, under $150/night, in August 2026")

# COMMAND ----------

print("--- Test 2: Weather only ---")
ask("What's the weather like in Tokyo in December?")

# COMMAND ----------

print("--- Test 3: Edge case ---")
ask("Just say hello.")
