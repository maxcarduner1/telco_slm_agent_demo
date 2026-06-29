# Databricks notebook source
# MAGIC %md
# MAGIC # Per-Component Optimization (Direct)
# MAGIC
# MAGIC Optimizes each of the agent's LLM calls independently. For every
# MAGIC (component, candidate_model) pair, fixes that component's LLM and runs
# MAGIC `gepa.optimize()` on just that component's prompt. The best (model,
# MAGIC prompt) per component is then deployed to the prompt registry.
# MAGIC
# MAGIC Mirrors the in-process pattern from `joint_optimization_direct.py`:
# MAGIC builds the agent graph directly with `ChatDatabricks` (no gateway
# MAGIC routing) and rebuilds it whenever a model changes.

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config

# COMMAND ----------

MAX_METRIC_CALLS_PER_MODEL = 50

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

run = mlflow.start_run(run_name="per_component_optimization")

mlflow.log_params({
    "optimizer": "gepa_per_component",
    "max_metric_calls_per_model": MAX_METRIC_CALLS_PER_MODEL,
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
seed_prompt_strings = {
    comp: (prompt_seeds[comp].template if hasattr(prompt_seeds[comp], "template") else str(prompt_seeds[comp]))
    for comp in config.components
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reflection templates

# COMMAND ----------

prompt_reflection_templates = smu.build_prompt_reflection_templates(config)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Baseline Eval

# COMMAND ----------

seed_candidate = smu.build_seed_candidate(prompt_seeds, config)
baseline_batch = adapter.evaluate(val_data, seed_candidate)
baseline_score = sum(baseline_batch.scores) / len(baseline_batch.scores)
print(f"Baseline score: {baseline_score:.3f}")
mlflow.log_metric("initial_eval_score", baseline_score)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Per-Component Optimization Loop
# MAGIC
# MAGIC For each component x candidate model: fix that component's LLM, run
# MAGIC gepa.optimize on just that component's prompt.

# COMMAND ----------

results = {}

for target_comp in config.components:
    print(f"\n{'='*60}")
    print(f"Optimizing: {target_comp}")
    print(f"{'='*60}")

    call_results = []

    for endpoint in config.candidate_models[target_comp]:
        print(f"\n  Candidate model: {endpoint}")

        # Base candidate fixes target's model + every component's seed prompt;
        # other components stay at their initial model.
        base_candidate = {}
        for comp in config.components:
            base_candidate[f"{comp}_prompt"] = seed_prompt_strings[comp]
            base_candidate[f"{comp}_model"] = (
                endpoint if comp == target_comp else config.initial_models[comp]
            )

        # Default args capture loop variables by value (avoids closure bug)
        def _make_full(gepa_candidate, _base=dict(base_candidate), _comp=target_comp):
            full = dict(_base)
            full[f"{_comp}_prompt"] = gepa_candidate["prompt"]
            return full

        single_adapter = smu.SingleComponentAdapter(
            adapter, _make_full, f"{target_comp}_prompt",
        )
        seed = {"prompt": seed_prompt_strings[target_comp]}
        single_template = {"prompt": prompt_reflection_templates[f"{target_comp}_prompt"]}

        print(f"    Running gepa.optimize (max_metric_calls={MAX_METRIC_CALLS_PER_MODEL})...")
        opt_start = time.perf_counter()
        try:
            result = gepa.optimize(
                seed_candidate=seed,
                trainset=train_data,
                valset=val_data,
                adapter=single_adapter,
                reflection_lm=f"databricks/{config.reflection_model}",
                reflection_prompt_template=single_template,
                reflection_minibatch_size=5,
                frontier_type="hybrid",
                max_metric_calls=MAX_METRIC_CALLS_PER_MODEL,
                display_progress_bar=True,
                use_mlflow=False,
            )
            score = result.val_aggregate_scores[result.best_idx]
            optimized_prompt = result.best_candidate["prompt"]
        except Exception as e:
            print(f"    ERROR: {e}")
            continue
        opt_duration = time.perf_counter() - opt_start

        prompt_changed = optimized_prompt != seed_prompt_strings[target_comp]
        print(f"    Optimization took {opt_duration:.0f}s")
        print(f"    Best valset score: {score:.3f}")
        print(f"    Prompt changed: {prompt_changed}")

        call_results.append({
            "component": target_comp,
            "endpoint": endpoint,
            "score": score,
            "optimized_prompt": optimized_prompt,
            "prompt_changed": prompt_changed,
            "opt_duration_s": opt_duration,
        })

    results[target_comp] = call_results
    print(f"\n  {target_comp}: {len(call_results)} candidates evaluated")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results + Winner Selection

# COMMAND ----------

print("="*60)
print("OPTIMIZATION RESULTS")
print("="*60)

winners = {}

for comp, call_results in results.items():
    print(f"\n--- {comp} ---")

    if not call_results:
        print("  No viable candidates!")
        continue

    ranked = sorted(call_results, key=lambda r: r["score"], reverse=True)

    for i, r in enumerate(ranked):
        changed = " *" if r["prompt_changed"] else ""
        print(f"  {i+1}. {r['endpoint']:40s} score={r['score']:.3f}{changed}")

    winners[comp] = ranked[0]
    mlflow.log_metric(f"winner_score/{comp}", ranked[0]["score"])
    mlflow.log_param(f"winner_model/{comp}", ranked[0]["endpoint"])
    print(f"  >> WINNER: {ranked[0]['endpoint']} (score={ranked[0]['score']:.3f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build final candidate from winners

# COMMAND ----------

final_candidate = {}
for comp in config.components:
    if comp in winners:
        final_candidate[f"{comp}_model"] = winners[comp]["endpoint"]
        final_candidate[f"{comp}_prompt"] = winners[comp]["optimized_prompt"]
    else:
        final_candidate[f"{comp}_model"] = config.initial_models[comp]
        final_candidate[f"{comp}_prompt"] = seed_prompt_strings[comp]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Post-Optimization Eval

# COMMAND ----------

opt_batch = adapter.evaluate(val_data, final_candidate)
optimized_score = sum(opt_batch.scores) / len(opt_batch.scores)
print(f"Baseline: {baseline_score:.3f} -> Optimized: {optimized_score:.3f} "
      f"(delta: {optimized_score - baseline_score:+.3f})")
mlflow.log_metric("final_eval_score", optimized_score)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy Winners to Registry

# COMMAND ----------

run_id = mlflow.active_run().info.run_id
for comp in config.components:
    mlflow.log_param(f"final_model/{comp}", final_candidate[f"{comp}_model"])
    model_changed = final_candidate[f"{comp}_model"] != config.initial_models[comp]
    mlflow.log_param(f"model_changed/{comp}", model_changed)

    prompt = final_candidate[f"{comp}_prompt"]
    full_name = config.prompt_names[comp]
    if prompt != seed_prompt_strings[comp]:
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

# COMMAND ----------

mlflow.end_run()
print("MLflow run ended.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## End-to-End Validation

# COMMAND ----------

adapter.apply_fn(final_candidate)

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
