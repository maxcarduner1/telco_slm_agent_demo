# Smart Model Upgrades

Optimize the **prompts and the per-endpoint LLMs** of an agentic application jointly, using GEPA as the optimizer, MLflow Prompt Registry as the prompt store, and AI Gateway V2 as the hot-swap layer for the models.

The user-facing surface is one function:

```python
result = smu.optimize_prompts_and_models(
    predict_fn=predict,
    train_data=train, val_data=val,
    prompt_uris=[
        "prompts:/cat.schema.supervisor@production",
        "prompts:/cat.schema.query_rewriter@production",
    ],
    gateway_endpoints={
        "wb-supervisor":     ["databricks-claude-sonnet-4", "databricks-gpt-5-4-mini"],
        "wb-query-rewriter": ["databricks-llama-4-maverick", "databricks-claude-haiku-4-5"],
    },
    scorers=SCORERS,
    max_metric_calls=200,
)
smu.promote_to_prod(result)
```

The reference agents (`examples/wanderbricks/`, `examples/hotpotqa/`) and the toys under `examples/toy_*/` are fixtures used to prove the loop is agent-agnostic.

---

## Why this exists

Most prompt-optimization workflows freeze the model and only evolve prompts. In practice, the right model for a routing supervisor is rarely the right model for a SQL query rewriter or a tool-calling worker. This library jointly searches the `(prompt, model)` space across an agent's components, scores candidates against a composite `quality / latency / cost` objective, and promotes the winner by re-pointing AI Gateway endpoints — no agent redeploy required.

Optimization runs **gateway-in-loop**: the agent calls models through `<endpoint>-exp` AI Gateway clones; GEPA swaps destinations on the clones during optimization; promotion is a single PATCH on the prod endpoints. `optimize_prompts_and_models` creates and tears down the exp endpoints automatically.

---

## Install

```bash
pip install -e .
```

From a Databricks notebook: `%pip install -e '..[demos]'` from anywhere under `notebooks/`.

Dev extras (pytest + pytest-mock): `pip install -e '.[dev]'`

---

## Public API

| | |
|---|---|
| `smu.optimize_prompts_and_models(...)` | Optimize prompts and/or model choices. Returns `Result`. |
| `smu.promote_to_prod(result)` | Apply the winner to production endpoints + prompt registry. |
| `smu.score(predict_fn, val_data, *, scorers)` | Run `predict_fn` over `val_data` and average scorer outputs. |
| `smu.setup_endpoints(endpoints)` | One-time bootstrap: create or sync prod gateway endpoints. |
| `smu.Result` | Output of `optimize_prompts_and_models`: `best_candidate`, `best_score`, `baseline_score`, plus the inputs needed for `promote_to_prod`. |

`optimize_prompts_and_models` accepts an independent `prompt_uris` list and a `gateway_endpoints` dict (either may be empty, but at least one must be non-empty), so you can tune just the prompts, just the models, or both.

---

## Repository layout

```
smart_model_upgrades/         # installable library
  __init__.py                 # public API re-exports
  optimization.py             # optimize_prompts_and_models + adapter + scoring + lifecycle
  ai_gateway.py               # thin requests wrapper for AI Gateway V2 CRUD
  prompt_registry.py          # internal CRUD helpers
  genie.py                    # Genie conversation cleanup

examples/                     # reference + toy agents
  wanderbricks/, wanderbricks_up_to_date/, hotpotqa/
  toy_translator/, toy_qa_critic/, toy_email_writer/, toy_3step_research/
  support_agent/, minimal_agent.py

notebooks/
  setup.py                    # one-time prompt registration + endpoint creation
  optimize.py                 # template optimization notebook
  ai_gateway_demo.py          # V2 CRUD walkthrough
  archive/                    # earlier explorations

tests/                        # pytest suite, no workspace required
```

### Per-agent dir

```
<agent>/
  agent.py                    # exports predict(inputs: dict)
  agent_config.yaml           # gateway_endpoints + prompt_registry (used by setup.py only)
  seed_prompts.yaml           # initial templates per component (used by setup.py only)
  eval_set.yaml               # rows of {...inputs, expected_answer}
```

The optimization loop never reads the agent directory — `optimize_prompts_and_models` takes prompt URIs and endpoint lists directly. The YAMLs only exist to bootstrap the prompts and endpoints once.

---

## Quickstart

1. **`notebooks/setup.py`** — edit `AGENT_DIR`, run all cells. Registers seed prompts at `@production` and creates production gateway endpoints. Idempotent.
2. **`notebooks/optimize.py`** — edit `PROMPT_URIS`, `GATEWAY_ENDPOINTS`, and your scorer. Runs GEPA, prints baseline-vs-optimized scores. Call `smu.promote_to_prod(result)` to roll the winner out.

### Programmatic / non-notebook

```python
import mlflow
from mlflow.genai.scorers import Correctness
import smart_model_upgrades as smu

from my_agent.agent import predict  # your code

# Build train_data and val_data yourself; rows follow the MLflow eval schema:
# {"inputs": dict, "expectations": dict}, where `expectations` typically holds
# {"expected_response": ...} for Correctness or {"guidelines": [...]} for the Guidelines judge.
train_data = [{"inputs": {...}, "expectations": {"expected_response": ...}}, ...]
val_data   = [{"inputs": {...}, "expectations": {"expected_response": ...}}, ...]

SCORERS = [Correctness(model="databricks:/databricks-gpt-5-4-nano")]

with mlflow.start_run():
    result = smu.optimize_prompts_and_models(
        predict_fn=predict,
        train_data=train_data, val_data=val_data,
        prompt_uris=["prompts:/cat.schema.supervisor@production"],
        gateway_endpoints={"my-supervisor": ["databricks-claude-sonnet-4", ...]},
        scorers=SCORERS,
        max_metric_calls=100,
    )

smu.promote_to_prod(result)
```

### The BYOA contract

Your **`agent.py`** must:

1. Expose `predict(inputs: dict)` at module level. The return value is passed straight to your scorers — strings, dicts, structured-response objects, and tool-calling agent outputs all work, as long as the scorers you configure know how to read them.
2. Load prompts via `mlflow.genai.load_prompt("prompts:/<name>@production")`, then format with `pv.format(...)`. The library patches `PromptVersion.template` at the class level for each evaluation, so any `pv.format(...)` (which reads `pv.template` internally) returns the GEPA candidate during the eval. No live-reload needed — works whether you call `load_prompt` per request or cache the `PromptVersion` once at module import. **Contract caveat:** if you cache the rendered string itself (`tmpl_str = pv.template` once, then reuse `tmpl_str` forever), the patch never fires and you'll silently optimize against the seed prompt — always go through `pv.format(...)`.
3. Route LLM calls through AI Gateway V2 endpoints (using the OpenAI Python client, `databricks_openai`, `langchain-openai`, or `dspy`'s OpenAI LM). The library transparently rewrites `model=<endpoint>` to `model=<endpoint>-exp` for the duration of optimization, so the agent code uses the prod endpoint name unchanged.
4. Enable MLflow autologging at import (`mlflow.openai.autolog()` or `mlflow.langchain.autolog()`). The library pulls per-call latency and token usage from the active trace.

`examples/toy_translator/agent.py` is a ~30-line reference.

---

## Scoring

`optimize_prompts_and_models`'s composite score is:

```
score = weight_quality * mean(scorers)
      + weight_latency * max(0, 1 - latency / latency_hard_gate)
      + weight_cost    * max(0, 1 - estimated_usd / cost_soft_gate)
```

Defaults: `weight_quality=0.7`, `weight_latency=0.2`, `weight_cost=0.1`, `latency_hard_gate=60.0` (seconds), `cost_soft_gate=0.02` (USD/call). All kwargs of `optimize_prompts_and_models`. Weights must be `>= 0` and sum to `1.0`. Candidates over the latency hard gate score 0; candidates over the cost soft gate score 0 on the cost component only.

**Scorers** can be MLflow `Scorer` objects (`Correctness`, `Guidelines`, custom `@scorer` callables) or plain `(inputs, expectations, answer) -> float` callables (which index into `expectations` themselves, typically `expectations["expected_response"]`). `Feedback` and `CategoricalRating` returns are converted to numeric automatically. Scorer failures are warned-once per (scorer, error-kind) so silent zero-quality runs are diagnosable.

**Cost** uses an internal per-model token-cost table (DBU per 1M tokens × $0.07/DBU) with a 500-input / 200-output token fallback when the active trace has no token usage. Pass `token_costs={"<model>": {"input": <DBU/1M>, "output": <DBU/1M>}}` to extend or override the table; if `weight_cost > 0` and a candidate model isn't covered, `optimize_prompts_and_models` raises with the expected shape rather than silently using a stub rate.

---

## Important gotchas

- **Endpoint routing is transparent for OpenAI-client agents.** The library monkeypatches `openai.resources.chat.completions.Completions.create` and `openai.resources.responses.Responses.create` (sync + async, all four) for the duration of optimization to rewrite the `model=` arg. Agents that hit the gateway via raw `requests` / `httpx` are not covered today.
- **Aliases must be underscores.** MLflow rejects hyphens — `production_previous`, not `production-previous`.
- **Scorer model URI format.** MLflow `Scorer` objects expect `databricks:/<model>` (colon-slash). litellm-consumed strings (the GEPA reflection LM) use `databricks/<model>` (slash). The library handles the reflection LM internally; you pick the format for your `Scorer` instances.
- **Gateway endpoint names cannot start with `databricks-`.** Pick a distinct namespace (e.g. `wanderbricks-*`, `hotpotqa-*`).
- **`optimize_prompts_and_models` always cleans up the `<endpoint>-exp` clones** in a `finally` block, including on exception. Re-running after a clean exit creates fresh clones.

---

## Development

```bash
pip install -e '.[dev]'
pytest
```

All tests are workspace-free; MLflow + AI Gateway calls are mocked.

---

## References

- [GEPA](https://github.com/gepa-ai/gepa) — the reflective prompt/parameter optimizer.
- [MLflow Prompt Registry](https://mlflow.org/docs/latest/genai/prompt-registry/) — versioned prompts with `@alias` lookup.
- [`mlflow.genai.optimize_prompts`](https://mlflow.org/docs/latest/genai/prompt-registry/optimize-prompts/) — the upstream API this library mirrors and extends with joint model selection.
