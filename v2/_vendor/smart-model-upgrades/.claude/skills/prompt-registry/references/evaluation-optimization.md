# Prompt Evaluation and Optimization

## Table of Contents

1. [Evaluation Overview](#evaluation-overview)
2. [Scorer Types](#scorer-types)
3. [Running an Evaluation](#running-an-evaluation)
4. [Training Data Structure](#training-data-structure)
5. [GEPA Prompt Optimization](#gepa-prompt-optimization)
6. [Comparing Baseline vs Optimized](#comparing-baseline-vs-optimized)
7. [Promoting the Winner](#promoting-the-winner)
8. [Best Practices](#best-practices)

---

## Evaluation Overview

MLflow's evaluation framework lets you systematically compare prompt versions using a
combination of code-based checks and LLM judges. The workflow is:

1. Define a `predict_fn` that loads a prompt and calls your LLM
2. Prepare evaluation data with inputs and expectations
3. Define scorers (code-based, LLM judges, or built-in)
4. Run `mlflow.genai.evaluate()` to get metrics
5. Compare across prompt versions

---

## Scorer Types

### Code-based scorers (`@scorer`)

Deterministic checks you write in Python. Fast, free, and reliable for structural validation.

```python
from mlflow.genai.scorers import scorer
from mlflow.entities import Feedback

@scorer
def response_structure(outputs) -> list:
    """Check required structural elements in the response."""
    text = str(outputs).lower()
    feedbacks = []

    has_greeting = any(w in text for w in ["hello", "dear", "thank you for"])
    feedbacks.append(Feedback(
        name="has_greeting",
        value=has_greeting,
        rationale="Found greeting" if has_greeting else "Missing greeting",
    ))

    has_closing = any(w in text for w in ["sincerely", "regards", "best wishes"])
    feedbacks.append(Feedback(
        name="has_closing",
        value=has_closing,
        rationale="Found closing" if has_closing else "Missing professional closing",
    ))

    return feedbacks
```

Each `Feedback` has:
- `name`: metric name (appears in results)
- `value`: boolean or numeric score
- `rationale`: explanation of the score

### LLM judges (`make_judge`)

Custom evaluation criteria assessed by a language model. Use for nuanced checks that
are hard to capture with keyword matching.

```python
from mlflow.genai.judges import make_judge

tone_judge = make_judge(
    name="tone_compliance",
    instructions="""
    Evaluate the response for tone and policy compliance.

    Input: {{ inputs }}
    Response: {{ outputs }}

    Check ALL of the following:
    1. Tone is warm, professional, and solution-oriented
    2. No legal jargon or confrontational language
    3. Empathy is expressed without admitting fault
    4. At least one specific next-step or timeline is mentioned

    Respond with exactly 'yes' if ALL criteria are met, or 'no' if any fail.
    """,
    model="databricks:/databricks-gpt-5",
)
```

The `{{ inputs }}` and `{{ outputs }}` placeholders are filled automatically by the
evaluation framework.

### Built-in scorers

MLflow provides ready-to-use scorers for common evaluation needs:

```python
from mlflow.genai.scorers import Correctness, Guidelines, Safety

# Checks if expected facts appear in the output
correctness = Correctness(model="databricks:/databricks-gpt-5")

# Checks if output follows custom guidelines
length_check = Guidelines(
    name="response_length",
    guidelines="The response must be between 150 to 350 words.",
    model="databricks:/databricks-gpt-5",
)

# Checks for harmful or unsafe content
safety = Safety(model="databricks:/databricks-gpt-5")
```

`Correctness` uses the `expected_facts` field from your evaluation data to verify factual
coverage.

---

## Running an Evaluation

```python
import mlflow

scorers = [response_structure, tone_judge, length_check, safety]

with mlflow.start_run(run_name="baseline-v1"):
    results = mlflow.genai.evaluate(
        data=eval_data,          # list of dicts (see Training Data Structure)
        predict_fn=predict_fn,   # function that takes inputs and returns a string
        scorers=scorers,
    )

# Print aggregated metrics
for metric, value in sorted(results.metrics.items()):
    if isinstance(value, float):
        print(f"  {metric}: {value:.4f}")
    else:
        print(f"  {metric}: {value}")
```

Each scorer produces metrics with `/mean`, `/min`, `/max` suffixes in the results.

---

## Training Data Structure

Both evaluation and optimization expect data in this format:

```python
data = [
    {
        "inputs": {
            "complaint": "I was charged twice for my policy renewal…",
        },
        "expectations": {
            "expected_facts": [
                "Response acknowledges the customer's concern",
                "Response includes an apology or empathetic statement",
                "Response describes resolution steps or next actions",
                "Response provides contact information for follow-up",
                "Response ends with a professional closing",
            ],
        },
    },
    # … more examples
]
```

- **`inputs`**: dict of template variable names → values. These are passed to `predict_fn`.
- **`expectations`**: dict containing `expected_facts` (list of strings). Used by `Correctness`
  scorer to check factual coverage.

### Extracting expected facts from gold responses

If you have gold-standard responses, you can extract structural signals programmatically:

```python
def extract_expected_facts(gold_response: str) -> list:
    """Extract quality signals from a reference response."""
    facts = []
    lower = gold_response.lower()

    if any(w in lower for w in ["understand", "acknowledge", "received"]):
        facts.append("Response acknowledges the customer's concern")
    if any(w in lower for w in ["apologize", "sorry", "regret"]):
        facts.append("Response includes an apology or empathetic statement")
    if any(w in lower for w in ["will", "next step", "follow up", "resolve"]):
        facts.append("Response describes resolution steps or next actions")
    if any(w in lower for w in ["contact", "phone", "email", "call"]):
        facts.append("Response provides contact information for follow-up")
    if any(w in lower for w in ["sincerely", "regards", "respectfully"]):
        facts.append("Response ends with a professional closing")

    if not facts:
        facts.append("Response is a complete, professional response")
    return facts
```

### Splitting data

A good practice is to split data into training (for optimization) and held-out evaluation
(for unbiased comparison):

```python
train_data = all_data[:16]   # used by GEPA optimizer
eval_data  = all_data[16:]   # used for before/after comparison
```

---

## GEPA Prompt Optimization

GEPA (Generative Evaluation-driven Prompt Augmentation) automatically improves your prompt
using a frontier model for reflection while targeting your production model for inference.

```python
from mlflow.genai.optimize import GepaPromptOptimizer

# Use a focused subset of scorers for optimization (keeps cost lower)
optimization_scorers = [tone_judge, safety]

result = mlflow.genai.optimize_prompts(
    predict_fn=predict_fn,               # your predict function
    train_data=train_data,               # training examples with inputs + expectations
    prompt_uris=[prompt.uri],            # URI of the prompt to optimize
    optimizer=GepaPromptOptimizer(
        reflection_model="databricks:/databricks-claude-sonnet-4",
    ),
    scorers=optimization_scorers,
)

optimized_prompt = result.optimized_prompts[0]
print(f"Optimized version: {optimized_prompt.version}")
print(f"Template:\n{optimized_prompt.template}")
```

Key points:
- **`predict_fn`** should load the prompt dynamically (by latest version) so it automatically
  picks up the optimized version when GEPA creates it.
- **`reflection_model`** is the frontier model that analyzes failures and proposes improvements.
  Use a strong model here (Claude Sonnet 4, GPT-5, etc.) even if your target model is smaller.
- **`prompt_uris`** takes a list — you can optimize multiple prompts in one call.
- **Scorer selection**: use fewer scorers for optimization than for evaluation. This keeps
  the optimization signal focused and reduces cost. Add the remaining scorers back for the
  post-optimization comparison.

---

## Comparing Baseline vs Optimized

After optimization, re-run evaluation on the held-out set and compare:

```python
import pandas as pd

# Baseline (run before optimization)
with mlflow.start_run(run_name="baseline"):
    baseline_results = mlflow.genai.evaluate(
        data=eval_data, predict_fn=predict_fn, scorers=all_scorers,
    )

# Optimized (predict_fn now loads the new version)
with mlflow.start_run(run_name="optimized"):
    optimized_results = mlflow.genai.evaluate(
        data=eval_data, predict_fn=predict_fn, scorers=all_scorers,
    )

# Build comparison table
comparison = []
all_metrics = set(baseline_results.metrics) | set(optimized_results.metrics)
for metric in sorted(all_metrics):
    if "/mean" in metric:
        b = baseline_results.metrics.get(metric, 0)
        o = optimized_results.metrics.get(metric, 0)
        comparison.append({
            "Metric": metric.replace("/mean", ""),
            "Baseline": f"{b:.3f}",
            "Optimized": f"{o:.3f}",
            "Delta": f"{o - b:+.3f}",
        })

print(pd.DataFrame(comparison).to_string(index=False))
```

---

## Promoting the Winner

Once the optimized version shows improvement on the held-out eval set:

```python
mlflow.genai.set_prompt_alias(
    name="catalog.schema.my_prompt",
    alias="production",
    version=optimized_prompt.version,
)
print(f"Promoted v{optimized_prompt.version} to production")
```

If you need to grant a service principal access for a deployed app:

```sql
GRANT CREATE FUNCTION, EXECUTE, MANAGE
ON SCHEMA catalog.schema
TO `<service-principal-id>`;
```

---

## Best Practices

1. **Start simple** — begin with a basic prompt and iteratively improve based on eval results.
2. **Consistent datasets** — use the same eval data across all versions for fair comparison.
3. **Track everything** — use `mlflow.start_run()` with descriptive names for each evaluation.
4. **Include hard examples** — put challenging edge cases in your eval set.
5. **Continue evaluating after deployment** — prompt quality can degrade as user behavior changes.
6. **Meaningful commit messages** — document what changed and why in every version.
7. **Use composite scoring** — weight multiple metrics to get a single ranking signal:
   ```python
   composite = 0.7 * correctness_score + 0.3 * compliance_score
   ```
