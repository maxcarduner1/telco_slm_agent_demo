# Databricks notebook source

# MAGIC %pip install --index-url https://pypi-proxy.dev.databricks.com/simple -e .. databricks-langchain databricks-agents langgraph langchain-core -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), ".."))


# COMMAND ----------

import json
import yaml

os.environ["MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION"] = "True"

import mlflow
from mlflow.entities import Feedback, SpanType, Trace
from mlflow.genai import scorer
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import RelevanceToQuery

from agent.agent import AGENT

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predict function
# MAGIC Wraps the agent for `mlflow.genai.evaluate()`. Must accept `question` as a
# MAGIC kwarg (matching the `inputs` key in the eval dataset) and return a string.

# COMMAND ----------

def predict_fn(question: str) -> str:
    req = {"input": [{"role": "user", "content": question}]}
    response = AGENT.predict(req)
    if response.output:
        return response.output[0].content[0]["text"]
    return ""

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspect trace spans
# MAGIC Run one query to see what span names and types `mlflow.langchain.autolog()`
# MAGIC generates. Use this to calibrate the custom scorer filters.

# COMMAND ----------

sample = predict_fn("What are the best-rated properties in London?")
trace_id = mlflow.get_last_active_trace_id()
trace = mlflow.get_trace(trace_id) if trace_id else None
if trace:
    for span in trace.data.spans:
        print(f"  {span.name:40s} type={span.span_type}")
    print("\n--- Supervisor span outputs ---")
    for span in trace.data.spans:
        if span.name == "supervisor":
            print(f"\nSpan outputs type: {type(span.outputs)}")
            print(json.dumps(span.outputs, indent=2, default=str)[:2000])
    print("\n--- Genie span outputs ---")
    for span in trace.data.spans:
        if span.name == "genie":
            print(f"\nSpan outputs type: {type(span.outputs)}")
            print(json.dumps(span.outputs, indent=2, default=str)[:2000])
else:
    print("No trace captured -- check that mlflow.langchain.autolog() is enabled.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Custom scorers
# MAGIC Count worker dispatches (genie calls + tool calls) and check against budget.
# MAGIC
# MAGIC If the span names from the cell above don't match the filters below,
# MAGIC update the name checks accordingly.

# COMMAND ----------

def _count_dispatches(trace: Trace) -> tuple[int, int]:
    """Count genie and enrichment node invocations from trace spans."""
    return len(trace.search_spans(name="genie")), len(trace.search_spans(name="enrichment"))


@scorer
def worker_dispatch_count(trace: Trace) -> list[Feedback]:
    """Count genie calls, enrichment calls, and total worker dispatches."""
    genie_calls, enrichment_calls = _count_dispatches(trace)
    total = genie_calls + enrichment_calls
    return [
        Feedback(name="genie_calls", value=genie_calls),
        Feedback(name="enrichment_calls", value=enrichment_calls),
        Feedback(name="worker_dispatches", value=total),
    ]


@scorer
def within_dispatch_budget(trace: Trace, expectations: dict) -> Feedback:
    """Check whether total dispatches stayed within the allowed budget."""
    max_allowed = expectations.get("max_worker_dispatches", 99)
    genie_calls, enrichment_calls = _count_dispatches(trace)
    total = genie_calls + enrichment_calls
    return Feedback(
        name="within_dispatch_budget",
        value=total <= max_allowed,
        rationale=f"Used {total}/{max_allowed} dispatches ({genie_calls} genie, {enrichment_calls} enrichment)",
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Trace-based LLM judges
# MAGIC Uses `make_judge` with `{{ trace }}` -- the judge autonomously explores
# MAGIC spans via MCP tools (ListSpans, GetSpan, SearchTraceRegex).

# COMMAND ----------

narration_tone = make_judge(
    name="narration_tone",
    instructions=(
        "Look at the {{ trace }} for spans named 'supervisor'. "
        "Each has outputs.update.supervisor_reasoning with a short message.\n\n"
        "These messages are shown to the user as status updates. "
        "Return true if they all sound friendly and conversational "
        "(like 'Let me look that up!'). "
        "Return false if any sound like internal notes "
        "(like 'The user wants...' or 'I need to query...')."
    ),
    feedback_value_type=bool,
    model="databricks:/databricks-gpt-5-mini",
)

answer_grounded = make_judge(
    name="answer_grounded",
    instructions=(
        "Analyze the {{ trace }} to determine if the agent's final answer is "
        "grounded in the data it actually retrieved.\n\n"
        "Steps:\n"
        "1. Use ListSpans to find spans named 'genie' and 'enrichment'\n"
        "2. Use GetSpan to inspect their outputs -- look inside "
        "outputs.update.genie_results and outputs.update.enrichment_results "
        "for the actual retrieved data\n"
        "3. Find the last 'supervisor' span (goto=__end__) and inspect "
        "outputs.update.messages for the final answer\n"
        "4. Compare: does the final answer only cite facts from the retrieved data?\n\n"
        "- PASS if all property names, prices, ratings, and weather data in the "
        "answer appear in the retrieved data\n"
        "- FAIL if the answer contains any invented or hallucinated facts"
    ),
    feedback_value_type=bool,
    model="databricks:/databricks-gpt-5-mini",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build eval dataset

# COMMAND ----------

with open(os.path.join(os.getcwd(), "..", "configs", "eval_set.yaml")) as f:
    raw_examples = yaml.safe_load(f)

eval_data = [
    {
        "inputs": {"question": ex["question"]},
        "expectations": {"max_worker_dispatches": ex["max_worker_dispatches"]},
    }
    for ex in raw_examples
]

print(f"Loaded {len(eval_data)} eval examples.")
for i, row in enumerate(eval_data):
    print(f"  {i+1}. {row['inputs']['question'][:80]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run evaluation

# COMMAND ----------

results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=predict_fn,
    scorers=[
        RelevanceToQuery(),
        answer_grounded,
        narration_tone,
        worker_dispatch_count,
        within_dispatch_budget,
    ],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results

# COMMAND ----------

results.metrics

# COMMAND ----------

results.eval_results_table
