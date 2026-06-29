# Databricks notebook source
# MAGIC %md
# MAGIC # Prepare HotpotQA Eval Set
# MAGIC
# MAGIC One-off: pulls a slice of the HotpotQA distractor split from HuggingFace
# MAGIC and writes `examples/hotpotqa/eval_set.yaml` in the standard
# MAGIC `{...inputs, expected_answer}` shape every agent dir uses.
# MAGIC Run this once before `notebooks/setup.py` / `notebooks/optimize.py`.

# COMMAND ----------

# MAGIC %pip install pyarrow requests pyyaml -qU

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import io
import os

import pyarrow.parquet as pq
import requests
import yaml

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config

# COMMAND ----------

NUM_SAMPLES = 30
HOTPOT_URL = (
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/"
    "refs%2Fconvert%2Fparquet/distractor/validation/0000.parquet"
)
EVAL_PATH = os.path.join(os.path.dirname(__file__), "eval_set.yaml")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download + reshape

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write

# COMMAND ----------

with open(EVAL_PATH, "w") as f:
    yaml.safe_dump(eval_rows, f)
print(f"Wrote {len(eval_rows)} samples to {EVAL_PATH}")
