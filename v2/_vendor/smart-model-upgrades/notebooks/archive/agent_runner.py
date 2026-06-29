# Databricks notebook source
# MAGIC %md
# MAGIC # WanderBricks Travel Agent -- Runner
# MAGIC Architecture: supervisor loop with tool-calling enrichment
# MAGIC - Supervisor decides next step (query_rewriter / enrichment / FINISH)
# MAGIC - Query rewriter -> Genie for property data
# MAGIC - Enrichment subgraph calls weather tools (standard ToolNode pattern)
# MAGIC - Supervisor writes final response directly when done

# COMMAND ----------

# MAGIC %pip install -e .. databricks-langchain databricks-agents langgraph langchain-core -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), ".."))


# COMMAND ----------

os.environ["DATABRICKS_HOST"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
os.environ["DATABRICKS_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

from agent.agent import model_config, AGENT, graph

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate config

# COMMAND ----------

db = model_config.get("databricks_resources")
gw = model_config.get("gateway_endpoints")
pr = model_config.get("prompt_registry")

assert db["genie_space_id"] != "YOUR_GENIE_SPACE_ID", \
    "Set 'genie_space_id' in config.yaml"
assert db["warehouse_id"] != "YOUR_WAREHOUSE_ID", \
    "Set 'warehouse_id' in config.yaml"

print("Config loaded.")
print(f"  Supervisor endpoint:     {gw['supervisor']['smart_endpoint']}")
print(f"  Query rewriter endpoint: {gw['query_rewriter']['smart_endpoint']}")
print(f"  Enrichment endpoint:     {gw['enrichment']['smart_endpoint']}")
print(f"  Genie space ID:          {db['genie_space_id']}")
print(f"  Max worker rounds:       {model_config.get('max_worker_rounds')}")
print(f"  Prompt registry:")
for key, name in pr.items():
    print(f"    {key}: {name}@production")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize the graph

# COMMAND ----------

from IPython.display import Image, display
from PIL import Image as PILImage
import io

graph_png_bytes = graph.get_graph(xray=True).draw_mermaid_png()
display(Image(graph_png_bytes))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test: predict (non-streaming)

# COMMAND ----------

input_example = model_config.get("input_example")
response = AGENT.predict(input_example)

for item in response.output:
    print(item.content[0]["text"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test: predict_stream (streaming)

# COMMAND ----------

for event in AGENT.predict_stream(input_example):
    if event.item and event.item.get("content"):
        print(event.item["content"][0]["text"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## More example queries

# COMMAND ----------

def ask(question: str):
    """Helper to run a question through predict and print the final output."""
    req = {"input": [{"role": "user", "content": question}]}
    response = AGENT.predict(req)
    for item in response.output:
        print(item.content[0]["text"])


def ask_stream(question: str):
    """Helper to stream all outputs (intermediate thoughts + final answer)."""
    req = {"input": [{"role": "user", "content": question}]}
    for event in AGENT.predict_stream(req):
        if event.type == "response.output_item.done":
            print(event.item["content"][0]["text"])
            print("---")

# COMMAND ----------

ask("Find me a place in Paris for 2 people, under $150/night, in August 2026")

# COMMAND ----------

ask("Compare Berlin vs Tokyo for a family of 4 in December -- need at least 2 bedrooms and a kitchen")

# COMMAND ----------

ask("What's the weather like in London in July?")

# COMMAND ----------

ask("What are the best-rated properties in London?")

# COMMAND ----------

ask("I want a warm beach getaway for under $100/night in January")
