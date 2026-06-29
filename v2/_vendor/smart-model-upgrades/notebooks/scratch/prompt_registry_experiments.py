# Databricks notebook source
# MAGIC %md
# MAGIC # Prompt Registry Experiments
# MAGIC
# MAGIC Tests the full lifecycle of MLflow Prompt Registry operations:
# MAGIC register, load, alias, swap, rollback, and live agent verification.
# MAGIC This validates the mechanics the optimization pipeline will use.

# COMMAND ----------

# MAGIC %pip install --index-url https://pypi-proxy.dev.databricks.com/simple -e ../.. databricks-langchain langgraph langchain-core -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), ".."))

import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# COMMAND ----------

UC_NAMESPACE = "users.max_marcussen"

PROMPT_NAMES = {
    "supervisor": f"{UC_NAMESPACE}.wanderbricks_supervisor",
    "query_rewriter": f"{UC_NAMESPACE}.wanderbricks_query_rewriter",
    "enrichment": f"{UC_NAMESPACE}.wanderbricks_enrichment",
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read existing prompts
# MAGIC Load all 3 prompts by `@production` alias. Print template and version.

# COMMAND ----------

original_versions = {}
for key, name in PROMPT_NAMES.items():
    pv = client.get_prompt_version_by_alias(name, "production")
    original_versions[key] = pv.version
    print(f"--- {key} ---")
    print(f"  Name:    {name}")
    print(f"  Version: {pv.version}")
    print(f"  Template (first 200 chars):\n    {pv.template[:200]}...")
    print()

print("Original versions saved:", original_versions)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1b. Latency to retrieve prompts from registry
# MAGIC Measure round-trip time for prompt GET requests. The agent calls
# MAGIC `load_prompt()` on every node invocation, so this adds to each LLM call.

# COMMAND ----------

import time

sup_name = PROMPT_NAMES["supervisor"]

# By alias (requires alias -> version resolution)
alias_times = []
for _ in range(20):
    start = time.perf_counter()
    _ = client.get_prompt_version_by_alias(sup_name, "production")
    alias_times.append((time.perf_counter() - start) * 1000)

print(f"By alias (@production):  min={min(alias_times):.1f} ms, "
      f"median={sorted(alias_times)[10]:.1f} ms, max={max(alias_times):.1f} ms")

# By version number (no alias resolution)
version_times = []
for _ in range(20):
    start = time.perf_counter()
    _ = client.get_prompt_version(sup_name, original_versions["supervisor"])
    version_times.append((time.perf_counter() - start) * 1000)

print(f"By version number:       min={min(version_times):.1f} ms, "
      f"median={sorted(version_times)[10]:.1f} ms, max={max(version_times):.1f} ms")

# COMMAND ----------

# All 3 prompts in sequence (simulates what the agent does per request)
start = time.perf_counter()
for key, name in PROMPT_NAMES.items():
    _ = client.get_prompt_version_by_alias(name, "production")
total_ms = (time.perf_counter() - start) * 1000
print(f"Load all 3 prompts sequentially: {total_ms:.1f} ms")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create a new prompt version
# MAGIC Register a modified supervisor prompt. Confirm version auto-incremented.
# MAGIC Load both old and new by version number to verify they are distinct.

# COMMAND ----------

# Modified supervisor: "brief, friendly" -> "concise, professional",
# plus "Mention how many results were found" added as last bullet.
MODIFIED_SUPERVISOR_TEMPLATE = (
    "You are a travel agent. Answer the user's question using the data you\n"
    "have. Only dispatch a worker when the data gathered so far is clearly\n"
    "insufficient -- prefer FINISH over another round-trip.\n"
    "\n"
    "Write your \"reasoning\" as a concise, professional status update addressed\n"
    "to the user (e.g. \"Searching for properties in Paris.\" or \"Retrieving\n"
    "weather data for August.\"). Keep it to one sentence.\n"
    "\n"
    "USER QUESTION: {{ user_question }}\n"
    "\n"
    "DATA GATHERED SO FAR:\n"
    "{{ scratchpad }}\n"
    "\n"
    "WORKER CALLS SO FAR: {{ calls_used }}\n"
    "\n"
    "WORKERS (use only when needed):\n"
    "- \"query_rewriter\": Search the vacation rental database. Use at most\n"
    "  twice -- once for the main query, once to refine if results are poor.\n"
    "- \"enrichment\": Get weather forecasts. Use once, only if weather is\n"
    "  relevant to the user's question.\n"
    "- \"FINISH\": Write your response. Pick this whenever you have reasonable\n"
    "  data -- do not wait for perfect data.\n"
    "\n"
    "RULES:\n"
    "- Default to FINISH. Only call a worker if the answer truly requires\n"
    "  data you do not have yet.\n"
    "- Never re-query for data you already have in the scratchpad.\n"
    "- If the question is not about travel, pick FINISH and politely decline.\n"
    "- When you pick FINISH, write your response in the \"response\" field:\n"
    "  * Ground every claim in the gathered data -- never invent facts.\n"
    "  * Include specific property names, prices, and ratings when available.\n"
    "  * Weave weather info into recommendations naturally.\n"
    "  * Use a warm, helpful tone and markdown formatting.\n"
    "  * Mention how many results were found.\n"
)

# COMMAND ----------

sup_name = PROMPT_NAMES["supervisor"]

new_pv = client.create_prompt_version(
    name=sup_name,
    template=MODIFIED_SUPERVISOR_TEMPLATE,
    description="Experiment: professional tone + mention result count",
)
print(f"Registered new version: {new_pv.version}")
print(f"Original version was:   {original_versions['supervisor']}")
assert new_pv.version > original_versions["supervisor"], "Version should have incremented"
print("Version increment confirmed.")

# COMMAND ----------

# Load both versions by number to confirm they are distinct.
old_pv = client.get_prompt_version(sup_name, original_versions["supervisor"])
new_pv_check = client.get_prompt_version(sup_name, new_pv.version)

print("Old template contains 'brief, friendly':", "brief, friendly" in old_pv.template)
print("New template contains 'concise, professional':", "concise, professional" in new_pv_check.template)
assert old_pv.template != new_pv_check.template, "Templates should differ"
print("Confirmed: old and new versions are distinct.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Alias management
# MAGIC Create `@staging` on the new version. Verify loading by alias.
# MAGIC Then swap `@production` to the new version.

# COMMAND ----------

client.set_prompt_alias(name=sup_name, alias="staging", version=new_pv.version)
print(f"Set @staging -> version {new_pv.version}")

staging_pv = client.get_prompt_version_by_alias(sup_name, "staging")
print(f"Loaded @staging: version {staging_pv.version}")
assert staging_pv.version == new_pv.version
print("Staging alias verified.")

# COMMAND ----------

prod_before = client.get_prompt_version_by_alias(sup_name, "production")
print(f"@production BEFORE swap: version {prod_before.version}")

client.set_prompt_alias(name=sup_name, alias="production", version=new_pv.version)

prod_after = client.get_prompt_version_by_alias(sup_name, "production")
print(f"@production AFTER swap:  version {prod_after.version}")
assert prod_after.version == new_pv.version
print("Production alias swapped to new version.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Rollback
# MAGIC Point `@production` back to the original version. Verify.

# COMMAND ----------

original_sup_version = original_versions["supervisor"]
client.set_prompt_alias(name=sup_name, alias="production", version=original_sup_version)

rolled_back = client.get_prompt_version_by_alias(sup_name, "production")
print(f"@production after rollback: version {rolled_back.version}")
assert rolled_back.version == original_sup_version
print("Rollback confirmed: @production is back to original version.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Live agent test
# MAGIC Run the same query with original and modified prompts.
# MAGIC The agent loads prompts from `@production` on each invocation,
# MAGIC so swapping the alias mid-notebook changes agent behavior.

# COMMAND ----------

from agent.agent import AGENT

# COMMAND ----------

def ask(question):
    """Run a question through the agent and return the final text."""
    req = {"input": [{"role": "user", "content": question}]}
    response = AGENT.predict(req)
    text = ""
    for item in response.output:
        text += item.content[0]["text"]
    return text

# COMMAND ----------

# Ensure @production points to original
client.set_prompt_alias(name=sup_name, alias="production", version=original_versions["supervisor"])
print(f"@production -> version {original_versions['supervisor']} (original)\n")

TEST_QUESTION = "Find me a place in Paris for 2 people, under $150/night, in August 2026"
result_original = ask(TEST_QUESTION)
print("--- ORIGINAL PROMPT RESULT ---")
print(result_original)

# COMMAND ----------

# Swap @production to modified version
client.set_prompt_alias(name=sup_name, alias="production", version=new_pv.version)
print(f"@production -> version {new_pv.version} (modified)\n")

result_modified = ask(TEST_QUESTION)
print("--- MODIFIED PROMPT RESULT ---")
print(result_modified)

# COMMAND ----------

print("Original response length:", len(result_original))
print("Modified response length:", len(result_modified))
print("Responses are identical:", result_original == result_modified)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Cleanup
# MAGIC Reset `@production` aliases to original versions. Delete `@staging`.

# COMMAND ----------

# Reset all @production aliases to original versions
for key, name in PROMPT_NAMES.items():
    client.set_prompt_alias(name=name, alias="production", version=original_versions[key])
    print(f"Reset {key} @production -> version {original_versions[key]}")

# Delete @staging alias
client.delete_prompt_alias(name=sup_name, alias="staging")
print(f"\nDeleted @staging alias from {sup_name}")

# Verify
print()
for key, name in PROMPT_NAMES.items():
    pv = client.get_prompt_version_by_alias(name, "production")
    print(f"  {key} @production = version {pv.version}")

print("\nCleanup complete.")
