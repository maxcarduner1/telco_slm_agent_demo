# Databricks notebook source
# MAGIC %md
# MAGIC # Setup -- Register Prompts + Create Gateway Endpoints
# MAGIC
# MAGIC One-time bootstrap per agent. Idempotent -- re-running cuts a new prompt
# MAGIC version and leaves matching endpoints alone.
# MAGIC
# MAGIC Pick an agent from the widget, then run all cells. Each entry under
# MAGIC `examples/` with an `agent_config.yaml` is selectable.

# COMMAND ----------

# MAGIC %pip install -e .. -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os

import mlflow
import yaml

import smart_model_upgrades as smu

os.environ["DATABRICKS_HOST"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
os.environ["DATABRICKS_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

EXAMPLES_DIR = os.path.normpath(os.path.join(os.getcwd(), "..", "examples"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pick an agent

# COMMAND ----------

AGENT_CHOICES = sorted(
    name for name in os.listdir(EXAMPLES_DIR)
    if os.path.isfile(os.path.join(EXAMPLES_DIR, name, "agent_config.yaml"))
)

dbutils.widgets.dropdown(
    name="agent",
    defaultValue="wanderbricks" if "wanderbricks" in AGENT_CHOICES else AGENT_CHOICES[0],
    choices=AGENT_CHOICES,
    label="Agent",
)
AGENT_TAG = dbutils.widgets.get("agent")
AGENT_DIR = os.path.join(EXAMPLES_DIR, AGENT_TAG)

with open(os.path.join(AGENT_DIR, "agent_config.yaml")) as f:
    agent_cfg = yaml.safe_load(f)
with open(os.path.join(AGENT_DIR, "seed_prompts.yaml")) as f:
    templates = yaml.safe_load(f)

print(f"Using agent: {AGENT_TAG}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register seed prompts at @production

# COMMAND ----------

for comp, full_name in agent_cfg["prompt_registry"].items():
    pv = mlflow.genai.register_prompt(
        name=full_name,
        template=templates[comp],
        commit_message=f"Initial registration of {comp}",
    )
    mlflow.genai.set_prompt_alias(name=full_name, alias="production", version=pv.version)
    print(f"{full_name}@production = v{pv.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create production gateway endpoints

# COMMAND ----------

endpoints = {
    cfg["smart_endpoint"]: cfg["initial_model"]
    for cfg in agent_cfg["gateway_endpoints"].values()
}
statuses = smu.setup_endpoints(endpoints, agent_tag=AGENT_TAG)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify

# COMMAND ----------

print("Prompts:")
for comp, full_name in agent_cfg["prompt_registry"].items():
    pv = mlflow.genai.load_prompt(f"prompts:/{full_name}@production")
    print(f"  {comp}: {full_name}@production (v{pv.version}, {len(pv.template)} chars)")

print("\nEndpoints:")
for name, status in statuses.items():
    print(f"  {name}: {status}")
