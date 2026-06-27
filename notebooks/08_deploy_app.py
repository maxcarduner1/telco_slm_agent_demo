# Databricks notebook source
# MAGIC %md
# MAGIC # 08 — Deploy App
# MAGIC
# MAGIC Deploys the agent source code to the Databricks App provisioned in `07`.
# MAGIC
# MAGIC **Dependencies (all must complete before this task runs):**
# MAGIC - `03_parse_documents` — document chunks loaded into Delta
# MAGIC - `04_create_vs_indexes` — Vector Search indexes built and synced
# MAGIC - `05_create_uc_functions` — UC SQL tools registered
# MAGIC - `07_provision_lakebase_app` — Lakebase project and App compute ready
# MAGIC
# MAGIC Idempotent — triggers a new deployment and waits for `SUCCEEDED`. Re-running
# MAGIC rolls the app forward to the latest source revision.

# COMMAND ----------

dbutils.widgets.text("app_name",         "otel-telco-agent", "App Name")
dbutils.widgets.text("source_code_path", ".",                "Source Code Path")
dbutils.widgets.text("catalog",          "",                 "UC Catalog")
dbutils.widgets.text("schema",           "",                 "UC Schema")
dbutils.widgets.text("warehouse_id",     "",                 "SQL Warehouse ID")

# COMMAND ----------

import requests
import time

app_name         = dbutils.widgets.get("app_name")         or "otel-telco-agent"
source_code_path = dbutils.widgets.get("source_code_path") or "."
catalog          = dbutils.widgets.get("catalog")
schema           = dbutils.widgets.get("schema")
warehouse_id     = dbutils.widgets.get("warehouse_id")

ws_url = spark.conf.get("spark.databricks.workspaceUrl")
token  = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

def api(method, path, json_body=None):
    url  = f"https://{ws_url}/{path}"
    resp = requests.request(method, url, headers=headers, json=json_body, timeout=60)
    return resp

print(f"Workspace : {ws_url}")
print(f"App name  : {app_name}")
print(f"Source    : {source_code_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Verify app exists

# COMMAND ----------

resp = api("GET", f"api/2.0/apps/{app_name}")
if resp.status_code != 200:
    raise RuntimeError(
        f"App '{app_name}' not found (HTTP {resp.status_code}). "
        "Run 07_provision_lakebase_app first."
    )

app_info = resp.json()
compute_state = app_info.get("compute_status", {}).get("state", "")
app_state     = app_info.get("app_status",     {}).get("state", "")
print(f"App '{app_name}' found.")
print(f"  Compute : {compute_state}")
print(f"  App     : {app_state}")
print(f"  URL     : {app_info.get('url', '')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1a. Patch workspace-specific env vars into the app config

# COMMAND ----------

# Bundle variables (catalog, schema, warehouse_id) are passed as base_parameters
# from databricks.yml and injected here so they take effect at app startup.
# This bridges the gap between databricks.yml variables and app.yaml (which has
# no native variable interpolation support).

env_overrides = {}
if catalog:      env_overrides["UC_CATALOG"]             = catalog
if schema:       env_overrides["UC_SCHEMA"]              = schema
if warehouse_id: env_overrides["DATABRICKS_WAREHOUSE_ID"] = warehouse_id

if env_overrides:
    current_env = app_info.get("config", {}).get("env", []) if isinstance(app_info.get("config"), dict) else []
    # Merge: keep existing vars not in overrides, then add overrides
    env_map = {e["name"]: e["value"] for e in current_env if isinstance(e, dict) and "name" in e}
    env_map.update(env_overrides)
    new_env = [{"name": k, "value": v} for k, v in env_map.items()]

    patch_resp = api("PATCH", f"api/2.0/apps/{app_name}", {
        "config": {"env": new_env}
    })
    if patch_resp.status_code in (200, 201):
        print(f"Patched app env vars: {list(env_overrides.keys())}")
    else:
        print(f"WARNING: Could not patch env vars ({patch_resp.status_code}): {patch_resp.text[:200]}")
else:
    print("No env overrides provided — skipping env patch.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Start app compute if stopped

# COMMAND ----------

if compute_state not in ("ACTIVE", "RUNNING"):
    print("App compute not running — starting...")
    start_resp = api("POST", f"api/2.0/apps/{app_name}/start", {})
    if start_resp.status_code not in (200, 201):
        raise RuntimeError(
            f"Failed to start app: {start_resp.status_code} {start_resp.text[:300]}"
        )
    # Wait for ACTIVE
    for i in range(30):
        time.sleep(10)
        r = api("GET", f"api/2.0/apps/{app_name}")
        if r.status_code == 200:
            cs = r.json().get("compute_status", {}).get("state", "")
            print(f"  [{(i+1)*10}s] compute state: {cs}")
            if cs in ("ACTIVE", "RUNNING"):
                print("  Compute ready.")
                break
    else:
        raise RuntimeError("App compute did not become ACTIVE within 5 min.")
else:
    print("App compute already running — skipping start.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Trigger deployment

# COMMAND ----------

deploy_resp = api("POST", f"api/2.0/apps/{app_name}/deployments", {
    "source_code_path": source_code_path,
})

if deploy_resp.status_code not in (200, 201):
    raise RuntimeError(
        f"Failed to start deployment: {deploy_resp.status_code} {deploy_resp.text[:500]}"
    )

deployment    = deploy_resp.json()
deployment_id = deployment.get("deployment_id", "")
print(f"Deployment started: {deployment_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Wait for deployment to complete

# COMMAND ----------

MAX_WAIT_S  = 600   # 10 min
POLL_EVERY  = 15

elapsed = 0
final_state = None

while elapsed < MAX_WAIT_S:
    time.sleep(POLL_EVERY)
    elapsed += POLL_EVERY

    poll = api("GET", f"api/2.0/apps/{app_name}/deployments/{deployment_id}")
    if poll.status_code != 200:
        print(f"  [{elapsed}s] poll error {poll.status_code}, retrying...")
        continue

    d     = poll.json()
    state = d.get("status", {}).get("state", "UNKNOWN")
    msg   = d.get("status", {}).get("message", "")
    print(f"  [{elapsed}s] {state}" + (f" — {msg}" if msg else ""))

    if state == "SUCCEEDED":
        final_state = "SUCCEEDED"
        break
    elif state in ("FAILED", "CANCELLED"):
        final_state = state
        break

if final_state != "SUCCEEDED":
    raise RuntimeError(f"Deployment did not succeed (final state: {final_state}). Check app logs.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Summary

# COMMAND ----------

app_resp = api("GET", f"api/2.0/apps/{app_name}")
app_url  = app_resp.json().get("url", "") if app_resp.status_code == 200 else ""

print("=" * 60)
print("DEPLOYMENT COMPLETE")
print("=" * 60)
print()
print(f"App   : {app_name}")
print(f"URL   : {app_url}")
print(f"Build : {deployment_id}")
print()
print("The agent is live. Open the URL above to start chatting.")
