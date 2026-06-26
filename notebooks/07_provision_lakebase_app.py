# Databricks notebook source
# MAGIC %md
# MAGIC # 07 - Provision Lakebase & App
# MAGIC
# MAGIC Provisions the infrastructure needed for the agent:
# MAGIC 1. **Lakebase** — Autoscaling Postgres for agent memory (short-term + long-term)
# MAGIC 2. **Databricks App** — Apps compute to host the LangGraph agent
# MAGIC
# MAGIC Both are idempotent — if they already exist, this notebook is a no-op.

# COMMAND ----------

dbutils.widgets.text("project_id", "telco-slm-agent-memory", "Lakebase Project ID")
dbutils.widgets.text("app_name", "otel-telco-agent", "App Name")

# COMMAND ----------

import requests
import time

project_id = dbutils.widgets.get("project_id")
app_name = dbutils.widgets.get("app_name")

ws_url = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

def api(method, path, json_body=None):
    """Call Databricks REST API."""
    url = f"https://{ws_url}/{path}"
    resp = requests.request(method, url, headers=headers, json=json_body, timeout=60)
    return resp

print(f"Workspace: {ws_url}")
print(f"Lakebase project: {project_id}")
print(f"App name: {app_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Provision Lakebase

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check if Lakebase project exists

# COMMAND ----------

resp = api("GET", f"api/2.0/postgres/projects/{project_id}")
if resp.status_code == 200:
    project = resp.json()
    print(f"Lakebase project '{project_id}' already exists.")
    print(f"  Owner: {project.get('status', {}).get('owner', '')}")
    print(f"  PG Version: {project.get('status', {}).get('pg_version', '')}")
    print(f"  Created: {project.get('create_time', '')}")
    lakebase_exists = True
else:
    print(f"Project '{project_id}' not found. Will create.")
    lakebase_exists = False

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Lakebase project if needed

# COMMAND ----------

if not lakebase_exists:
    print(f"Creating Lakebase project: {project_id}")
    create_resp = api("POST", "api/2.0/postgres/projects", {
        "project_id": project_id,
        "spec": {"display_name": project_id}
    })
    if create_resp.status_code in (200, 201):
        print("  Project created. Waiting for endpoint to become ACTIVE...")
    else:
        raise RuntimeError(f"Failed to create project: {create_resp.status_code} {create_resp.text[:500]}")

    # Wait for endpoint
    for i in range(30):
        time.sleep(10)
        ep_resp = api("GET", f"api/2.0/postgres/projects/{project_id}/branches/production/endpoints")
        if ep_resp.status_code == 200:
            endpoints = ep_resp.json()
            if isinstance(endpoints, list) and len(endpoints) > 0:
                state = endpoints[0].get("status", {}).get("current_state", "")
                if state == "ACTIVE":
                    print(f"  Endpoint ACTIVE after {(i+1)*10}s")
                    break
                print(f"  State: {state} (waiting...)")
    else:
        print("  WARNING: Endpoint not ACTIVE within 5 min. Check console.")
else:
    print("Skipping Lakebase creation — already exists.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verify endpoint and get host

# COMMAND ----------

ep_resp = api("GET", f"api/2.0/postgres/projects/{project_id}/branches/production/endpoints")
endpoints = ep_resp.json()
if isinstance(endpoints, list) and len(endpoints) > 0:
    ep = endpoints[0]
    pg_host = ep.get("status", {}).get("hosts", {}).get("host", "")
    ep_state = ep.get("status", {}).get("current_state", "")
    min_cu = ep.get("status", {}).get("autoscaling_limit_min_cu", "")
    max_cu = ep.get("status", {}).get("autoscaling_limit_max_cu", "")
    print(f"Endpoint: {ep_state}")
    print(f"  Host: {pg_host}")
    print(f"  Scaling: {min_cu}–{max_cu} CU")
else:
    raise RuntimeError("No endpoints found for Lakebase project")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create agent_memory database and tables

# COMMAND ----------

import psycopg2

# Generate OAuth credential
cred_resp = api("POST", f"api/2.0/postgres/projects/{project_id}/branches/production/endpoints/primary:generateCredential", {})
if cred_resp.status_code != 200:
    raise RuntimeError(f"Failed to generate credential: {cred_resp.status_code} {cred_resp.text[:300]}")

pg_token = cred_resp.json().get("token", "")
user_email = spark.sql("SELECT current_user()").collect()[0][0]

print(f"Connecting to Lakebase as {user_email}...")
conn = psycopg2.connect(host=pg_host, port=5432, dbname="postgres",
                        user=user_email, password=pg_token, sslmode="require")
conn.autocommit = True
cur = conn.cursor()

# Create database if not exists
cur.execute("SELECT 1 FROM pg_database WHERE datname = 'agent_memory'")
if cur.fetchone() is None:
    cur.execute("CREATE DATABASE agent_memory")
    print("  Created database: agent_memory")
else:
    print("  Database 'agent_memory' already exists.")
cur.close()
conn.close()

# COMMAND ----------

# Connect to agent_memory and create schema
conn = psycopg2.connect(host=pg_host, port=5432, dbname="agent_memory",
                        user=user_email, password=pg_token, sslmode="require")
conn.autocommit = True
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS conversations (
    thread_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    thread_id TEXT REFERENCES conversations(thread_id),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tool_calls JSONB,
    created_at TIMESTAMP DEFAULT NOW()
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS long_term_memory (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    category TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    last_accessed TIMESTAMP DEFAULT NOW(),
    access_count INT DEFAULT 0
)
""")

cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id, created_at)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_ltm_user ON long_term_memory(user_id, category)")

print("Memory tables ready:")
print("  - conversations (thread metadata)")
print("  - messages (chat history)")
print("  - long_term_memory (persistent facts)")

cur.close()
conn.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Provision Databricks App

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check if app exists

# COMMAND ----------

app_resp = api("GET", f"api/2.0/apps/{app_name}")
if app_resp.status_code == 200:
    app_info = app_resp.json()
    app_state = app_info.get("app_status", {}).get("state", "")
    app_url = app_info.get("url", "")
    print(f"App '{app_name}' already exists.")
    print(f"  State: {app_state}")
    print(f"  URL: {app_url}")
    app_exists = True
else:
    print(f"App '{app_name}' not found. Will create.")
    app_exists = False

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create app if needed

# COMMAND ----------

if not app_exists:
    print(f"Creating Databricks App: {app_name}")
    create_resp = api("POST", "api/2.0/apps", {
        "name": app_name,
        "description": "Telco Network Analytics Agent powered by OTel SLMs + LangGraph",
    })
    if create_resp.status_code in (200, 201):
        app_info = create_resp.json()
        print(f"  App created: {app_info.get('name', '')}")
        print(f"  URL: {app_info.get('url', '')}")
        print(f"  Compute starting...")
    else:
        raise RuntimeError(f"Failed to create app: {create_resp.status_code} {create_resp.text[:500]}")
else:
    print("Skipping app creation — already exists.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("=" * 60)
print("INFRASTRUCTURE PROVISIONING COMPLETE")
print("=" * 60)
print()
print(f"Lakebase Project: {project_id}")
print(f"  Host: {pg_host}")
print(f"  Database: agent_memory")
print()
print(f"Databricks App: {app_name}")
if app_exists:
    print(f"  URL: {app_url}")
print()
print("Next steps:")
print("  1. Deploy agent code: databricks apps deploy otel-telco-agent --source-code-path ./")
print("  2. Or via DAB: databricks bundle deploy")
