# Databricks notebook source
# MAGIC %md
# MAGIC # 07 - Provision Lakebase & App
# MAGIC
# MAGIC Provisions the infrastructure needed for the agent:
# MAGIC 1. **Lakebase** — Autoscaling Postgres for agent memory (short-term + long-term)
# MAGIC 2. **Databricks App** — Apps compute to host the LangGraph agent
# MAGIC
# MAGIC Both are idempotent — if they already exist, this notebook is a no-op.
# MAGIC
# MAGIC **API note:** All Databricks REST calls go through `WorkspaceClient().api_client.do()`
# MAGIC so that the SDK handles OAuth. The Lakebase `create_project` endpoint requires
# MAGIC `project_id` as a *query parameter* (`?project_id=...`), not in the JSON body.

# COMMAND ----------

dbutils.widgets.text("project_id", "telco-slm-agent-memory", "Lakebase Project ID")
dbutils.widgets.text("app_name",   "otel-telco-agent",       "App Name")
dbutils.widgets.text("catalog",    "",                        "UC Catalog (for SP USE CATALOG grant)")

# COMMAND ----------

import time
from databricks.sdk import WorkspaceClient

project_id = dbutils.widgets.get("project_id") or "telco-slm-agent-memory"
app_name   = dbutils.widgets.get("app_name")   or "otel-telco-agent"

# SDK client with ambient notebook credentials
w = WorkspaceClient()

def api(method, path, body=None, query=None):
    """Call Databricks REST API via SDK (handles OAuth for all endpoints)."""
    return w.api_client.do(method, f"/{path}", body=body, query=query)

print(f"Project : {project_id}")
print(f"App     : {app_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Provision Lakebase

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check if Lakebase project exists

# COMMAND ----------

try:
    project = api("GET", f"api/2.0/postgres/projects/{project_id}")
    print(f"Lakebase project '{project_id}' already exists.")
    status = project.get("status", {})
    print(f"  Owner: {status.get('owner', '')}")
    print(f"  PG Version: {status.get('pg_version', '')}")
    lakebase_exists = True
except Exception as e:
    if "NOT_FOUND" in str(e) or "not found" in str(e).lower():
        print(f"Project '{project_id}' not found. Will create.")
        lakebase_exists = False
    else:
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Lakebase project if needed

# COMMAND ----------

if not lakebase_exists:
    print(f"Creating Lakebase project: {project_id}")
    # project_id is a query parameter; spec goes in the body
    api("POST", "api/2.0/postgres/projects",
        body={"spec": {"display_name": project_id}},
        query={"project_id": project_id})
    print("  Project creation started. Waiting for endpoint to become ACTIVE...")

    for i in range(30):
        time.sleep(10)
        try:
            eps_resp = api("GET", f"api/2.0/postgres/projects/{project_id}/branches/production/endpoints")
            eps = eps_resp if isinstance(eps_resp, list) else eps_resp.get("endpoints", [])
            if eps:
                state = eps[0].get("status", {}).get("current_state", "")
                if state == "ACTIVE":
                    print(f"  Endpoint ACTIVE after {(i+1)*10}s")
                    break
                print(f"  State: {state} (waiting...)")
        except Exception:
            pass
    else:
        print("  WARNING: Endpoint not ACTIVE within 5 min.")
else:
    print("Skipping Lakebase creation — already exists.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verify endpoint and get host

# COMMAND ----------

eps_resp = api("GET", f"api/2.0/postgres/projects/{project_id}/branches/production/endpoints")
eps = eps_resp if isinstance(eps_resp, list) else eps_resp.get("endpoints", [])

if not eps:
    raise RuntimeError("No endpoints found for Lakebase project")

ep      = eps[0]
pg_host  = ep.get("status", {}).get("hosts", {}).get("host", "")
ep_state = ep.get("status", {}).get("current_state", "")
min_cu   = ep.get("status", {}).get("autoscaling_limit_min_cu", "")
max_cu   = ep.get("status", {}).get("autoscaling_limit_max_cu", "")
print(f"Endpoint : {ep_state}")
print(f"  Host   : {pg_host}")
print(f"  Scaling: {min_cu}–{max_cu} CU")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create agent_memory database and tables

# COMMAND ----------

import psycopg2

# Generate OAuth credential via SDK native method (avoids action-path routing issues)
cred = w.postgres.generate_database_credential(
    endpoint=f"projects/{project_id}/branches/production/endpoints/primary"
)
pg_token = cred.token if hasattr(cred, "token") else cred.get("token", "")
user_email = spark.sql("SELECT current_user()").collect()[0][0]

print(f"Connecting to Lakebase as {user_email}...")
conn = psycopg2.connect(host=pg_host, port=5432, dbname="postgres",
                        user=user_email, password=pg_token, sslmode="require")
conn.autocommit = True
cur = conn.cursor()

cur.execute("SELECT 1 FROM pg_database WHERE datname = 'agent_memory'")
if cur.fetchone() is None:
    cur.execute("CREATE DATABASE agent_memory")
    print("  Created database: agent_memory")
else:
    print("  Database 'agent_memory' already exists.")
cur.close()
conn.close()

# COMMAND ----------

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
print("  - conversations")
print("  - messages")
print("  - long_term_memory")

cur.close()
conn.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create dedicated Lakebase access group and grant CAN_USE

# COMMAND ----------

# App SPs are not automatically in the workspace 'users' group, so we create
# a dedicated group for this app, add its SP, and grant the group CAN_USE on
# the Lakebase project. This is idempotent.

from databricks.sdk.service.iam import Patch, PatchOp, PatchSchema

group_name = f"{app_name}-lakebase-users"

# Create group if it doesn't exist
existing_group = None
for g in w.groups.list(filter=f'displayName eq "{group_name}"'):
    existing_group = g
    break

if existing_group is None:
    existing_group = w.groups.create(display_name=group_name)
    print(f"Created group '{group_name}' (id={existing_group.id})")
else:
    print(f"Group '{group_name}' already exists (id={existing_group.id})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Provision Databricks App

# COMMAND ----------

try:
    app_info  = api("GET", f"api/2.0/apps/{app_name}")
    app_state = app_info.get("app_status", {}).get("state", "")
    app_url   = app_info.get("url", "")
    print(f"App '{app_name}' already exists.")
    print(f"  State : {app_state}")
    print(f"  URL   : {app_url}")
    app_exists = True
except Exception as e:
    if "NOT_FOUND" in str(e) or "not found" in str(e).lower():
        print(f"App '{app_name}' not found. Will create.")
        app_exists = False
    else:
        raise

# COMMAND ----------

if not app_exists:
    print(f"Creating Databricks App: {app_name}")
    app_info = api("POST", "api/2.0/apps", body={
        "name": app_name,
        "description": "Telco Network Analytics Agent powered by OTel SLMs + LangGraph",
    })
    print(f"  App created: {app_info.get('name', '')}")
    print(f"  URL: {app_info.get('url', '')}")
    app_url = app_info.get("url", "")
else:
    print("Skipping app creation — already exists.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Add App SP to the group and grant group CAN_USE

# COMMAND ----------

app_info_now   = api("GET", f"api/2.0/apps/{app_name}")
app_sp_id      = str(app_info_now.get("service_principal_id", ""))
app_sp_client_id = app_info_now.get("service_principal_client_id", "")
print(f"App SP id        : {app_sp_id}")
print(f"App SP client_id : {app_sp_client_id}")

if app_sp_id:
    # Add SP to the dedicated group (idempotent — duplicate adds are no-ops)
    w.groups.patch(
        id=existing_group.id,
        operations=[Patch(op=PatchOp.ADD, path="members", value=[{"value": app_sp_id}])],
        schemas=[PatchSchema.URN_IETF_PARAMS_SCIM_API_MESSAGES_2_0_PATCH_OP],
    )
    print(f"Added SP {app_sp_id} to group '{group_name}'")

    # Grant the group CAN_USE on the Lakebase project (workspace IAM layer)
    api("PUT", f"api/2.0/permissions/database-projects/{project_id}", {
        "access_control_list": [
            {"group_name": group_name, "permission_level": "CAN_USE"},
            {"user_name":  user_email, "permission_level": "CAN_MANAGE"},
        ]
    })
    print(f"Granted CAN_USE to group '{group_name}' on '{project_id}'")

    # Create/ensure a Lakebase OAuth role for the SP with databricks_superuser.
    # App SPs are not auto-provisioned as Postgres users — we must register them
    # explicitly with LAKEBASE_OAUTH_V1 auth so the token is accepted at connect time.
    from databricks.sdk.service.postgres import (
        Role as PgRole, RoleRoleSpec, RoleAuthMethod, RoleMembershipRole, RoleIdentityType
    )
    branch_parent = f"projects/{project_id}/branches/production"
    existing_sp_role = None
    for r in w.postgres.list_roles(parent=branch_parent):
        if r.status and r.status.postgres_role == app_sp_client_id:
            existing_sp_role = r
            break

    if existing_sp_role is None:
        w.postgres.create_role(
            parent=branch_parent,
            role=PgRole(spec=RoleRoleSpec(
                auth_method=RoleAuthMethod.LAKEBASE_OAUTH_V1,
                identity_type=RoleIdentityType.SERVICE_PRINCIPAL,
                membership_roles=[RoleMembershipRole.DATABRICKS_SUPERUSER],
                postgres_role=app_sp_client_id,
            )),
        )
        print(f"  Created Lakebase OAuth role for SP '{app_sp_client_id}' with databricks_superuser")
    else:
        print(f"  Lakebase OAuth role already exists for SP '{app_sp_client_id}'")
else:
    print("WARNING: Could not retrieve App SP id — skipping group membership")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("=" * 60)
print("INFRASTRUCTURE PROVISIONING COMPLETE")
print("=" * 60)
print()
print(f"Lakebase Project : {project_id}")
print(f"  Host           : {pg_host}")
print(f"  Database       : agent_memory")
print()
print(f"Databricks App   : {app_name}")
print(f"  URL            : {app_url}")
