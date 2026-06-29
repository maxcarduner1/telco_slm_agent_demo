# Databricks notebook source
# MAGIC %md
# MAGIC # 08 - Grant App Unity Catalog Permissions
# MAGIC
# MAGIC Grants UC object permissions directly to a Databricks App service principal.
# MAGIC Run this after UC functions and Vector Search indexes exist, and after the
# MAGIC Databricks App has been created so its service principal can be resolved.

# COMMAND ----------

dbutils.widgets.text("catalog", "cmegdemos_catalog", "UC Catalog")
dbutils.widgets.text("schema", "network_analytics_enablement", "UC Schema")
dbutils.widgets.text("app_name", "otel-telco-agent", "Databricks App Name")
dbutils.widgets.text("prompt_name", "", "Optional MLflow Prompt Registry object name")

# COMMAND ----------

from databricks.sdk import WorkspaceClient

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
app_name = dbutils.widgets.get("app_name")
prompt_name = dbutils.widgets.get("prompt_name")

print(f"Catalog : {catalog}")
print(f"Schema  : {schema}")
print(f"App     : {app_name}")
if prompt_name:
    print(f"Prompt  : {prompt_name}")

w = WorkspaceClient()
app_info = w.apps.get(name=app_name)
app_sp_client_id = app_info.service_principal_client_id

if not app_sp_client_id:
    raise RuntimeError(f"Could not resolve service principal client id for app '{app_name}'")

print(f"App SP client id: {app_sp_client_id}")

# COMMAND ----------

# UC grants are issued directly to the App service principal client ID because
# not all workspaces resolve IAM groups as UC SQL principals consistently.
# TODO: Optional future enhancement — support OBO/user-delegated permissioning.
quoted_schema = f"`{catalog}`.`{schema}`"
uc_statements = [
    f"GRANT USE CATALOG ON CATALOG `{catalog}` TO `{app_sp_client_id}`",
    f"GRANT USE SCHEMA ON SCHEMA {quoted_schema} TO `{app_sp_client_id}`",
    f"GRANT EXECUTE ON SCHEMA {quoted_schema} TO `{app_sp_client_id}`",
    f"GRANT SELECT ON TABLE {quoted_schema}.`otel_runbooks_vs_index` TO `{app_sp_client_id}`",
    f"GRANT SELECT ON TABLE {quoted_schema}.`otel_standards_vs_index` TO `{app_sp_client_id}`",
    f"GRANT SELECT ON TABLE {quoted_schema}.`otel_incidents_vs_index` TO `{app_sp_client_id}`",
]

for stmt in uc_statements:
    spark.sql(stmt)
    print(f"Applied UC grant: {stmt}")

if prompt_name.strip():
    prompt_fqn = f"{catalog}.{schema}.{prompt_name.strip()}"
    prompt_schema_grants = [
        f"GRANT CREATE FUNCTION ON SCHEMA {quoted_schema} TO `{app_sp_client_id}`",
        f"GRANT MANAGE ON SCHEMA {quoted_schema} TO `{app_sp_client_id}`",
    ]
    for stmt in prompt_schema_grants:
        try:
            spark.sql(stmt)
            print(f"Applied prompt schema grant: {stmt}")
        except Exception as e:
            print(f"WARNING: Could not apply prompt schema grant '{stmt}': {e}")

    prompt_grants = [
        f"GRANT EXECUTE ON MODEL `{catalog}`.`{schema}`.`{prompt_name.strip()}` TO `{app_sp_client_id}`",
        f"GRANT SELECT ON MODEL `{catalog}`.`{schema}`.`{prompt_name.strip()}` TO `{app_sp_client_id}`",
    ]
    for stmt in prompt_grants:
        try:
            spark.sql(stmt)
            print(f"Applied prompt grant: {stmt}")
        except Exception as e:
            print(f"WARNING: Could not apply prompt grant '{stmt}': {e}")

print("UC permission grants complete.")
