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
dbutils.widgets.text("ai_gateway_endpoint", "", "Optional AI Gateway endpoint name")
dbutils.widgets.text("embedding_endpoint", "otel-embedding2-300m", "Embedding Serving Endpoint")
dbutils.widgets.text("reranker_endpoint", "otel-reranker-600m", "Reranker Serving Endpoint")
dbutils.widgets.text("vs_endpoint", "demo_telco_vs_endpoint", "Vector Search Endpoint")

# COMMAND ----------

from databricks.sdk import WorkspaceClient

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
app_name = dbutils.widgets.get("app_name")
prompt_name = dbutils.widgets.get("prompt_name")
ai_gateway_endpoint = dbutils.widgets.get("ai_gateway_endpoint")
embedding_endpoint = dbutils.widgets.get("embedding_endpoint")
reranker_endpoint = dbutils.widgets.get("reranker_endpoint")
vs_endpoint = dbutils.widgets.get("vs_endpoint")

print(f"Catalog : {catalog}")
print(f"Schema  : {schema}")
print(f"App     : {app_name}")
if prompt_name:
    print(f"Prompt  : {prompt_name}")
if ai_gateway_endpoint:
    print(f"Gateway : {ai_gateway_endpoint}")
print(f"Embedding endpoint: {embedding_endpoint}")
print(f"Reranker endpoint : {reranker_endpoint}")
print(f"VS endpoint       : {vs_endpoint}")

w = WorkspaceClient()
app_info = w.apps.get(name=app_name)
app_sp_client_id = app_info.service_principal_client_id

if not app_sp_client_id:
    raise RuntimeError(f"Could not resolve service principal client id for app '{app_name}'")

print(f"App SP client id: {app_sp_client_id}")
group_name = f"{app_name}-lakebase-users"


def _service_principal_acl(permission_level):
    return [
        {"service_principal_name": app_sp_client_id, "permission_level": permission_level},
        {"group_name": group_name, "permission_level": permission_level},
        {"user_name": w.current_user.me().user_name, "permission_level": "CAN_MANAGE"},
    ]


def _patch_permissions(resource_path, acl):
    w.api_client.do(
        "PATCH",
        f"/api/2.0/permissions/{resource_path}",
        body={"access_control_list": acl},
    )

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

if ai_gateway_endpoint.strip():
    try:
        _patch_permissions(
            f"ai-gateway-endpoints/{ai_gateway_endpoint.strip()}",
            _service_principal_acl("CAN_QUERY"),
        )
        print(f"Applied AI Gateway CAN_QUERY grants on '{ai_gateway_endpoint}'")
    except Exception as e:
        print(f"WARNING: Could not apply AI Gateway grants on '{ai_gateway_endpoint}': {e}")

serving_by_name = {}
try:
    serving_resp = w.api_client.do("GET", "/api/2.0/serving-endpoints")
    for endpoint in serving_resp.get("endpoints", []):
        name = endpoint.get("name")
        endpoint_id = endpoint.get("id")
        if name and endpoint_id:
            serving_by_name[name] = endpoint_id
except Exception as e:
    print(f"WARNING: Could not list serving endpoints for permission grants: {e}")

for endpoint_name in [embedding_endpoint.strip(), reranker_endpoint.strip()]:
    if not endpoint_name:
        continue
    endpoint_id = serving_by_name.get(endpoint_name)
    if not endpoint_id:
        print(f"WARNING: Could not resolve serving endpoint ID for '{endpoint_name}'")
        continue
    try:
        _patch_permissions(
            f"serving-endpoints/{endpoint_id}",
            _service_principal_acl("CAN_QUERY"),
        )
        print(f"Applied serving endpoint CAN_QUERY grants on '{endpoint_name}'")
    except Exception as e:
        print(f"WARNING: Could not apply serving endpoint grants on '{endpoint_name}': {e}")

vs_id = None
if vs_endpoint.strip():
    try:
        vs_resp = w.api_client.do("GET", "/api/2.0/vector-search/endpoints")
        for endpoint in vs_resp.get("endpoints", []):
            if endpoint.get("name") == vs_endpoint.strip():
                vs_id = endpoint.get("id")
                break
    except Exception as e:
        print(f"WARNING: Could not list Vector Search endpoints for permission grants: {e}")

    if not vs_id:
        print(f"WARNING: Could not resolve Vector Search endpoint ID for '{vs_endpoint}'")
    else:
        try:
            _patch_permissions(
                f"vector-search-endpoints/{vs_id}",
                _service_principal_acl("CAN_USE"),
            )
            print(f"Applied Vector Search CAN_USE grants on '{vs_endpoint}'")
        except Exception as e:
            print(f"WARNING: Could not apply Vector Search grants on '{vs_endpoint}': {e}")

print("App permission grants complete.")
