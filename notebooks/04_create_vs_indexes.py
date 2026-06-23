# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Create Vector Search Indexes
# MAGIC
# MAGIC Creates 3 Vector Search indexes (runbooks, standards, incidents) on a shared endpoint,
# MAGIC configured for auto-sync from the chunk Delta tables.

# COMMAND ----------

dbutils.widgets.text("catalog", "cmegdemos_catalog", "Catalog")
dbutils.widgets.text("schema", "network_analytics_enablement", "Schema")
dbutils.widgets.text("vs_endpoint", "demo_telco_vs_endpoint", "VS Endpoint Name")

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
vs_endpoint_name = dbutils.widgets.get("vs_endpoint")

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find or create VS endpoint

# COMMAND ----------

# Use the specified endpoint (default: demo_telco_vs_endpoint)
if not vs_endpoint_name:
    vs_endpoint_name = "demo_telco_vs_endpoint"

print(f"Using VS endpoint: {vs_endpoint_name}")

# Verify endpoint exists and is online
try:
    ep_info = vsc.get_endpoint(vs_endpoint_name)
    state = ep_info.get("endpoint_status", {}).get("state", "UNKNOWN")
    print(f"  Status: {state}")
except Exception as e:
    print(f"  Warning: Could not verify endpoint - {e}")
    print(f"  Proceeding anyway (endpoint may still be usable)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Vector Search Indexes
# MAGIC
# MAGIC Using Delta Sync mode with auto-sync from chunk tables.
# MAGIC Embedding will be computed by the OTel-Embedding-335M model serving endpoint.

# COMMAND ----------

# Index configurations
indexes = [
    {
        "index_name": f"{catalog}.{schema}.runbooks_vs_index",
        "source_table": f"{catalog}.{schema}.telco_docs_runbooks_chunks",
        "primary_key": "chunk_id",
        "text_column": "chunk_text",
    },
    {
        "index_name": f"{catalog}.{schema}.standards_vs_index",
        "source_table": f"{catalog}.{schema}.telco_docs_standards_chunks",
        "primary_key": "chunk_id",
        "text_column": "chunk_text",
    },
    {
        "index_name": f"{catalog}.{schema}.incidents_vs_index",
        "source_table": f"{catalog}.{schema}.telco_docs_incidents_chunks",
        "primary_key": "chunk_id",
        "text_column": "chunk_text",
    },
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Note on embedding model
# MAGIC
# MAGIC These indexes use `databricks-bge-large-en` as the embedding model for initial setup.
# MAGIC Once the OTel-Embedding-335M model is deployed to a serving endpoint, the indexes
# MAGIC can be recreated to use that custom embedding endpoint instead.
# MAGIC
# MAGIC For the demo, the built-in BGE embedding works well and avoids the dependency on
# MAGIC GPU model serving being ready before index creation.

# COMMAND ----------

for idx_config in indexes:
    index_name = idx_config["index_name"]
    source_table = idx_config["source_table"]
    primary_key = idx_config["primary_key"]
    text_column = idx_config["text_column"]

    # Check if index already exists
    try:
        existing = vsc.get_index(endpoint_name=vs_endpoint_name, index_name=index_name)
        print(f"Index {index_name} already exists. Syncing...")
        existing.sync()
        continue
    except Exception:
        pass  # Index doesn't exist, create it

    print(f"Creating index: {index_name}")
    print(f"  Source: {source_table}")
    print(f"  Endpoint: {vs_endpoint_name}")

    vsc.create_delta_sync_index(
        endpoint_name=vs_endpoint_name,
        index_name=index_name,
        source_table_name=source_table,
        pipeline_type="TRIGGERED",
        primary_key=primary_key,
        embedding_source_column=text_column,
        embedding_model_endpoint_name="databricks-bge-large-en",
    )
    print(f"  Created and syncing: {index_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Index Status

# COMMAND ----------

import time

print("Checking index sync status...")
for idx_config in indexes:
    index_name = idx_config["index_name"]
    try:
        idx = vsc.get_index(endpoint_name=vs_endpoint_name, index_name=index_name)
        status = idx.describe()
        sync_state = status.get("status", {}).get("detailed_state", "UNKNOWN")
        print(f"  {index_name}: {sync_state}")
    except Exception as e:
        print(f"  {index_name}: Error - {e}")

print("\nNote: Indexes will continue syncing in the background. This may take several minutes.")
print("You can check status in the Databricks UI under Compute > Vector Search.")
