# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Create Vector Search Indexes
# MAGIC
# MAGIC Pre-computes embeddings using the OTel-Embedding-335M model endpoint, stores them
# MAGIC in the chunk tables, then creates Delta Sync Vector Search indexes with pre-computed
# MAGIC embedding vectors.
# MAGIC
# MAGIC At query time, the agent will embed queries using the same endpoint and pass the
# MAGIC vector to `similarity_search(query_vector=...)`.

# COMMAND ----------

dbutils.widgets.text("catalog", "cmegdemos_catalog", "Catalog")
dbutils.widgets.text("schema", "network_analytics_enablement", "Schema")
dbutils.widgets.text("vs_endpoint", "demo_telco_vs_endpoint", "VS Endpoint Name")
dbutils.widgets.text("embedding_endpoint", "otel-embedding2-300m", "Embedding Model Endpoint")
dbutils.widgets.text("app_name", "otel-telco-agent", "Databricks App Name")

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
vs_endpoint_name = dbutils.widgets.get("vs_endpoint")
embedding_endpoint = dbutils.widgets.get("embedding_endpoint")
app_name = dbutils.widgets.get("app_name")

EMBEDDING_DIM = 768
BATCH_SIZE = 32

print(f"Catalog: {catalog}")
print(f"Schema: {schema}")
print(f"VS Endpoint: {vs_endpoint_name}")
print(f"Embedding Endpoint: {embedding_endpoint} (dim={EMBEDDING_DIM})")
print(f"App Name: {app_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pre-compute embeddings with OTel-Embedding-335M
# MAGIC
# MAGIC The OTel embedding model returns raw arrays (non-standard format), so we
# MAGIC pre-compute embeddings and store them in the chunk tables. VS indexes are then
# MAGIC created with `embedding_vector_column` pointing to the stored vectors.

# COMMAND ----------

import requests
import time
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType
from pyspark.sql.functions import col

# Auth for calling the model serving endpoint (notebook identity)
_ws_url = spark.conf.get("spark.databricks.workspaceUrl")
_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()


def get_embeddings(texts, max_retries=3):
    """Call the OTel embedding endpoint. Returns list of 768-dim float arrays."""
    url = f"https://{_ws_url}/serving-endpoints/{embedding_endpoint}/invocations"
    headers = {"Authorization": f"Bearer {_token}", "Content-Type": "application/json"}

    for attempt in range(max_retries):
        resp = requests.post(url, headers=headers, json={"input": texts}, timeout=120)
        if resp.status_code == 200:
            result = resp.json()
            if isinstance(result, list):
                return result
            elif isinstance(result, dict) and "data" in result:
                return [d["embedding"] for d in result["data"]]
            else:
                raise ValueError(f"Unexpected response: {str(result)[:200]}")
        elif resp.status_code == 429:
            time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"Embedding call failed ({resp.status_code}): {resp.text[:300]}")
    raise RuntimeError("Max retries exceeded")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compute and store embeddings for each chunk table

# COMMAND ----------

tables_to_embed = [
    ("telco_docs_runbooks_chunks", "chunk_text"),
    ("telco_docs_standards_chunks", "chunk_text"),
    ("telco_docs_incidents_chunks", "chunk_text"),
]

for table_name, text_col in tables_to_embed:
    fqn = f"`{catalog}`.`{schema}`.`{table_name}`"
    print(f"\nProcessing: {table_name}")

    df = spark.table(fqn)

    # Idempotent: skip embedding computation if column already exists
    if "embedding" in df.columns:
        print(f"  Already has embeddings — skipping (delete table to force recompute)")
        continue

    rows = df.collect()
    texts = [row[text_col] or "empty" for row in rows]
    print(f"  Chunks: {len(texts)}")

    # Batch embed
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch = [t if t.strip() else "empty" for t in batch]
        embeddings = get_embeddings(batch)
        all_embeddings.extend(embeddings)
        print(f"  Embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")

    # Convert to float arrays and build new DataFrame with embedding column
    embedding_floats = [[float(v) for v in emb] for emb in all_embeddings]

    # Add embedding column to existing data
    rows_with_embeddings = []
    for row, emb in zip(rows, embedding_floats):
        rows_with_embeddings.append((*row, emb))

    # Build schema with embedding column appended
    new_schema = StructType(df.schema.fields + [StructField("embedding", ArrayType(FloatType()), False)])
    df_with_emb = spark.createDataFrame(rows_with_embeddings, schema=new_schema)

    # Overwrite table with embeddings
    df_with_emb.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(fqn)
    print(f"  Saved with embeddings: {spark.table(fqn).count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Enable Change Data Feed
# MAGIC Required for Vector Search Delta Sync indexes.

# COMMAND ----------

chunk_tables = [
    f"`{catalog}`.`{schema}`.`telco_docs_runbooks_chunks`",
    f"`{catalog}`.`{schema}`.`telco_docs_standards_chunks`",
    f"`{catalog}`.`{schema}`.`telco_docs_incidents_chunks`",
]

for table in chunk_tables:
    spark.sql(f"ALTER TABLE {table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    print(f"  CDF enabled: {table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Vector Search Indexes
# MAGIC
# MAGIC Using Delta Sync mode with pre-computed embedding vectors from OTel-Embedding-335M.

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

# Verify endpoint
print(f"Using VS endpoint: {vs_endpoint_name}")
try:
    ep_info = vsc.get_endpoint(vs_endpoint_name)
    state = ep_info.get("endpoint_status", {}).get("state", "UNKNOWN")
    print(f"  Status: {state}")
except Exception as e:
    print(f"  Warning: {e}")

# COMMAND ----------

indexes = [
    {
        "index_name": f"{catalog}.{schema}.otel_runbooks_vs_index",
        "source_table": f"{catalog}.{schema}.telco_docs_runbooks_chunks",
        "primary_key": "chunk_id",
    },
    {
        "index_name": f"{catalog}.{schema}.otel_standards_vs_index",
        "source_table": f"{catalog}.{schema}.telco_docs_standards_chunks",
        "primary_key": "chunk_id",
    },
    {
        "index_name": f"{catalog}.{schema}.otel_incidents_vs_index",
        "source_table": f"{catalog}.{schema}.telco_docs_incidents_chunks",
        "primary_key": "chunk_id",
    },
]

for idx_config in indexes:
    index_name = idx_config["index_name"]
    source_table = idx_config["source_table"]
    primary_key = idx_config["primary_key"]

    # Check if index already exists
    try:
        existing = vsc.get_index(endpoint_name=vs_endpoint_name, index_name=index_name)
        print(f"Index {index_name} already exists. Syncing...")
        existing.sync()
        continue
    except Exception:
        pass

    print(f"Creating index: {index_name}")
    print(f"  Source: {source_table}")
    print(f"  Embedding: pre-computed (dim={EMBEDDING_DIM})")

    try:
        vsc.create_delta_sync_index(
            endpoint_name=vs_endpoint_name,
            index_name=index_name,
            source_table_name=source_table,
            pipeline_type="TRIGGERED",
            primary_key=primary_key,
            embedding_vector_column="embedding",
            embedding_dimension=EMBEDDING_DIM,
        )
        print(f"  Created: {index_name}")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"  Index already exists — skipping create, syncing instead")
            vsc.get_index(endpoint_name=vs_endpoint_name, index_name=index_name).sync()
        else:
            raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Index Status

# COMMAND ----------

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

print("\nIndexes syncing in background. Check Compute > Vector Search in the UI.")
print(f"\nAt query time, embed queries via '{embedding_endpoint}' and use:")
print("  index.similarity_search(query_vector=embedding)")

