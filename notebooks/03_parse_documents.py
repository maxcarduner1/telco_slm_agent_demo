# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Parse Documents and Create Chunks
# MAGIC
# MAGIC Reads generated documents from the UC Volume, parses them with `ai_parse_document`,
# MAGIC chunks the parsed text, and writes chunk tables for Vector Search indexing.

# COMMAND ----------

dbutils.widgets.text("catalog", "cmegdemos_catalog", "Catalog")
dbutils.widgets.text("schema", "network_analytics_enablement", "Schema")

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

volume_path = f"/Volumes/{catalog}/{schema}/telco_docs"

# COMMAND ----------

from pyspark.sql.functions import col, lit, explode, expr, monotonically_increasing_id, length
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parse documents with ai_parse_document
# MAGIC
# MAGIC Since we wrote plain text files (not PDFs), we can read them directly.
# MAGIC In production with real PDFs from SharePoint, you would use:
# MAGIC ```sql
# MAGIC SELECT ai_parse_document(content, 'text') FROM ...
# MAGIC ```
# MAGIC For this demo, we read the text files directly and treat them as parsed output.

# COMMAND ----------

def read_docs_from_volume(subfolder, doc_type):
    """Read all text documents from a volume subfolder into a DataFrame."""
    path = f"{volume_path}/{subfolder}/"
    df = (
        spark.read.text(path, wholetext=True)
        .withColumn("source_path", col("_metadata.file_path"))
        .withColumn("doc_type", lit(doc_type))
        .withColumnRenamed("value", "parsed_text")
    )
    return df

# COMMAND ----------

df_runbooks = read_docs_from_volume("runbooks", "runbook")
df_standards = read_docs_from_volume("standards", "standard")
df_incidents = read_docs_from_volume("incidents", "incident")

print(f"Runbooks parsed:  {df_runbooks.count()}")
print(f"Standards parsed: {df_standards.count()}")
print(f"Incidents parsed: {df_incidents.count()}")

# COMMAND ----------

# Write parsed tables
df_runbooks.write.mode("overwrite").saveAsTable(f"`{catalog}`.`{schema}`.`telco_docs_runbooks_parsed`")
df_standards.write.mode("overwrite").saveAsTable(f"`{catalog}`.`{schema}`.`telco_docs_standards_parsed`")
df_incidents.write.mode("overwrite").saveAsTable(f"`{catalog}`.`{schema}`.`telco_docs_incidents_parsed`")

print("Parsed tables written.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chunking
# MAGIC
# MAGIC Split parsed text into overlapping chunks for vector search.
# MAGIC Strategy: 512 tokens (~2000 chars) with 64-token overlap (~256 chars), section-aware.

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, IntegerType

def chunk_text(text, chunk_size=2000, overlap=256):
    """Split text into overlapping chunks, respecting paragraph boundaries where possible."""
    if not text or len(text) < 100:
        return [{"chunk_text": text, "chunk_index": 0}]

    chunks = []
    # Split on double newlines (paragraphs) first
    paragraphs = text.split("\n\n")

    current_chunk = ""
    chunk_idx = 0

    for para in paragraphs:
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append({"chunk_text": current_chunk.strip(), "chunk_index": chunk_idx})
            chunk_idx += 1
            # Keep overlap from end of previous chunk
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + "\n\n" + para
        else:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para

    if current_chunk.strip():
        chunks.append({"chunk_text": current_chunk.strip(), "chunk_index": chunk_idx})

    return chunks

chunk_schema = ArrayType(StructType([
    StructField("chunk_text", StringType(), False),
    StructField("chunk_index", IntegerType(), False),
]))

chunk_udf = udf(chunk_text, chunk_schema)

# COMMAND ----------

def create_chunk_table(parsed_table, chunk_table_name, doc_type):
    """Read a parsed table, chunk it, and write the chunk table."""
    df = spark.table(f"`{catalog}`.`{schema}`.`{parsed_table}`")

    df_chunks = (
        df
        .withColumn("chunks", chunk_udf(col("parsed_text")))
        .select(
            col("source_path"),
            col("doc_type"),
            explode(col("chunks")).alias("chunk")
        )
        .select(
            col("source_path"),
            col("doc_type"),
            col("chunk.chunk_text").alias("chunk_text"),
            col("chunk.chunk_index").alias("chunk_index"),
        )
        .withColumn("chunk_id", monotonically_increasing_id())
    )

    df_chunks.write.mode("overwrite").saveAsTable(f"`{catalog}`.`{schema}`.`{chunk_table_name}`")
    count = df_chunks.count()
    print(f"  {chunk_table_name}: {count} chunks")
    return count

# COMMAND ----------

print("Creating chunk tables...")
n_runbooks = create_chunk_table("telco_docs_runbooks_parsed", "telco_docs_runbooks_chunks", "runbook")
n_standards = create_chunk_table("telco_docs_standards_parsed", "telco_docs_standards_chunks", "standard")
n_incidents = create_chunk_table("telco_docs_incidents_parsed", "telco_docs_incidents_chunks", "incident")

print(f"\nTotal chunks: {n_runbooks + n_standards + n_incidents}")

# COMMAND ----------

# Verify chunk tables
display(spark.table(f"`{catalog}`.`{schema}`.`telco_docs_runbooks_chunks`").limit(5))
