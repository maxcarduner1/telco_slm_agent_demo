# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — Setup Catalog & Schema
# MAGIC
# MAGIC Idempotently creates the Unity Catalog schema used by all downstream tasks.
# MAGIC Must complete before any notebook that reads or writes to the catalog.
# MAGIC
# MAGIC **Run order:** No upstream dependencies. `01_generate_kpi_data` and
# MAGIC `02_generate_documents` (and transitively every task in the job) depend on this.

# COMMAND ----------

dbutils.widgets.text("catalog", "cmegdemos_catalog",            "Catalog")
dbutils.widgets.text("schema",  "network_analytics_enablement", "Schema")

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema  = dbutils.widgets.get("schema")

print(f"Catalog : {catalog}")
print(f"Schema  : {schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Verify catalog exists

# COMMAND ----------

catalogs = [r.catalog for r in spark.sql("SHOW CATALOGS").collect()]
if catalog not in catalogs:
    raise RuntimeError(
        f"Catalog '{catalog}' does not exist. "
        "Create it first or check the catalog widget value."
    )
print(f"Catalog '{catalog}' found.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create schema if needed

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`")
print(f"Schema '{catalog}.{schema}' is ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create UC volume for telco documents

# COMMAND ----------

spark.sql(f"CREATE VOLUME IF NOT EXISTS `{catalog}`.`{schema}`.`telco_docs`")
print(f"Volume '{catalog}.{schema}.telco_docs' is ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("=" * 60)
print("CATALOG / SCHEMA SETUP COMPLETE")
print("=" * 60)
print()
print(f"Catalog : {catalog}")
print(f"Schema  : {catalog}.{schema}")
print(f"Volume  : {catalog}.{schema}.telco_docs")
print()
print("All downstream tasks will write to this schema.")
