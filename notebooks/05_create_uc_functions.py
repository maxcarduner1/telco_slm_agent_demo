# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Create UC Functions (Agent Tools)
# MAGIC
# MAGIC Registers Unity Catalog SQL functions that serve as typed, parameterized tools
# MAGIC for the LangGraph agent. These replace a Genie Space — the agent calls them
# MAGIC directly with structured parameters instead of generating free-form SQL.
# MAGIC
# MAGIC Functions:
# MAGIC 1. `get_kpi_metrics` — Query raw KPI values with filters
# MAGIC 2. `get_threshold_breaches` — Find metric violations
# MAGIC 3. `compare_regions` — Aggregate a metric across regions
# MAGIC 4. `get_network_events` — Query events by severity/type
# MAGIC 5. `get_churn_risk` — Churn trends by region/segment

# COMMAND ----------

dbutils.widgets.text("catalog", "cmegdemos_catalog", "Catalog")
dbutils.widgets.text("schema", "network_analytics_enablement", "Schema")

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

spark.sql(f"USE CATALOG `{catalog}`")
spark.sql(f"USE SCHEMA `{schema}`")

print(f"Registering UC functions in {catalog}.{schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. get_kpi_metrics
# MAGIC
# MAGIC Query raw KPI time-series data with optional region/site/metric filters.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION `{catalog}`.`{schema}`.get_kpi_metrics(
  metric_name STRING COMMENT 'One of: throughput_mbps, latency_ms, coverage_pct, dropped_call_rate, handover_success_rate, attach_success_rate, volte_mos',
  region STRING DEFAULT NULL COMMENT 'Filter by region. Values: Pacific Northwest, Northeast, Southeast, Mountain West, Great Plains. NULL = all regions',
  site_id STRING DEFAULT NULL COMMENT 'Filter by specific site. NULL = all sites',
  hours_back INT DEFAULT 24 COMMENT 'Look-back window in hours from now'
)
RETURNS TABLE (
  timestamp TIMESTAMP,
  region STRING,
  site_id STRING,
  metric_value DOUBLE
)
COMMENT 'Query network KPI time-series data for a specific metric with optional region/site filters and time window.'
RETURN
  SELECT timestamp, region, site_id,
    CASE metric_name
      WHEN 'throughput_mbps' THEN throughput_mbps
      WHEN 'latency_ms' THEN latency_ms
      WHEN 'coverage_pct' THEN coverage_pct
      WHEN 'dropped_call_rate' THEN dropped_call_rate
      WHEN 'handover_success_rate' THEN handover_success_rate
      WHEN 'attach_success_rate' THEN attach_success_rate
      WHEN 'volte_mos' THEN volte_mos
    END AS metric_value
  FROM `{catalog}`.`{schema}`.network_kpis_hourly
  WHERE timestamp >= TIMESTAMPADD(HOUR, -hours_back, current_timestamp())
    AND (region = get_kpi_metrics.region OR get_kpi_metrics.region IS NULL)
    AND (site_id = get_kpi_metrics.site_id OR get_kpi_metrics.site_id IS NULL)
  ORDER BY timestamp DESC
""")
print("Created: get_kpi_metrics")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. get_threshold_breaches
# MAGIC
# MAGIC Find KPI readings that exceed (or fall below) a given threshold.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION `{catalog}`.`{schema}`.get_threshold_breaches(
  metric_name STRING COMMENT 'Metric to check: throughput_mbps, latency_ms, coverage_pct, dropped_call_rate, handover_success_rate, attach_success_rate, volte_mos',
  threshold DOUBLE COMMENT 'Threshold value to check against',
  direction STRING DEFAULT 'above' COMMENT 'Check values above or below the threshold',
  region STRING DEFAULT NULL COMMENT 'Filter by region (Pacific Northwest, Northeast, Southeast, Mountain West, Great Plains). NULL = all',
  hours_back INT DEFAULT 24 COMMENT 'Look-back window in hours'
)
RETURNS TABLE (
  timestamp TIMESTAMP,
  region STRING,
  site_id STRING,
  metric_value DOUBLE,
  breach_amount DOUBLE
)
COMMENT 'Find network KPI readings that breach a threshold. Use direction=above for metrics where high is bad (latency, dropped calls) and direction=below for metrics where low is bad (throughput, coverage).'
RETURN
  SELECT timestamp, region, site_id, metric_value,
    CASE direction
      WHEN 'above' THEN metric_value - threshold
      ELSE threshold - metric_value
    END AS breach_amount
  FROM (
    SELECT timestamp, region, site_id,
      CASE metric_name
        WHEN 'throughput_mbps' THEN throughput_mbps
        WHEN 'latency_ms' THEN latency_ms
        WHEN 'coverage_pct' THEN coverage_pct
        WHEN 'dropped_call_rate' THEN dropped_call_rate
        WHEN 'handover_success_rate' THEN handover_success_rate
        WHEN 'attach_success_rate' THEN attach_success_rate
        WHEN 'volte_mos' THEN volte_mos
      END AS metric_value
    FROM `{catalog}`.`{schema}`.network_kpis_hourly
    WHERE timestamp >= TIMESTAMPADD(HOUR, -hours_back, current_timestamp())
      AND (region = get_threshold_breaches.region OR get_threshold_breaches.region IS NULL)
  ) readings
  WHERE CASE direction
    WHEN 'above' THEN metric_value > threshold
    ELSE metric_value < threshold
  END
  ORDER BY breach_amount DESC
""")
print("Created: get_threshold_breaches")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. compare_regions
# MAGIC
# MAGIC Compare a metric across all regions with aggregation.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION `{catalog}`.`{schema}`.compare_regions(
  metric_name STRING COMMENT 'Metric to compare: throughput_mbps, latency_ms, coverage_pct, dropped_call_rate, handover_success_rate, attach_success_rate, volte_mos',
  hours_back INT DEFAULT 24 COMMENT 'Look-back window in hours',
  agg STRING DEFAULT 'avg' COMMENT 'Aggregation function: avg, max, min, p95'
)
RETURNS TABLE (
  region STRING,
  agg_value DOUBLE,
  sample_count BIGINT
)
COMMENT 'Compare a metric across all regions using an aggregation function. Useful for identifying which regions are performing best/worst.'
RETURN
  SELECT region,
    CASE agg
      WHEN 'avg' THEN AVG(metric_value)
      WHEN 'max' THEN MAX(metric_value)
      WHEN 'min' THEN MIN(metric_value)
      WHEN 'p95' THEN PERCENTILE_APPROX(metric_value, 0.95)
    END AS agg_value,
    COUNT(*) AS sample_count
  FROM (
    SELECT region,
      CASE metric_name
        WHEN 'throughput_mbps' THEN throughput_mbps
        WHEN 'latency_ms' THEN latency_ms
        WHEN 'coverage_pct' THEN coverage_pct
        WHEN 'dropped_call_rate' THEN dropped_call_rate
        WHEN 'handover_success_rate' THEN handover_success_rate
        WHEN 'attach_success_rate' THEN attach_success_rate
        WHEN 'volte_mos' THEN volte_mos
      END AS metric_value
    FROM `{catalog}`.`{schema}`.network_kpis_hourly
    WHERE timestamp >= TIMESTAMPADD(HOUR, -hours_back, current_timestamp())
  ) readings
  GROUP BY region
  ORDER BY agg_value DESC
""")
print("Created: compare_regions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. get_network_events
# MAGIC
# MAGIC Query network events with filters on region, severity, and type.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION `{catalog}`.`{schema}`.get_network_events(
  region STRING DEFAULT NULL COMMENT 'Filter by region (Pacific Northwest, Northeast, Southeast, Mountain West, Great Plains). NULL = all',
  severity STRING DEFAULT NULL COMMENT 'Filter by severity: CRITICAL, MAJOR, MINOR, WARNING. NULL = all',
  event_type STRING DEFAULT NULL COMMENT 'Filter by event type: OUTAGE, DEGRADATION, MAINTENANCE, ALARM. NULL = all',
  hours_back INT DEFAULT 72 COMMENT 'Look-back window in hours',
  unresolved_only BOOLEAN DEFAULT FALSE COMMENT 'If TRUE, only return events that have not been resolved'
)
RETURNS TABLE (
  event_id STRING,
  timestamp TIMESTAMP,
  region STRING,
  site_id STRING,
  event_type STRING,
  severity STRING,
  description STRING,
  resolved_at TIMESTAMP,
  is_unresolved BOOLEAN
)
COMMENT 'Query network events (outages, degradations, alarms, maintenance) with optional filters. Use unresolved_only=TRUE to find active incidents.'
RETURN
  SELECT event_id, timestamp, region, site_id, event_type, severity,
    description, resolved_at,
    CASE WHEN resolved_at IS NULL THEN TRUE ELSE FALSE END AS is_unresolved
  FROM `{catalog}`.`{schema}`.network_events
  WHERE timestamp >= TIMESTAMPADD(HOUR, -hours_back, current_timestamp())
    AND (region = get_network_events.region OR get_network_events.region IS NULL)
    AND (severity = get_network_events.severity OR get_network_events.severity IS NULL)
    AND (event_type = get_network_events.event_type OR get_network_events.event_type IS NULL)
    AND (unresolved_only = FALSE OR resolved_at IS NULL)
  ORDER BY timestamp DESC
""")
print("Created: get_network_events")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. get_churn_risk
# MAGIC
# MAGIC Query customer churn data by region and segment.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION `{catalog}`.`{schema}`.get_churn_risk(
  region STRING DEFAULT NULL COMMENT 'Filter by region (Pacific Northwest, Northeast, Southeast, Mountain West, Great Plains). NULL = all',
  segment STRING DEFAULT NULL COMMENT 'Filter by customer segment: Consumer, Enterprise, Prepaid. NULL = all',
  min_churn_rate DOUBLE DEFAULT 0.0 COMMENT 'Minimum churn rate to include (0.0-1.0)',
  days_back INT DEFAULT 30 COMMENT 'Look-back window in days'
)
RETURNS TABLE (
  date DATE,
  region STRING,
  segment STRING,
  churn_rate DOUBLE,
  net_adds INT
)
COMMENT 'Query customer churn data by region and segment. Higher churn_rate values indicate more customers leaving. Negative net_adds means net subscriber losses.'
RETURN
  SELECT date, region, segment, churn_rate, net_adds
  FROM `{catalog}`.`{schema}`.customer_churn_daily
  WHERE date >= DATEADD(DAY, -days_back, current_date())
    AND churn_rate >= min_churn_rate
    AND (region = get_churn_risk.region OR get_churn_risk.region IS NULL)
    AND (segment = get_churn_risk.segment OR get_churn_risk.segment IS NULL)
  ORDER BY churn_rate DESC
""")
print("Created: get_churn_risk")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify all functions registered

# COMMAND ----------

functions = spark.sql(f"""
  SHOW USER FUNCTIONS IN `{catalog}`.`{schema}`
  LIKE '*get_*'
""")
display(functions)

# COMMAND ----------

print("All UC functions registered successfully.")
print("Functions available as agent tools:")
print(f"  - {catalog}.{schema}.get_kpi_metrics")
print(f"  - {catalog}.{schema}.get_threshold_breaches")
print(f"  - {catalog}.{schema}.compare_regions")
print(f"  - {catalog}.{schema}.get_network_events")
print(f"  - {catalog}.{schema}.get_churn_risk")
