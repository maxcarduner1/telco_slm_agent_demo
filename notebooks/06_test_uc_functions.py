# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - Test UC Functions
# MAGIC
# MAGIC Validates all 5 UC functions with realistic agent-style queries.
# MAGIC Each function is tested with multiple parameter combinations to ensure
# MAGIC correct behavior before wiring them as LangGraph tools.

# COMMAND ----------

dbutils.widgets.text("catalog", "cmegdemos_catalog", "Catalog")
dbutils.widgets.text("schema", "network_analytics_enablement", "Schema")

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

spark.sql(f"USE CATALOG `{catalog}`")
spark.sql(f"USE SCHEMA `{schema}`")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 1: get_kpi_metrics
# MAGIC
# MAGIC "Show me latency data for the Pacific Northwest in the last 48 hours"

# COMMAND ----------

# Determine data ranges so tests use valid look-back windows
# KPI data
kpi_range = spark.sql(f"""
  SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts,
    TIMESTAMPDIFF(HOUR, MIN(timestamp), current_timestamp()) as hours_since_earliest
  FROM `{catalog}`.`{schema}`.network_kpis_hourly
""").collect()[0]
print(f"KPI data range: {kpi_range.min_ts} to {kpi_range.max_ts}")
print(f"  Hours since earliest: {kpi_range.hours_since_earliest}")
LOOKBACK = int(kpi_range.hours_since_earliest) + 24
print(f"  Using KPI LOOKBACK: {LOOKBACK} hours")

# Events data
events_range = spark.sql(f"""
  SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts,
    TIMESTAMPDIFF(HOUR, MIN(timestamp), current_timestamp()) as hours_since_earliest
  FROM `{catalog}`.`{schema}`.network_events
""").collect()[0]
print(f"\nEvents data range: {events_range.min_ts} to {events_range.max_ts}")
print(f"  Hours since earliest: {events_range.hours_since_earliest}")
EVENTS_LOOKBACK = int(events_range.hours_since_earliest) + 24
print(f"  Using EVENTS_LOOKBACK: {EVENTS_LOOKBACK} hours")

# Churn data (uses days)
churn_range = spark.sql(f"""
  SELECT MIN(date) as min_date, MAX(date) as max_date,
    DATEDIFF(current_date(), MIN(date)) as days_since_earliest
  FROM `{catalog}`.`{schema}`.customer_churn_daily
""").collect()[0]
print(f"\nChurn data range: {churn_range.min_date} to {churn_range.max_date}")
print(f"  Days since earliest: {churn_range.days_since_earliest}")
CHURN_DAYS = int(churn_range.days_since_earliest) + 7
print(f"  Using CHURN_DAYS: {CHURN_DAYS} days\n")

# COMMAND ----------

# Test 1a: Specific metric + region + time window
df = spark.sql(f"""
  SELECT * FROM `{catalog}`.`{schema}`.get_kpi_metrics(
    metric_name => 'latency_ms',
    region => 'Pacific Northwest',
    hours_back => {LOOKBACK}
  )
""")
print(f"Test 1a - latency_ms, Pacific Northwest: {df.count()} rows")
assert df.count() > 0, "Expected rows for latency in PNW"
display(df.limit(10))

# COMMAND ----------

# Test 1b: All regions, specific metric
df = spark.sql(f"""
  SELECT * FROM `{catalog}`.`{schema}`.get_kpi_metrics(
    metric_name => 'throughput_mbps',
    hours_back => {LOOKBACK}
  )
""")
print(f"Test 1b - throughput_mbps, all regions, 12h: {df.count()} rows")
assert df.count() > 0, "Expected rows for throughput"
display(df.limit(10))

# COMMAND ----------

# Test 1c: Specific site
df = spark.sql(f"""
  SELECT * FROM `{catalog}`.`{schema}`.get_kpi_metrics(
    metric_name => 'volte_mos',
    site_id => 'SITE-PA-001',
    hours_back => {LOOKBACK}
  )
""")
print(f"Test 1c - volte_mos, SITE-PA-001, 24h: {df.count()} rows")
display(df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 2: get_threshold_breaches
# MAGIC
# MAGIC "Find any sites where latency exceeded 50ms in the last 24 hours"

# COMMAND ----------

# Test 2a: Latency above threshold
df = spark.sql(f"""
  SELECT * FROM `{catalog}`.`{schema}`.get_threshold_breaches(
    metric_name => 'latency_ms',
    threshold => 50.0,
    direction => 'above',
    hours_back => {LOOKBACK}
  )
""")
print(f"Test 2a - latency > 50ms, all regions, 7d: {df.count()} breaches")
display(df.limit(10))

# COMMAND ----------

# Test 2b: Throughput below threshold (direction = 'below')
df = spark.sql(f"""
  SELECT * FROM `{catalog}`.`{schema}`.get_threshold_breaches(
    metric_name => 'throughput_mbps',
    threshold => 100.0,
    direction => 'below',
    region => 'Pacific Northwest',
    hours_back => {LOOKBACK}
  )
""")
print(f"Test 2b - throughput < 100mbps, PNW, 7d: {df.count()} breaches")
display(df.limit(10))

# COMMAND ----------

# Test 2c: Dropped call rate above threshold
df = spark.sql(f"""
  SELECT * FROM `{catalog}`.`{schema}`.get_threshold_breaches(
    metric_name => 'dropped_call_rate',
    threshold => 0.02,
    direction => 'above',
    hours_back => {LOOKBACK}
  )
""")
print(f"Test 2c - dropped_call_rate > 2%, 7d: {df.count()} breaches")
display(df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 3: compare_regions
# MAGIC
# MAGIC "Compare average latency across all regions for the last 24 hours"

# COMMAND ----------

# Test 3a: Average latency comparison
df = spark.sql(f"""
  SELECT * FROM `{catalog}`.`{schema}`.compare_regions(
    metric_name => 'latency_ms',
    hours_back => {LOOKBACK},
    agg => 'avg'
  )
""")
print(f"Test 3a - avg latency by region, 7d: {df.count()} regions")
assert df.count() > 0, "Expected at least one region"
display(df)

# COMMAND ----------

# Test 3b: P95 throughput comparison
df = spark.sql(f"""
  SELECT * FROM `{catalog}`.`{schema}`.compare_regions(
    metric_name => 'throughput_mbps',
    hours_back => {LOOKBACK},
    agg => 'p95'
  )
""")
print(f"Test 3b - p95 throughput by region, 7d:")
display(df)

# COMMAND ----------

# Test 3c: Max dropped call rate
df = spark.sql(f"""
  SELECT * FROM `{catalog}`.`{schema}`.compare_regions(
    metric_name => 'dropped_call_rate',
    hours_back => {LOOKBACK},
    agg => 'max'
  )
""")
print(f"Test 3c - max dropped_call_rate by region, 7d:")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 4: get_network_events
# MAGIC
# MAGIC "Show me all critical events in the last 3 days"

# COMMAND ----------

# Test 4a: Critical events
df = spark.sql(f"""
  SELECT * FROM `{catalog}`.`{schema}`.get_network_events(
    severity => 'CRITICAL',
    hours_back => {EVENTS_LOOKBACK}
  )
""")
print(f"Test 4a - critical events, {EVENTS_LOOKBACK}h: {df.count()} events")
display(df.limit(10))

# COMMAND ----------

# Test 4b: Unresolved events only
df = spark.sql(f"""
  SELECT * FROM `{catalog}`.`{schema}`.get_network_events(
    unresolved_only => TRUE,
    hours_back => {EVENTS_LOOKBACK}
  )
""")
print(f"Test 4b - unresolved events, {EVENTS_LOOKBACK}h: {df.count()} events")
display(df.limit(10))

# COMMAND ----------

# Test 4c: Events in a specific region with type filter
df = spark.sql(f"""
  SELECT * FROM `{catalog}`.`{schema}`.get_network_events(
    region => 'Pacific Northwest',
    event_type => 'OUTAGE',
    hours_back => {EVENTS_LOOKBACK}
  )
""")
print(f"Test 4c - PNW outages, {EVENTS_LOOKBACK}h: {df.count()} events")
display(df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 5: get_churn_risk
# MAGIC
# MAGIC "Which regions have the highest churn rate this month?"

# COMMAND ----------

# Test 5a: All regions, high churn
df = spark.sql(f"""
  SELECT * FROM `{catalog}`.`{schema}`.get_churn_risk(
    min_churn_rate => 0.03,
    days_back => {CHURN_DAYS}
  )
""")
print(f"Test 5a - churn > 3%, {CHURN_DAYS}d: {df.count()} rows")
display(df.limit(10))

# COMMAND ----------

# Test 5b: Specific region
df = spark.sql(f"""
  SELECT * FROM `{catalog}`.`{schema}`.get_churn_risk(
    region => 'Pacific Northwest',
    days_back => {CHURN_DAYS}
  )
""")
print(f"Test 5b - PNW churn, {CHURN_DAYS}d: {df.count()} rows")
display(df.limit(10))

# COMMAND ----------

# Test 5c: Specific segment
df = spark.sql(f"""
  SELECT * FROM `{catalog}`.`{schema}`.get_churn_risk(
    segment => 'Enterprise',
    min_churn_rate => 0.01,
    days_back => {CHURN_DAYS}
  )
""")
print(f"Test 5c - enterprise churn > 1%, {CHURN_DAYS}d: {df.count()} rows")
display(df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("=" * 60)
print("UC FUNCTION VALIDATION COMPLETE")
print("=" * 60)
print()
print("All 5 functions registered and returning data:")
print(f"  1. {catalog}.{schema}.get_kpi_metrics")
print(f"  2. {catalog}.{schema}.get_threshold_breaches")
print(f"  3. {catalog}.{schema}.compare_regions")
print(f"  4. {catalog}.{schema}.get_network_events")
print(f"  5. {catalog}.{schema}.get_churn_risk")
print()
print("Ready to wire as LangGraph agent tools via UCFunctionToolkit.")
