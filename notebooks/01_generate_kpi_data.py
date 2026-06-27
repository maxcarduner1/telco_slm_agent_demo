# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Generate Synthetic Network KPI Data
# MAGIC
# MAGIC Generates 90 days of hourly network KPI data across 6 regions and 50 sites,
# MAGIC with injected anomalies for demo purposes.

# COMMAND ----------

dbutils.widgets.text("catalog", "cmegdemos_catalog", "Catalog")
dbutils.widgets.text("schema", "network_analytics_enablement", "Schema")

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# Catalog already exists on this workspace; just ensure schema exists
spark.sql(f"USE CATALOG `{catalog}`")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`")

# COMMAND ----------

import random
from datetime import datetime, timedelta
from faker import Faker
from pyspark.sql import Row
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, TimestampType, IntegerType, DateType
)

fake = Faker()
Faker.seed(42)
random.seed(42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

REGIONS = [
    "Pacific Northwest",
    "Southern California",
    "Mountain West",
    "Great Plains",
    "Southeast",
    "Northeast"
]

# 50 sites distributed across regions
SITES = {}
sites_per_region = 8
for i, region in enumerate(REGIONS):
    for j in range(sites_per_region if i < 4 else 9):
        site_id = f"SITE-{region[:2].upper()}-{j+1:03d}"
        SITES[site_id] = region

# Trim to 50
SITES = dict(list(SITES.items())[:50])

DAYS = 90
# Anchor synthetic data to "now" so default look-back queries return fresh rows.
END_TIMESTAMP = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
START_DATE = END_TIMESTAMP - timedelta(days=DAYS)
print(f"Generating KPI data window: {START_DATE} -> {END_TIMESTAMP}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate network_kpis_hourly

# COMMAND ----------

def generate_kpi_row(site_id, region, ts, anomaly=False):
    """Generate a single hourly KPI observation."""
    # Base values (healthy network)
    coverage = random.gauss(99.2, 0.3)
    throughput = random.gauss(145, 20)
    latency = random.gauss(12, 2)
    dropped_call_rate = random.gauss(0.2, 0.08)
    handover_success = random.gauss(99.5, 0.2)
    attach_success = random.gauss(99.7, 0.15)
    volte_mos = random.gauss(4.3, 0.15)

    if anomaly:
        # Inject correlated degradation (simulates real network issues)
        degradation_type = random.choice(["coverage", "congestion", "interference", "backhaul"])
        if degradation_type == "coverage":
            coverage -= random.uniform(3, 8)
            dropped_call_rate += random.uniform(0.5, 2.0)
            handover_success -= random.uniform(1, 4)
        elif degradation_type == "congestion":
            throughput -= random.uniform(40, 80)
            latency += random.uniform(10, 30)
            volte_mos -= random.uniform(0.5, 1.5)
        elif degradation_type == "interference":
            coverage -= random.uniform(2, 5)
            throughput -= random.uniform(20, 50)
            latency += random.uniform(5, 15)
        elif degradation_type == "backhaul":
            throughput -= random.uniform(60, 100)
            latency += random.uniform(20, 50)
            dropped_call_rate += random.uniform(0.3, 1.0)

    return Row(
        timestamp=ts,
        region=region,
        site_id=site_id,
        coverage_pct=max(min(coverage, 100.0), 85.0),
        throughput_mbps=max(throughput, 10.0),
        latency_ms=max(latency, 3.0),
        dropped_call_rate=max(dropped_call_rate, 0.0),
        handover_success_rate=max(min(handover_success, 100.0), 90.0),
        attach_success_rate=max(min(attach_success, 100.0), 92.0),
        volte_mos=max(min(volte_mos, 5.0), 2.0)
    )


# Generate anomaly windows: specific sites/time ranges that are degraded
anomaly_windows = []
for _ in range(15):  # 15 anomaly events over 90 days
    anomaly_site = random.choice(list(SITES.keys()))
    anomaly_start_day = random.randint(0, DAYS - 3)
    anomaly_duration_hours = random.randint(4, 48)
    anomaly_start = START_DATE + timedelta(days=anomaly_start_day, hours=random.randint(0, 23))
    anomaly_end = anomaly_start + timedelta(hours=anomaly_duration_hours)
    anomaly_windows.append((anomaly_site, anomaly_start, anomaly_end))

# Also inject a RECENT anomaly in Pacific Northwest for the demo walkthrough
recent_anomaly_end = END_TIMESTAMP - timedelta(hours=2)
recent_anomaly_start = recent_anomaly_end - timedelta(hours=42)
pnw_sites = [s for s, r in SITES.items() if r == "Pacific Northwest"]
for site in pnw_sites[:3]:  # Hit 3 PNW sites
    anomaly_windows.append((site, recent_anomaly_start, recent_anomaly_end))


def is_anomaly(site_id, ts):
    for a_site, a_start, a_end in anomaly_windows:
        if site_id == a_site and a_start <= ts <= a_end:
            return True
    return False


# Generate all rows
rows = []
for day in range(DAYS):
    for hour in range(24):
        ts = START_DATE + timedelta(days=day, hours=hour)
        for site_id, region in SITES.items():
            anomaly = is_anomaly(site_id, ts)
            rows.append(generate_kpi_row(site_id, region, ts, anomaly))

print(f"Generated {len(rows):,} KPI rows")

# COMMAND ----------

kpi_schema = StructType([
    StructField("timestamp", TimestampType(), False),
    StructField("region", StringType(), False),
    StructField("site_id", StringType(), False),
    StructField("coverage_pct", DoubleType(), False),
    StructField("throughput_mbps", DoubleType(), False),
    StructField("latency_ms", DoubleType(), False),
    StructField("dropped_call_rate", DoubleType(), False),
    StructField("handover_success_rate", DoubleType(), False),
    StructField("attach_success_rate", DoubleType(), False),
    StructField("volte_mos", DoubleType(), False),
])

df_kpis = spark.createDataFrame(rows, schema=kpi_schema)
df_kpis.write.mode("overwrite").saveAsTable(f"`{catalog}`.`{schema}`.`network_kpis_hourly`")

print(f"Wrote network_kpis_hourly: {df_kpis.count()} rows")
display(df_kpis.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate network_events

# COMMAND ----------

EVENT_TYPES = ["ALARM", "MAINTENANCE", "OUTAGE", "DEGRADATION"]
SEVERITIES = ["CRITICAL", "MAJOR", "MINOR", "WARNING"]

EVENT_DESCRIPTIONS = {
    "ALARM": [
        "High VSWR detected on sector antenna",
        "Power amplifier degradation",
        "Temperature threshold exceeded in RRU",
        "S1 link flap detected",
        "X2 handover failure rate above threshold",
        "PRACH preamble collision rate elevated",
    ],
    "MAINTENANCE": [
        "Planned software upgrade - eNB firmware v4.2.1",
        "Antenna tilt adjustment - optimization",
        "Battery replacement - backup power unit",
        "Fiber splice repair on backhaul link",
        "RAN parameter optimization window",
    ],
    "OUTAGE": [
        "Complete site outage - power failure",
        "Backhaul fiber cut - third party dig",
        "Hardware failure - baseband unit",
        "Core network connectivity lost",
    ],
    "DEGRADATION": [
        "Throughput degradation - congestion detected",
        "Increased latency on S1-U interface",
        "Coverage hole reported - antenna misalignment suspected",
        "VoLTE MOS below threshold - codec degradation",
        "Handover success rate below 97%",
        "Attach failure rate elevated - MME overload",
    ],
}

event_rows = []
for _ in range(200):
    event_type = random.choices(EVENT_TYPES, weights=[40, 20, 10, 30])[0]
    severity = random.choices(SEVERITIES, weights=[10, 25, 35, 30])[0]
    if event_type == "OUTAGE":
        severity = random.choices(["CRITICAL", "MAJOR"], weights=[70, 30])[0]

    day_offset = random.randint(0, DAYS - 1)
    hour = random.randint(0, 23)
    ts = START_DATE + timedelta(days=day_offset, hours=hour)

    site_id = random.choice(list(SITES.keys()))
    region = SITES[site_id]

    desc = random.choice(EVENT_DESCRIPTIONS[event_type])

    # Some events are resolved, some not
    resolved = None
    if random.random() < 0.85:
        resolve_hours = random.randint(1, 72)
        resolved = ts + timedelta(hours=resolve_hours)

    event_rows.append(Row(
        event_id=f"EVT-{fake.uuid4()[:8].upper()}",
        timestamp=ts,
        site_id=site_id,
        region=region,
        event_type=event_type,
        severity=severity,
        description=desc,
        resolved_at=resolved,
    ))

# Add recent PNW events that correlate with the KPI anomaly
recent_event_base = END_TIMESTAMP - timedelta(days=2, hours=4)
pnw_recent_events = [
    Row(event_id="EVT-PNW-001", timestamp=recent_event_base + timedelta(minutes=22),
        site_id=pnw_sites[0], region="Pacific Northwest", event_type="ALARM",
        severity="MAJOR", description="High VSWR detected on sector antenna",
        resolved_at=None),
    Row(event_id="EVT-PNW-002", timestamp=recent_event_base + timedelta(minutes=45),
        site_id=pnw_sites[1], region="Pacific Northwest", event_type="DEGRADATION",
        severity="MAJOR", description="Increased latency on S1-U interface",
        resolved_at=None),
    Row(event_id="EVT-PNW-003", timestamp=recent_event_base + timedelta(hours=1, minutes=10),
        site_id=pnw_sites[2], region="Pacific Northwest", event_type="DEGRADATION",
        severity="CRITICAL", description="Throughput degradation - congestion detected",
        resolved_at=None),
]
event_rows.extend(pnw_recent_events)

event_schema = StructType([
    StructField("event_id", StringType(), False),
    StructField("timestamp", TimestampType(), False),
    StructField("site_id", StringType(), False),
    StructField("region", StringType(), False),
    StructField("event_type", StringType(), False),
    StructField("severity", StringType(), False),
    StructField("description", StringType(), False),
    StructField("resolved_at", TimestampType(), True),
])

df_events = spark.createDataFrame(event_rows, schema=event_schema)
df_events.write.mode("overwrite").saveAsTable(f"`{catalog}`.`{schema}`.`network_events`")

print(f"Wrote network_events: {df_events.count()} rows")
display(df_events.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate customer_churn_daily

# COMMAND ----------

SEGMENTS = ["Consumer", "Enterprise", "Prepaid"]

churn_rows = []
recent_churn_spike_start = (END_TIMESTAMP - timedelta(days=6)).date()
for day in range(DAYS):
    date = (START_DATE + timedelta(days=day)).date()
    for region in REGIONS:
        for segment in SEGMENTS:
            base_churn = {"Consumer": 0.04, "Enterprise": 0.015, "Prepaid": 0.06}[segment]
            churn = random.gauss(base_churn, base_churn * 0.15)

            # Spike churn in PNW during anomaly period
            if region == "Pacific Northwest" and recent_churn_spike_start <= date:
                churn += random.uniform(0.005, 0.015)

            net_adds = int(random.gauss(50, 30))
            if churn > base_churn * 1.3:
                net_adds -= int(random.uniform(20, 60))

            churn_rows.append(Row(
                date=date,
                region=region,
                segment=segment,
                churn_rate=max(churn, 0.001),
                net_adds=net_adds,
            ))

churn_schema = StructType([
    StructField("date", DateType(), False),
    StructField("region", StringType(), False),
    StructField("segment", StringType(), False),
    StructField("churn_rate", DoubleType(), False),
    StructField("net_adds", IntegerType(), False),
])

df_churn = spark.createDataFrame(churn_rows, schema=churn_schema)
df_churn.write.mode("overwrite").saveAsTable(f"`{catalog}`.`{schema}`.`customer_churn_daily`")

print(f"Wrote customer_churn_daily: {df_churn.count()} rows")
display(df_churn.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create UC Volume for documents

# COMMAND ----------

spark.sql(f"CREATE VOLUME IF NOT EXISTS `{catalog}`.`{schema}`.`telco_docs`")
print(f"Volume created: {catalog}.{schema}.telco_docs")
