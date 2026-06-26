# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Load / Generate Telco Documents
# MAGIC
# MAGIC Populates the UC Volume with 23 telco documents (runbooks, standards, incidents)
# MAGIC that the RAG pipeline indexes in the next step.
# MAGIC
# MAGIC **Resolution order for each file:**
# MAGIC 1. **Already in volume** — skip (idempotent re-runs)
# MAGIC 2. **Pre-built in GitHub** — download directly (fast, no token cost)
# MAGIC 3. **Generate with Claude** — fall back to Foundation Model API generation
# MAGIC
# MAGIC To use pre-built docs, commit the generated `.txt` files to the `docs/` folder
# MAGIC in the GitHub repo and set `docs_base_url` to the raw GitHub URL below.
# MAGIC Leave `docs_base_url` empty to always generate with Claude.

# COMMAND ----------

dbutils.widgets.text("catalog",       "cmegdemos_catalog",            "Catalog")
dbutils.widgets.text("schema",        "network_analytics_enablement", "Schema")
dbutils.widgets.text(
    "docs_base_url",
    "https://raw.githubusercontent.com/maxcarduner1/telco_slm_agent_demo/main/docs",
    "Pre-built Docs Base URL (empty = always generate)",
)

# COMMAND ----------

import os
import json
import requests

catalog      = dbutils.widgets.get("catalog")
schema       = dbutils.widgets.get("schema")
docs_base    = dbutils.widgets.get("docs_base_url").rstrip("/")

volume_path  = f"/Volumes/{catalog}/{schema}/telco_docs"

# Create subdirectories if they don't exist
for subdir in ("runbooks", "standards", "incidents"):
    dbutils.fs.mkdirs(f"{volume_path}/{subdir}")

API_BASE = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
TOKEN    = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

print(f"Volume path  : {volume_path}")
print(f"Docs base URL: {docs_base if docs_base else '(empty — will generate all)'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper functions

# COMMAND ----------

def file_exists_in_volume(path):
    """Return True if a file already exists in the UC Volume."""
    try:
        dbutils.fs.ls(path)
        return True
    except Exception:
        return False


def try_github_download(category, filename):
    """
    Attempt to download a pre-built document from the GitHub repo.
    Returns the file content string on success, or None if unavailable.
    """
    if not docs_base:
        return None
    url = f"{docs_base}/{category}/{filename}"
    try:
        resp = requests.get(url, timeout=20)
        # Treat anything under 200 chars as a missing/placeholder file
        if resp.status_code == 200 and len(resp.text) > 200:
            return resp.text
    except Exception:
        pass
    return None


def generate_with_claude(prompt, max_tokens=2500):
    """Generate document text using the Foundation Model API (Claude Sonnet 4)."""
    response = requests.post(
        f"{API_BASE}/serving-endpoints/databricks-claude-sonnet-4/invocations",
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
        json={
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a technical writer for a major telecommunications company. "
                        "Write detailed, realistic technical documents. Use proper section "
                        "headings, numbered steps, and technical terminology. "
                        "Do not use markdown formatting — write in plain text with clear structure."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def write_doc(volume_file_path, category, filename, prompt, max_tokens=2500):
    """
    Write one document to the volume using the resolution order:
      1. Already present  →  skip
      2. GitHub download  →  write
      3. Claude generate  →  write
    Returns the source used: 'exists', 'github', or 'generated'.
    """
    if file_exists_in_volume(volume_file_path):
        return "exists"

    content = try_github_download(category, filename)
    if content is not None:
        source = "github"
    else:
        content = generate_with_claude(prompt, max_tokens=max_tokens)
        source  = "generated"

    dbutils.fs.put(volume_file_path, content, overwrite=True)
    return source

# COMMAND ----------

# MAGIC %md
# MAGIC ## Runbooks (10 docs)

# COMMAND ----------

runbook_topics = [
    "Troubleshooting High VSWR on Sector Antennas - includes step-by-step diagnosis, common causes (water ingress, connector corrosion, cable damage), escalation criteria, and resolution procedures",
    "RAN Latency Troubleshooting Procedure - S1-U interface latency investigation, backhaul link diagnostics, core network path analysis, and performance baseline restoration",
    "Coverage Hole Investigation and Remediation - RF drive test procedures, antenna tilt optimization, neighbor cell parameter adjustment, and coverage prediction validation",
    "VoLTE Quality Degradation Response - MOS score investigation, codec parameter verification, IMS core health checks, and quality restoration procedures",
    "Network Congestion Management SOP - traffic pattern analysis, PRB utilization thresholds, load balancing activation, carrier aggregation optimization, and capacity planning triggers",
    "Handover Failure Troubleshooting - X2/Xn interface diagnostics, neighbor relation optimization, handover parameter tuning (A3 offset, TTT, hysteresis), and mobility robustness optimization",
    "Site Outage Recovery Procedure - power system diagnostics, equipment restart sequences, backhaul failover activation, and service restoration verification checklist",
    "Massive MIMO Beam Management Troubleshooting - beam pattern verification, CSI-RS configuration checks, codebook optimization, and coverage/capacity trade-off analysis",
    "5G NR NSA/SA Interworking Issues - EN-DC setup failure diagnosis, SgNB addition failure troubleshooting, split bearer configuration, and measurement gap optimization",
    "Backhaul Capacity Planning and Monitoring - fiber utilization trending, microwave link performance, Ethernet CRC error investigation, and capacity upgrade trigger criteria",
]

counts = {"exists": 0, "github": 0, "generated": 0}

print("Processing runbook documents...")
for i, topic in enumerate(runbook_topics):
    filename   = f"runbook_{i+1:02d}.txt"
    file_path  = f"{volume_path}/runbooks/{filename}"
    title      = topic.split(" - ")[0]
    prompt     = (
        f"Write a detailed network operations runbook titled: '{title}'\n\n"
        "The document should include:\n"
        "- Document ID and revision history\n"
        "- Purpose and scope\n"
        "- Prerequisites and tools needed\n"
        "- Detailed step-by-step procedures with specific CLI commands and parameter values\n"
        "- Decision trees for common scenarios\n"
        "- Escalation criteria and contacts\n"
        "- Verification and closure steps\n\n"
        f"Context: {topic}"
    )
    source = write_doc(file_path, "runbooks", filename, prompt)
    counts[source] += 1
    print(f"  {filename}: {source}")

print(f"\nRunbooks — exists:{counts['exists']}  github:{counts['github']}  generated:{counts['generated']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Standards Documents (5 docs)

# COMMAND ----------

standards_topics = [
    "3GPP TS 38.331 - RRC Protocol Specification for NR - Focus on measurement configuration, reporting criteria, handover procedures, and RRC state transitions relevant to network optimization",
    "3GPP TS 23.501 - 5G System Architecture - Network function descriptions (AMF, SMF, UPF), service-based interfaces, PDU session management, and QoS framework",
    "O-RAN WG4 - Open Fronthaul Interface Specification - Split 7.2x architecture, eCPRI transport, synchronization requirements, and C/U-plane message formats",
    "3GPP TS 28.552 - Performance Measurements for 5G NR - KPI definitions for coverage, capacity, accessibility, retainability, and mobility, including measurement procedures and formulas",
    "O-RAN WG2 - AI/ML Workflow Description - Near-RT RIC architecture, xApp lifecycle, A1 policy management, conflict resolution, and ML model deployment procedures",
]

counts = {"exists": 0, "github": 0, "generated": 0}

print("Processing standards documents...")
for i, topic in enumerate(standards_topics):
    filename  = f"standard_{i+1:02d}.txt"
    file_path = f"{volume_path}/standards/{filename}"
    title     = topic.split(" - ")[0]
    prompt    = (
        f"Write a technical summary document of the following telecommunications standard: '{title}'\n\n"
        "The document should include:\n"
        "- Standard reference number and version\n"
        "- Scope and applicability\n"
        "- Key definitions and acronyms\n"
        "- Technical specifications and parameters with specific values\n"
        "- Relevant procedures and message flows\n"
        "- Implementation notes for network operators\n\n"
        f"Context: {topic}\n\n"
        "Note: This is a summary/extract for operational reference, not the full standard."
    )
    source = write_doc(file_path, "standards", filename, prompt)
    counts[source] += 1
    print(f"  {filename}: {source}")

print(f"\nStandards — exists:{counts['exists']}  github:{counts['github']}  generated:{counts['generated']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Incident / RCA Reports (8 docs)

# COMMAND ----------

incident_topics = [
    "RCA: Pacific Northwest Region Latency Spike - June 2026 - Root cause was S1-U interface congestion due to misconfigured QoS bearer mapping after planned software upgrade. Affected 12 sites for 36 hours. Resolution involved rolling back QCI mapping tables and implementing pre-change validation checklist.",
    "RCA: Southern California Coverage Degradation - May 2026 - Antenna VSWR alarm triggered by water ingress in Type-N connectors after heavy rain. 5 sites affected, coverage dropped 4-6% in affected sectors. Resolution required connector replacement and weatherproofing upgrade.",
    "RCA: Mountain West Handover Storm - April 2026 - Ping-pong handovers between overlapping NR and LTE cells caused by aggressive A3 offset parameters. User complaints about dropped VoLTE calls. Resolution involved hysteresis adjustment from 1dB to 3dB and TTT increase from 160ms to 320ms.",
    "RCA: Northeast Region Complete Outage - March 2026 - Fiber cut on primary backhaul ring due to third-party construction. 28 sites lost connectivity for 4 hours until DWDM protection switching was manually activated. Post-incident: automated protection switching configuration deployed.",
    "RCA: Great Plains Throughput Degradation - May 2026 - PRB utilization exceeded 85% during regional event (college graduation). Carrier aggregation not activated on 3 high-traffic sites due to configuration oversight. Resolution: enabled CA on affected sites, added to capacity planning watch list.",
    "RCA: Southeast VoLTE Quality Degradation - April 2026 - AMR-WB codec negotiation failing intermittently due to IMS core software bug. MOS scores dropped from 4.2 to 3.1 for affected calls. Vendor patch applied within 48 hours.",
    "RCA: Pacific Northwest 5G SA Registration Failures - February 2026 - AMF overload during peak hours causing attach timeouts. Root cause was undersized AMF pool after subscriber migration from NSA. Resolution: scaled AMF instances from 4 to 8, implemented load-based auto-scaling.",
    "RCA: Multi-Region DNS Resolution Delay - January 2026 - Recursive DNS servers experiencing cache poisoning from misconfigured DNSSEC validation. Affected all regions intermittently. Resolution involved DNS cache flush, DNSSEC validation rule update, and monitoring enhancement.",
]

counts = {"exists": 0, "github": 0, "generated": 0}

print("Processing incident/RCA documents...")
for i, topic in enumerate(incident_topics):
    filename  = f"incident_{i+1:02d}.txt"
    file_path = f"{volume_path}/incidents/{filename}"
    parts     = topic.split(" - ")
    title     = f"{parts[0]} - {parts[1]}"
    prompt    = (
        f"Write a detailed Root Cause Analysis (RCA) report titled: '{title}'\n\n"
        "The document should include:\n"
        "- Incident ID, severity, and duration\n"
        "- Executive summary\n"
        "- Timeline of events (detection, investigation, resolution)\n"
        "- Impact assessment (affected KPIs with specific numbers, customer impact)\n"
        "- Root cause analysis (5-whys or fishbone approach)\n"
        "- Corrective actions taken\n"
        "- Preventive measures and recommendations\n"
        "- Lessons learned\n\n"
        f"Context: {topic}"
    )
    source = write_doc(file_path, "incidents", filename, prompt)
    counts[source] += 1
    print(f"  {filename}: {source}")

print(f"\nIncidents — exists:{counts['exists']}  github:{counts['github']}  generated:{counts['generated']}")

# COMMAND ----------

# Verify totals
runbook_count  = len(dbutils.fs.ls(f"{volume_path}/runbooks/"))
standards_count = len(dbutils.fs.ls(f"{volume_path}/standards/"))
incidents_count = len(dbutils.fs.ls(f"{volume_path}/incidents/"))

print(f"\nDocument load complete:")
print(f"  Runbooks   : {runbook_count}")
print(f"  Standards  : {standards_count}")
print(f"  Incidents  : {incidents_count}")
print(f"  Total      : {runbook_count + standards_count + incidents_count}")
print()
print("If any docs were generated with Claude, run:")
print("  ./scripts/pull_docs_from_volume.sh <catalog> <schema> <profile>")
print("then commit docs/ to GitHub to enable pre-built downloads for future deployments.")
