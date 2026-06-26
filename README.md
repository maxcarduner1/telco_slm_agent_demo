# OTel SLM Agent Demo — Telco Network Analytics Assistant

A production-ready agentic telecom network operations assistant ("TelcoGPT") built on Databricks. It demonstrates how **Open Telco (OTel) Small Language Models** can power cost-efficient, domain-specialized RAG while a frontier model (Claude Sonnet 4) handles supervisor orchestration — all deployed as a full-stack Databricks App with long-term memory.

## Overview

**TelcoGPT** answers questions about network health by combining:

- **Live KPI data** — Structured queries via Unity Catalog SQL functions (throughput, latency, coverage, dropped calls, VoLTE quality, churn risk)
- **Domain-specific RAG** — Retrieval over operational runbooks, 3GPP/O-RAN standards, and incident post-mortems using OTel SLMs
- **Multi-agent orchestration** — LangGraph supervisor routes to specialized sub-agents, each backed by a dedicated Vector Search index
- **Persistent memory** — Lakebase (PostgreSQL autoscale) for both conversation state and long-term user preferences

### Key Design Goals

| Goal | How |
|------|-----|
| Cost-efficient RAG | OTel SLMs (335M–1.2B params) handle all retrieval/reranking/generation |
| No hallucinations | OTel-LLM trained to abstain when context is insufficient |
| Single GPU footprint | Full OTel stack (~4.3 GB VRAM) fits on `GPU_SMALL` endpoint |
| Frontier orchestration | Claude Sonnet 4 for intent routing, planning, and final synthesis |
| Portability | Self-contained DAB bundle; deploys to any workspace |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  DATABRICKS APP (Apps Compute)               │
│           React Chat UI  +  FastAPI (ResponseAgent)          │
└─────────────────────────┬───────────────────────────────────┘
                           │
                           v
┌─────────────────────────────────────────────────────────────┐
│               LANGGRAPH SUPERVISOR AGENT                     │
│             (Frontier Model — Claude Sonnet 4)               │
│                                                             │
│  ┌─────────────────────┐  ┌──────────────────────────┐     │
│  │  Short-term Memory  │  │   Long-term Memory        │     │
│  │  (LangGraph state)  │  │   (Lakebase PostgreSQL)   │     │
│  └─────────────────────┘  └──────────────────────────┘     │
└────┬──────────┬──────────┬───────────────┬─────────────────┘
     │          │          │               │
     v          v          v               v
┌─────────┐ ┌────────┐ ┌──────────┐ ┌──────────────┐
│  SQL    │ │Runbook │ │Standards │ │  Incident/   │
│  Agent  │ │  RAG   │ │   RAG    │ │   RCA RAG    │
│(KPI SQL)│ │ (SLM)  │ │  (SLM)  │ │    (SLM)     │
└────┬────┘ └───┬────┘ └────┬─────┘ └──────┬───────┘
     │          │           │              │
     v          v           v              v
  UC SQL      VS Index    VS Index      VS Index
  Functions   Runbooks    3GPP/O-RAN   Incidents
     │
     v
 Delta Tables
 (KPI Data)
```

### OTel SLM Stack

All RAG operations use HuggingFace OTel models (Apache 2.0, trained on 326K+ telecom samples):

| Model | Role | VRAM |
|-------|------|------|
| `OTel-Embedding-335M` | Dense retrieval embeddings | 0.7 GB |
| `OTel-Reranker-0.6B` | Cross-encoder reranking | 1.2 GB |
| `OTel-LLM-1.2B-IT` | Domain-optimized generation | 2.4 GB |
| **Total** | | **~4.3 GB** |

---

## Demo Use Cases

| Use Case | Description |
|----------|-------------|
| **UC1 — Network Health Summary** | "How is my network today?" → KPI metrics → threshold flags |
| **UC2 — Root Cause Analysis** | "What's causing this latency spike?" → Multi-agent flow (KPIs + incidents + runbooks) → synthesized RCA |
| **UC3 — Remediation Guidance** | "How do I fix this?" → Runbook + standards agents → step-by-step procedures |

---

## Project Structure

```
Otel_SLM_Agent_Demo/
├── notebooks/                        # Data pipeline & infrastructure
│   ├── 00_provision_endpoints.py     # Auto-provision VS + OTel endpoints (runs in parallel)
│   ├── 01_generate_kpi_data.py       # Synthetic network KPI data (50 sites, 6 regions, 90 days)
│   ├── 02_generate_documents.py      # Load telco docs from GitHub, fall back to Claude generation
│   ├── 03_parse_documents.py         # Chunk documents for Vector Search
│   ├── 04_create_vs_indexes.py       # Vector Search indexes with OTel-Embedding-335M
│   ├── 05_create_uc_functions.py     # UC SQL functions as agent tools
│   ├── 06_test_uc_functions.py       # Validate UC function outputs
│   └── 07_provision_lakebase_app.py  # Lakebase memory setup
├── docs/                             # Pre-generated telco documents (committed to repo)
│   ├── runbooks/                     # 10 operational runbooks
│   ├── standards/                    # 5 standards summaries (3GPP, O-RAN)
│   └── incidents/                    # 8 incident RCA reports
├── agent_app/                        # LangGraph agent application
│   ├── agent.py                      # ReAct agent with LangGraph + ResponseAgent
│   ├── server.py                     # FastAPI/uvicorn entry point
│   ├── tools.py                      # UC function tools + RAG search tools
│   ├── prompts.py                    # TelcoGPT system prompt
│   └── memory.py                     # Lakebase checkpointing + long-term store
├── e2e-chatbot-app-next/             # React + Express.js chat UI (full-stack)
├── scripts/
│   ├── start_app.py                  # Start the chat application
│   └── pull_docs_from_volume.sh      # Pull generated docs from UC Volume → docs/ for committing
├── databricks.yml                    # DAB bundle (data setup job + app resource)
├── app.yaml                          # Chat app runtime config
├── pyproject.toml                    # Python dependencies
├── requirements.txt                  # Additional runtime dependencies
└── design_doc.md                     # Full architecture & implementation guide
```

---

## Prerequisites

- Databricks workspace with Unity Catalog enabled
- Databricks CLI configured (`databricks configure --profile <your-profile>`)
- Serverless compute enabled
- GPU endpoint capacity for OTel model serving (`GPU_SMALL`)
- Lakebase (PostgreSQL autoscale) available in your workspace

### Endpoints

All OTel model serving endpoints and the Vector Search endpoint are **automatically provisioned** by `notebooks/00_provision_endpoints.py`, which runs as the first task in the data setup job (in parallel with the data pipeline). You do not need to create these manually.

| Endpoint | How it's created |
|----------|-----------------|
| `demo_telco_vs_endpoint` | Auto — `00_provision_endpoints.py` |
| `otel-embedding2-300m` | Auto — downloads OTel-Embedding-335M from HuggingFace, registers to UC, deploys GPU_SMALL |
| `otel-reranker-600m` | Auto — downloads OTel-Reranker-0.6B from HuggingFace, registers to UC, deploys GPU_SMALL |
| `otel-llm-1b-it` | Auto — downloads OTel-LLM-1.2B-IT from HuggingFace, registers to UC, deploys GPU_SMALL |
| `databricks-claude-sonnet-4` | **Manual** — built-in PAYG endpoint; available in most workspaces automatically |

**Supervisor model for workspaces without `databricks-claude-sonnet-4`:** Section 5 of `00_provision_endpoints.py` contains a placeholder to create an Anthropic external model endpoint wrapped with an AI Gateway. Set the `secret_scope` widget (pointing to a Databricks Secret holding your Anthropic API key) to activate it, then update `LLM_ENDPOINT` in `app.yaml` and `databricks.yml` to the gateway endpoint name.

---

## Deployment

### 1. Configure environment variables

Update `databricks.yml` and `app.yaml` with your workspace-specific values:

```yaml
UC_CATALOG: your_catalog
UC_SCHEMA: your_schema
DATABRICKS_WAREHOUSE_ID: your_warehouse_id
LAKEBASE_PROJECT: your-lakebase-project
VS_ENDPOINT: your_vs_endpoint
```

### 2. Deploy the DAB bundle

```bash
# Authenticate
databricks auth login --profile <your-profile>

# Deploy resources (job + app)
databricks bundle deploy --profile <your-profile>
```

### 3. Run the data setup job

```bash
databricks jobs run-now --job-name otel-demo-data-setup --profile <your-profile>
```

The job runs the following tasks. Tasks with no `depends_on` start immediately in parallel:

| Task | Depends on | What it does |
|------|-----------|--------------|
| `provision_endpoints` | — | Creates VS endpoint + deploys OTel embedding/reranker/LLM models to GPU_SMALL; provisions external Claude gateway if configured |
| `generate_kpi_data` | — | Generates 90-day synthetic KPI Delta tables (50 sites, 6 regions) |
| `generate_documents` | `generate_kpi_data` | Loads 23 telco docs from GitHub repo; falls back to Claude generation if not found |
| `parse_documents` | `generate_documents` | Chunks documents for Vector Search |
| `create_vs_indexes` | `parse_documents` | Embeds chunks with OTel-Embedding-335M, creates 3 Delta Sync VS indexes |
| `create_uc_functions` | `generate_kpi_data` | Registers 5 UC SQL functions as agent tools |
| `provision_lakebase_app` | — | Provisions Lakebase PostgreSQL for agent memory; creates Databricks App |

`provision_endpoints` and `provision_lakebase_app` run in parallel with the data pipeline at job start. By the time `create_vs_indexes` needs the VS endpoint and embedding model (~30–45 min in), both are ready.

### 4. Deploy and start the app

```bash
databricks bundle run telco_agent --profile <your-profile>
```

### Repairing a failed run

If a multi-task job fails partway through, repair from the failure point rather than re-running from scratch:

```bash
databricks jobs repair-run <RUN_ID> --rerun-all-failed-tasks --profile <your-profile>
```

Run ID is in the job run URL after `run/`.

---

## Data Model

### KPI Tables (generated by notebook 01)

| Table | Description |
|-------|-------------|
| `network_kpis_hourly` | 90-day hourly KPI timeseries — 50 sites × 6 regions |
| `network_events` | Network events and anomalies |
| `customer_churn_daily` | Daily churn risk scores by site |

**KPIs tracked:** throughput (DL/UL), latency, coverage (RSRP/RSRQ), dropped call rate, handover success rate, VoLTE quality

### UC Functions (agent tools)

| Function | Description |
|----------|-------------|
| `get_kpi_metrics` | Query KPIs for a site/region/time range |
| `get_threshold_breaches` | Identify KPIs violating configured thresholds |
| `compare_regions` | Side-by-side regional KPI comparison |
| `get_network_events` | Retrieve network events/anomalies |
| `get_churn_risk` | Customer churn risk scores |

### Vector Search Indexes

| Index | Content | Embedding Model |
|-------|---------|-----------------|
| `runbooks_vs_index` | Operational runbooks and SOPs | OTel-Embedding-335M |
| `standards_vs_index` | 3GPP/O-RAN specifications | OTel-Embedding-335M |
| `incidents_vs_index` | Incident reports and post-mortems | OTel-Embedding-335M |

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph 1.1+ |
| Frontier LLM | Claude Sonnet 4 (`databricks-claude-sonnet-4`) |
| OTel SLMs | OTel-Embedding-335M, OTel-Reranker-0.6B, OTel-LLM-1.2B-IT |
| Agent framework | Databricks AI Bridge (ResponseAgent) |
| Vector Search | Databricks Vector Search |
| Memory | Lakebase autoscale (PostgreSQL) |
| API server | FastAPI + Uvicorn |
| Chat UI | React 18 + Vite + Tailwind CSS |
| UI backend | Express.js + Vercel AI SDK |
| Observability | MLflow 3.0 tracing |
| Infrastructure | Databricks Asset Bundles (DAB) |
| Compute | Databricks Apps + Serverless Jobs |

---

## Configuration Reference

All configuration is driven by environment variables (set in `databricks.yml` and `app.yaml`):

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_ENDPOINT` | Frontier model endpoint | `databricks-claude-sonnet-4` |
| `EMBEDDING_ENDPOINT` | OTel embedding endpoint | `otel-embedding2-300m` |
| `VS_ENDPOINT` | Vector Search endpoint name | `demo_telco_vs_endpoint` |
| `LAKEBASE_PROJECT` | Lakebase project for memory | `telco-slm-agent-memory` |
| `LAKEBASE_BRANCH` | Lakebase branch | `production` |
| `LAKEBASE_DATABASE` | Memory database name | `agent_memory` |
| `UC_CATALOG` | Unity Catalog catalog | `cmegdemos_catalog` |
| `UC_SCHEMA` | Unity Catalog schema | `network_analytics_enablement` |
| `DATABRICKS_WAREHOUSE_ID` | SQL warehouse for UC functions | — |

---

## Development

### Local setup

```bash
pip install -e .
```

### Running the agent server locally

```bash
uvicorn agent_app.server:app --host 0.0.0.0 --port 8000 --reload
```

### Running the chat UI locally

```bash
cd e2e-chatbot-app-next
npm install
npm run dev
```

See [e2e-chatbot-app-next/README.md](e2e-chatbot-app-next/README.md) for full chat app documentation.

---

## References

- [Design Document](design_doc.md) — Full architecture, data schemas, implementation steps
- [OTel HuggingFace Models](https://huggingface.co/OTel) — Open Telco SLM model cards
- [Databricks App Templates — LangGraph Advanced](https://github.com/databricks/app-templates/blob/main/agent-langgraph-advanced/README.md)
- [Databricks Asset Bundles](https://docs.databricks.com/dev-tools/bundles/index.html)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
