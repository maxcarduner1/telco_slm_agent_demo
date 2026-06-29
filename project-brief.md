# TelcoGPT — Project Brief

**OTel SLM Agent Demo: Cost-Efficient Agentic AI for Telecom Network Operations**

---

## What Is TelcoGPT?

TelcoGPT is a solution accelerator for an agentic AI assistant for telecom network operations, built entirely on Databricks. It gives network operations teams a conversational interface for answering three classes of questions:

- **"How is my network right now?"** — Live KPI health summaries with threshold flags
- **"What's causing this degradation?"** — AI-synthesized root cause analysis combining live data with historical incidents and runbooks
- **"How do I fix it?"** — Step-by-step remediation guidance drawn from operational runbooks and 3GPP/O-RAN standards

The system is designed around a key architectural principle: **use the smallest model capable of each task**. Domain-specialized Small Language Models (OTel SLMs) handle all retrieval-augmented generation, while a frontier model (Claude Sonnet 4) is reserved only for high-level orchestration.

---

## Architecture Overview

TelcoGPT uses a **multi-agent, multi-tier** architecture where each layer is sized for its task.

```
┌──────────────────────────────────────────────────────────────┐
│                  DATABRICKS APP                              │
│          React Chat UI  +  FastAPI Agent Server              │
└─────────────────────┬────────────────────────────────────────┘
                      │ HTTP (streaming)
                      v
┌──────────────────────────────────────────────────────────────┐
│            SUPERVISOR AGENT  (Claude Sonnet 4)               │
│                                                              │
│  Responsibilities:                                           │
│  • Classify user intent, plan multi-step queries             │
│  • Route to the right sub-agent(s) or tool(s)               │
│  • Synthesize final answer from sub-agent results            │
│  • Manage conversation state + long-term user context        │
│                                                              │
│  Memory:  Short-term (LangGraph)  +  Long-term (Lakebase)   │
└────┬──────────┬──────────┬───────────────┬───────────────────┘
     │          │          │               │
     v          v          v               v
 ┌────────┐ ┌────────┐ ┌──────────┐ ┌──────────────┐
 │  KPI   │ │Runbook │ │Standards │ │  Incident    │
 │ Query  │ │  RAG   │ │   RAG    │ │  / RCA RAG   │
 │ Agent  │ │ (OTel) │ │  (OTel) │ │   (OTel)     │
 └───┬────┘ └───┬────┘ └────┬─────┘ └──────┬───────┘
     │          │           │              │
     v          v           v              v
 UC SQL       VS Index    VS Index      VS Index
 Functions    Runbooks    3GPP/O-RAN   Incidents
     │
     v
 Delta Tables
 (3.6M KPI rows,
  90 days × 50 sites)
```

### Layers

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Chat UI** | React 18 + Vite + Tailwind CSS, Express.js backend | Streaming conversational interface |
| **Agent server** | FastAPI + Uvicorn (Databricks Apps) | Hosts LangGraph ReAct agent, exposes `/chat` API |
| **Supervisor LLM** | Claude Sonnet 4 (`databricks-claude-sonnet-4`) | Intent routing, multi-step planning, final synthesis |
| **OTel SLM stack** | OTel-Embedding-335M + OTel-Reranker-0.6B + OTel-LLM-1.2B-IT | All RAG operations (embed → retrieve → rerank → generate) |
| **Vector Search** | Databricks Vector Search (3 indexes) | Semantic retrieval over runbooks, standards, and incident reports |
| **Structured data** | 5 UC SQL Functions over Delta tables | Live KPI queries, threshold detection, regional comparison, churn risk |
| **Memory** | Lakebase (PostgreSQL autoscale) | Conversation checkpointing + cross-session user preferences |
| **Observability** | MLflow 3.0 | Full trace capture for every agent turn, tool call, and sub-agent response |
| **Infrastructure** | Databricks Asset Bundles (DAB) | One-command deploy of jobs + app + resources |

### Data Pipeline

A 7-task serverless Databricks job provisions all demo data:

1. **Generate KPI data** — 3.6M rows of synthetic hourly telco metrics (50 sites, 6 regions, 90 days) with injected anomalies
2. **Generate documents** — 23 telco PDFs (runbooks, 3GPP/O-RAN standards, incident RCAs) written by Claude Sonnet 4
3. **Parse & chunk** — Section-aware splitting at 512 tokens with 64-token overlap
4. **Create Vector Search indexes** — Pre-compute 768-dim OTel embeddings, sync to 3 Delta Sync VS indexes
5. **Register UC functions** — 5 typed SQL functions exposed as agent tools
6. **Validate** — Automated test queries against all UC functions
7. **Provision Lakebase** — Idempotent setup of PostgreSQL autoscale instance for agent memory

---

## The OTel SLM Advantage

### What Are OTel Models?

The [Open Telco (OTel) models](https://huggingface.co/farbodtavakkoli) are a family of HuggingFace models (Apache 2.0) fine-tuned on 326,000+ telecom-domain samples — 3GPP specifications, network runbooks, incident reports, and telco operational data. They are purpose-built for the **embed → retrieve → rerank → generate** RAG pipeline.

| Model | Base Architecture | Size | VRAM | Role |
|-------|-----------------|------|------|------|
| `OTel-Embedding-335M` | BAAI/bge-large-en-v1.5 | 335M params | 0.7 GB | Dense retrieval embeddings |
| `OTel-Reranker-0.6B` | Qwen/Qwen3-0.6B | 600M params | 1.2 GB | Cross-encoder reranking (top-20 → top-5) |
| `OTel-LLM-1.2B-IT` | google/gemma-3-1b-it | 1.2B params | 2.4 GB | Domain-grounded answer generation |
| **Total stack** | | | **~4.3 GB** | Fits on one `GPU_SMALL` endpoint |

### Why OTel Instead of a Frontier Model for RAG?

#### 1. Domain Precision
General-purpose embedding models (e.g., `text-embedding-3-large`) represent telco concepts like _VSWR_, _handover failure_, and _O-RAN WG4 fronthaul_ in a broad semantic space shared with unrelated text. OTel-Embedding-335M was fine-tuned specifically on this vocabulary and these document types — retrieval quality is meaningfully higher for domain queries.

#### 2. Abstention Over Hallucination
`OTel-LLM-1.2B-IT` was trained to **decline to answer** when the retrieved context does not contain sufficient information to support a response. In a network operations context — where a wrong remediation step can take down infrastructure — this is non-negotiable. Frontier models are optimized to produce fluent answers; OTel is optimized to produce _correct or nothing_.

#### 3. Fixed Cost, Not Variable Token Spend
All three OTel models run on a single `GPU_SMALL` endpoint at ~**$301/month**, regardless of query volume. A frontier model used for RAG generation charges per token on every sub-agent call. At moderate query volumes, the fixed cost model breaks even around 10K RAG calls/month and becomes significantly cheaper above that threshold (see Cost Analysis below).

#### 4. Single GPU Footprint
The full OTel RAG stack — embedding, reranking, and generation — fits in **4.3 GB VRAM**, well within a single `GPU_SMALL` (24 GB A10). This enables a compact, manageable serving footprint with no model orchestration overhead.

#### 5. Low Inference Latency
At 335M–1.2B parameters, OTel models respond in ~50–150ms per call. The full retrieve-rerank-generate pipeline typically completes in under 500ms, keeping total end-to-end agent latency under 10 seconds even for multi-hop queries.

#### 6. Data Sovereignty
All OTel inference runs within your Databricks workspace — no data leaves to an external API. For telcos with strict data handling requirements around network topology, customer churn, and operational procedures, this is critical.

#### 7. License Clarity
Apache 2.0 means unrestricted commercial use, redistribution, and modification. No seat licenses, usage caps, or legal review required to embed in a production product.

### Division of Labor: OTel vs. Claude

The architecture deliberately separates concerns:

| Responsibility | Model | Why |
|---------------|-------|-----|
| Intent classification, multi-step planning | Claude Sonnet 4 | Requires broad reasoning, tool selection across 8 diverse tools |
| Final answer synthesis from sub-agent outputs | Claude Sonnet 4 | Requires coherent prose generation integrating structured + unstructured evidence |
| Telco document embedding (indexing) | OTel-Embedding-335M | Domain-tuned vector space → better retrieval recall |
| Query-time embedding | OTel-Embedding-335M | Consistent embedding space with indexed documents |
| Cross-encoder reranking | OTel-Reranker-0.6B | Precision boost without generating tokens |
| Grounded generation from retrieved context | OTel-LLM-1.2B-IT | Domain-aware generation with abstention when context is absent |

Claude only sees distilled, pre-retrieved context — its token consumption is bounded by the supervisor turn, not by raw document retrieval.

---

## Cost Analysis

### Estimated Monthly Infrastructure Costs

| Component | Pricing Type | Estimated Cost | Notes |
|-----------|-------------|---------------|-------|
| **OTel SLM endpoint** (`GPU_SMALL`) | Fixed | ~$301/month | All 3 models; 24/7 serving |
| **Databricks Apps compute** | Serverless DBU | ~$15/month | 8 hrs/day × 22 working days = 176 active hrs; ~0.5 DBU/hr at ~$0.07/DBU + standby |
| **Vector Search endpoint** | Shared/workspace | Minimal incremental | Assumes shared endpoint |
| **Serverless job compute** | Per-run | ~$5–10 one-time | Data setup job (~45 min) |
| **Lakebase (agent memory)** | Variable (autoscale) | $0–$504/month | See note below |
| **Claude Sonnet 4 (supervisor)** | Pay-per-token | Variable | See note below |

**Databricks Apps compute:** The FastAPI/uvicorn agent server runs as a Databricks App on serverless compute. Assuming **8 active hours/day across 22 working days (176 hours/month)**, a small app consuming ~0.5 DBU/hr at the standard serverless rate (~$0.07/DBU) gives ~$6 in pure compute, plus a small standby/idle charge, totalling approximately **$15/month**. Outside business hours the app scales to near-zero. Continuous 24/7 deployment would cost ~$45/month.

**Lakebase:** Autoscales from minimum compute (~$0.07/hr at idle) to CU_1 ($0.70/hr) and above under load. For a demo scenario with light traffic, expect $20–100/month. Persistent chat history can be disabled entirely for ephemeral-mode operation at zero cost.

**Claude Sonnet 4 (supervisor only):** Each agent turn consumes roughly 1,500–2,500 input tokens and 300–600 output tokens for the supervisor. At Databricks pay-as-you-go pricing:

| Monthly Query Volume | Estimated Claude Cost |
|---------------------|----------------------|
| 1,000 queries | ~$15 |
| 5,000 queries | ~$75 |
| 10,000 queries | ~$150 |

**Total estimated cost (moderate usage, 5K queries/month, 8 hrs/day active):**

| Component | Demo | Production |
|-----------|------|-----------|
| OTel SLM endpoint (GPU_SMALL, 24/7) | $301 | $301 |
| Databricks Apps compute (8 hrs/day) | $15 | $15 |
| Claude Sonnet 4 supervisor (5K queries) | $75 | $75 |
| Lakebase memory | $0 (ephemeral) | $20–$100 |
| **Total** | **~$390** | **~$410–$490** |

### Cost Comparison: OTel SLMs vs. All-Frontier RAG

If OTel-LLM-1.2B-IT were replaced with Claude Sonnet 4 for RAG generation, each query would incur 3 additional frontier model calls (one per sub-agent: runbook, standards, incidents), each processing ~1,200 retrieved tokens plus generating ~400 output tokens.

| Monthly Queries | OTel Architecture | All-Frontier RAG | Monthly Savings |
|----------------|-------------------|-----------------|----------------|
| 5,000 | ~$390 | ~$575 | ~$185 |
| 10,000 | ~$465 | ~$835 | ~$370 |
| 50,000 | ~$615 | ~$3,135 | ~$2,520 |
| 100,000 | ~$765 | ~$6,105 | ~$5,340 |

The OTel fixed cost makes the architecture strongly favorable at any sustained query volume above ~8K/month. Below that volume, cost difference is small but OTel still wins on latency and abstention quality.

> Estimates use approximate Databricks PAYG token pricing for Claude Sonnet 4. Actual costs depend on workspace agreements, query complexity, and usage patterns.

---

## Infrastructure Requirements

### Pre-Provisioned (Workspace)

| Resource | Purpose |
|----------|---------|
| `otel-embedding2-300m` (Model Serving) | Embedding endpoint for VS index creation and query-time embedding |
| `otel-llm-1b-it` (Model Serving) | Reranker + generation endpoint (`GPU_SMALL`) |
| `demo_telco_vs_endpoint` (Vector Search) | Hosts the 3 domain indexes |
| SQL Warehouse | Executes UC function queries |

### Provisioned by DAB

| Resource | Provisioned By |
|----------|---------------|
| Delta tables (KPI data) | Data setup job |
| UC Volume + document files | Data setup job |
| Vector Search indexes (3) | Data setup job |
| UC SQL Functions (5) | Data setup job |
| Lakebase instance | Data setup job |
| Databricks App (FastAPI agent) | `databricks bundle deploy` |

### Compute

- **Data pipeline:** Serverless (client 5), no cluster management
- **Agent app:** Databricks Apps compute (serverless, no idle cost)
- **OTel models:** `GPU_SMALL` endpoint (A10, 24 GB VRAM)

---

## Demo Flow

Three representative conversations exercise the full architecture:

**UC1 — Network Health Summary**
> "Give me a network health summary for the Pacific Northwest."

Agent routes to UC SQL functions → queries `network_kpis_hourly` → identifies threshold breaches → returns formatted KPI table with red/green status flags.

**UC2 — Root Cause Analysis**
> "We have elevated latency in the Pacific Northwest. What's causing it?"

Agent runs in parallel: UC functions (live KPI + event correlation) + incident index (historical RCA patterns for PNW latency) + runbook index (VSWR and backhaul troubleshooting). Claude synthesizes a ranked list of probable causes with supporting evidence citations.

**UC3 — Remediation Guidance**
> "Walk me through fixing the VSWR issue."

Agent searches runbook and standards indexes → OTel-LLM generates step-by-step procedure grounded in retrieved SOPs → Claude formats into numbered remediation plan with doc references.

---

## Summary

TelcoGPT demonstrates that **frontier models and domain SLMs are complementary, not competing**. Claude Sonnet 4 provides the reasoning and synthesis capability that makes the agent genuinely useful; OTel SLMs provide the cost efficiency, domain precision, and operational safety (abstention) that make it deployable at production scale.

The result is a full-stack agentic AI system that runs on Databricks for under $600/month at moderate query volumes — with all data, compute, and model inference remaining within the workspace.
