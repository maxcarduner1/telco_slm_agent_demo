# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — Provision Model Serving Endpoints
# MAGIC
# MAGIC Idempotently checks and provisions all inference infrastructure required by the
# MAGIC OTel SLM Agent Demo. Safe to re-run — every section skips if already deployed.
# MAGIC
# MAGIC | # | Resource | Type | Purpose |
# MAGIC |---|----------|------|---------|
# MAGIC | 1 | `demo_telco_vs_endpoint` | Vector Search | Hosts the three RAG indexes |
# MAGIC | 2 | `otel-embedding2-300m` | GPU_SMALL serving | Telco-domain embeddings (OTel-Embedding-335M) |
# MAGIC | 3 | `otel-reranker-600m` | GPU_SMALL serving | Cross-encoder reranking (OTel-Reranker-0.6B) |
# MAGIC | 4 | `otel-llm-1b-it` | GPU_SMALL serving | Domain-grounded generation (OTel-LLM-1.2B-IT) |
# MAGIC | 5 | `telco-supervisor-gateway` | External model + AI Gateway | Claude supervisor — **see PLACEHOLDER section** |
# MAGIC
# MAGIC **Run order:** No dependencies on other notebooks. Designed to run in parallel with
# MAGIC `01_generate_kpi_data` and `07_provision_lakebase_app` at job start.
# MAGIC
# MAGIC `04_create_vs_indexes` needs items 1 and 2 to be READY. Since task 04 follows the
# MAGIC chain 01→02→03→04 (typically 30–45 min), those endpoints will be ready in time.
# MAGIC
# MAGIC **First-run note:** Sections 2–4 download ~5 GB of HuggingFace model weights,
# MAGIC log them to MLflow, and register them in Unity Catalog. This takes ~20–30 min
# MAGIC on first run. Subsequent runs detect existing endpoints and skip immediately.

# COMMAND ----------

dbutils.widgets.text("catalog",            "cmegdemos_catalog",            "Catalog")
dbutils.widgets.text("schema",             "network_analytics_enablement", "Schema")
dbutils.widgets.text("vs_endpoint",        "demo_telco_vs_endpoint",       "VS Endpoint")
dbutils.widgets.text("embedding_endpoint", "otel-embedding2-300m",         "Embedding Endpoint")
dbutils.widgets.text("reranker_endpoint",  "otel-reranker-600m",           "Reranker Endpoint")
dbutils.widgets.text("llm_endpoint",       "otel-llm-1b-it",               "LLM Endpoint")
# ── Supervisor / External Model (Section 5) ───────────────────────────────
dbutils.widgets.text("gateway_endpoint",   "telco-supervisor-gateway",     "Supervisor Gateway Endpoint")
dbutils.widgets.text("secret_scope",       "",                             "Secret Scope (Anthropic key)")
dbutils.widgets.text("secret_key",         "anthropic_api_key",            "Secret Key  (Anthropic key)")
dbutils.widgets.text("claude_model_name",  "claude-sonnet-4-5",            "Anthropic Model ID")

# COMMAND ----------

import json
import time
import tempfile
import requests
import mlflow
from mlflow.tracking import MlflowClient

catalog           = dbutils.widgets.get("catalog")
schema            = dbutils.widgets.get("schema")
vs_endpoint       = dbutils.widgets.get("vs_endpoint")
embedding_ep      = dbutils.widgets.get("embedding_endpoint")
reranker_ep       = dbutils.widgets.get("reranker_endpoint")
llm_ep            = dbutils.widgets.get("llm_endpoint")
gateway_ep        = dbutils.widgets.get("gateway_endpoint")
secret_scope      = dbutils.widgets.get("secret_scope")
secret_key        = dbutils.widgets.get("secret_key")
claude_model_name = dbutils.widgets.get("claude_model_name")

ws_url  = spark.conf.get("spark.databricks.workspaceUrl")
token   = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
_headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

mlflow.set_registry_uri("databricks-uc")
_mlflow_client = MlflowClient(registry_uri="databricks-uc")


def api(method, path, body=None, timeout=120):
    """Call the Databricks REST API."""
    return requests.request(
        method, f"https://{ws_url}/{path}",
        headers=_headers, json=body, timeout=timeout,
    )


def get_serving_state(ep_name):
    """Return (exists: bool, ready_state: str)."""
    r = api("GET", f"api/2.0/serving-endpoints/{ep_name}")
    if r.status_code == 404:
        return False, ""
    r.raise_for_status()
    return True, r.json().get("state", {}).get("ready", "NOT_READY")


def poll_serving_ready(ep_name, timeout_secs=1800, interval_secs=30):
    """Block until endpoint ready=READY or raise TimeoutError."""
    print(f"  Polling '{ep_name}' for READY (timeout {timeout_secs // 60} min)...")
    deadline = time.time() + timeout_secs
    while time.time() < deadline:
        _, state = get_serving_state(ep_name)
        print(f"    ready={state}")
        if state == "READY":
            return
        time.sleep(interval_secs)
    raise TimeoutError(f"'{ep_name}' not READY after {timeout_secs}s — check Serving UI")


def get_latest_uc_model_version(uc_model_name):
    """Return the latest registered version string for a UC model, or None."""
    try:
        versions = _mlflow_client.search_model_versions(f"name='{uc_model_name}'")
        if versions:
            return sorted(versions, key=lambda v: int(v.version), reverse=True)[0].version
    except Exception:
        pass
    return None


def create_gpu_serving_endpoint(ep_name, uc_model, model_version):
    """Create a GPU_SMALL serving endpoint for a UC-registered MLflow model."""
    body = {
        "name": ep_name,
        "config": {
            "served_entities": [{
                "entity_name":          uc_model,
                "entity_version":       model_version,
                "workload_type":        "GPU_SMALL",
                "workload_size":        "Small",
                "scale_to_zero_enabled": True,
            }]
        },
        "tags": [{"key": "project", "value": "otel-slm-agent-demo"}],
    }
    r = api("POST", "api/2.0/serving-endpoints", body)
    if r.status_code not in (200, 201):
        raise RuntimeError(
            f"Failed to create endpoint '{ep_name}' ({r.status_code}): {r.text[:500]}"
        )
    print(f"  Endpoint '{ep_name}' submitted — waiting for READY...")


print(f"Workspace  : {ws_url}")
print(f"Catalog    : {catalog}.{schema}")
print()
print(f"VS endpoint  : {vs_endpoint}")
print(f"Embedding EP : {embedding_ep}")
print(f"Reranker EP  : {reranker_ep}")
print(f"LLM EP       : {llm_ep}")
print(f"Gateway EP   : {gateway_ep}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Vector Search Endpoint

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)

print(f"Checking VS endpoint: {vs_endpoint}")
try:
    ep_info = vsc.get_endpoint(vs_endpoint)
    state   = ep_info.get("endpoint_status", {}).get("state", "UNKNOWN")
    print(f"  EXISTS — state: {state}")
    vs_online = (state == "ONLINE")
except Exception:
    print("  NOT FOUND — creating STANDARD endpoint...")
    vsc.create_endpoint(name=vs_endpoint, endpoint_type="STANDARD")
    vs_online = False

if not vs_online:
    print("  Polling for ONLINE (up to 30 min)...")
    for _ in range(60):
        time.sleep(30)
        state = vsc.get_endpoint(vs_endpoint).get("endpoint_status", {}).get("state", "")
        print(f"    state={state}")
        if state == "ONLINE":
            break
    else:
        raise TimeoutError(f"VS endpoint '{vs_endpoint}' did not reach ONLINE in 30 min")

print(f"VS endpoint ONLINE ✓")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. OTel Embedding Model — `otel-embedding2-300m`
# MAGIC
# MAGIC Deploys **`farbodtavakkoli/OTel-Embedding-335M`** (BAAI/bge-large-en-v1.5 fine-tuned
# MAGIC on 326K+ telco samples) using the MLflow `sentence_transformers` flavor.
# MAGIC
# MAGIC - **Input:** `{"input": ["text1", "text2", ...]}` — OpenAI-compatible embeddings format
# MAGIC - **Output:** `{"data": [{"embedding": [...]}, ...]}`
# MAGIC - **Dim:** 768  — matches the pre-computed vectors in `04_create_vs_indexes`

# COMMAND ----------

UC_EMBEDDING_MODEL = f"{catalog}.{schema}.otel_embedding_335m"

exists, state = get_serving_state(embedding_ep)
if exists:
    print(f"Endpoint '{embedding_ep}' already exists (ready={state}) — skipping ✓")
else:
    # ── Step 1: Register in UC if not already there ───────────────────────
    model_ver = get_latest_uc_model_version(UC_EMBEDDING_MODEL)
    if model_ver:
        print(f"UC model '{UC_EMBEDDING_MODEL}' already registered (v{model_ver})")
    else:
        print(f"Registering '{UC_EMBEDDING_MODEL}' from HuggingFace...")
        from sentence_transformers import SentenceTransformer

        embed_model = SentenceTransformer("farbodtavakkoli/OTel-Embedding-335M")

        from mlflow.models import infer_signature
        _emb_vec = embed_model.encode("example telco query").tolist()
        _emb_sig = infer_signature(
            {"input": "example telco query"},
            {"data": [{"embedding": _emb_vec, "index": 0, "object": "embedding"}],
             "model": "otel-embedding", "object": "list",
             "usage": {"prompt_tokens": 4, "total_tokens": 4}},
        )
        with mlflow.start_run(run_name="register_otel_embedding_335m"):
            mlflow.sentence_transformers.log_model(
                model=embed_model,
                task="llm/v1/embeddings",
                artifact_path="model",
                registered_model_name=UC_EMBEDDING_MODEL,
                signature=_emb_sig,
                pip_requirements=[
                    "sentence-transformers>=3.0.0",
                    "torch>=2.0.0",
                ],
            )
        model_ver = get_latest_uc_model_version(UC_EMBEDDING_MODEL)
        print(f"  Registered as v{model_ver} ✓")

    # ── Step 2: Create the serving endpoint ──────────────────────────────
    create_gpu_serving_endpoint(embedding_ep, UC_EMBEDDING_MODEL, model_ver)
    poll_serving_ready(embedding_ep)
    print(f"Embedding endpoint READY ✓")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. OTel Reranker Model — `otel-reranker-600m`
# MAGIC
# MAGIC Deploys **`farbodtavakkoli/OTel-Reranker-0.6B`** (Qwen3-0.6B cross-encoder) as an
# MAGIC MLflow PyFunc model.
# MAGIC
# MAGIC - **Input:** `{"query": "...", "passages": ["...", "...", ...]}`
# MAGIC - **Output:** `{"scores": [0.92, 0.31, ...]}` — one float per passage, higher = more relevant
# MAGIC
# MAGIC The agent's retrieval pipeline calls this after VS search to rerank top-20 → top-5.

# COMMAND ----------

UC_RERANKER_MODEL = f"{catalog}.{schema}.otel_reranker_06b"

exists, state = get_serving_state(reranker_ep)
if exists:
    print(f"Endpoint '{reranker_ep}' already exists (ready={state}) — skipping ✓")
else:
    # ── Step 1: Register in UC if not already there ───────────────────────
    model_ver = get_latest_uc_model_version(UC_RERANKER_MODEL)
    if model_ver:
        print(f"UC model '{UC_RERANKER_MODEL}' already registered (v{model_ver})")
    else:
        print(f"Registering '{UC_RERANKER_MODEL}' from HuggingFace...")

        tmpdir = tempfile.mkdtemp(prefix="otel_reranker_")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        reranker_tokenizer = AutoTokenizer.from_pretrained(
            "farbodtavakkoli/OTel-Reranker-0.6B"
        )
        reranker_hf_model = AutoModelForSequenceClassification.from_pretrained(
            "farbodtavakkoli/OTel-Reranker-0.6B"
        )
        reranker_tokenizer.save_pretrained(tmpdir)
        reranker_hf_model.save_pretrained(tmpdir)
        print(f"  Model weights saved to {tmpdir}")

        # ── PyFunc wrapper ────────────────────────────────────────────────
        class OTelRerankerModel(mlflow.pyfunc.PythonModel):
            """
            Cross-encoder reranker for OTel-Reranker-0.6B (Qwen3-0.6B based).

            Input  — dict or single-row DataFrame:
              query    : str
              passages : list[str]  or JSON-encoded list string

            Output — dict:
              scores   : list[float]  (one per passage, higher = more relevant)
            """

            def load_context(self, context):
                import torch
                from transformers import (
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                )
                path = context.artifacts["model_path"]
                self._tokenizer = AutoTokenizer.from_pretrained(path)
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    path, torch_dtype=torch.float16,
                )
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model = self._model.to(self._device).eval()

            def predict(self, context, model_input, params=None):
                import torch
                import pandas as pd

                if isinstance(model_input, pd.DataFrame):
                    query    = model_input["query"].iloc[0]
                    passages = model_input["passages"].iloc[0]
                else:
                    query    = model_input.get("query", "")
                    passages = model_input.get("passages", [])

                if isinstance(passages, str):
                    passages = json.loads(passages)

                pairs      = [(query, p) for p in passages]
                scores     = []
                batch_size = 8

                for i in range(0, len(pairs), batch_size):
                    batch  = pairs[i : i + batch_size]
                    inputs = self._tokenizer(
                        [p[0] for p in batch],
                        [p[1] for p in batch],
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    ).to(self._device)

                    with torch.no_grad():
                        logits = self._model(**inputs).logits
                        # Binary classification → use positive-class score
                        if logits.shape[-1] == 2:
                            batch_scores = logits[:, 1].tolist()
                        else:
                            batch_scores = logits.squeeze(-1).tolist()

                    if not isinstance(batch_scores, list):
                        batch_scores = [batch_scores]
                    scores.extend(batch_scores)

                return {"scores": scores}

        # ── Log + register ────────────────────────────────────────────────
        from mlflow.models import infer_signature
        import pandas as _pd
        _rnk_sig = infer_signature(
            _pd.DataFrame({"query": ["example query"], "passages": ['["passage one", "passage two"]']}),
            {"scores": [0.9, 0.3]},
        )
        with mlflow.start_run(run_name="register_otel_reranker_06b"):
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=OTelRerankerModel(),
                artifacts={"model_path": tmpdir},
                registered_model_name=UC_RERANKER_MODEL,
                signature=_rnk_sig,
                pip_requirements=[
                    "transformers>=4.40.0",
                    "torch>=2.0.0",
                    "accelerate>=0.30.0",
                ],
            )
        model_ver = get_latest_uc_model_version(UC_RERANKER_MODEL)
        print(f"  Registered as v{model_ver} ✓")

    # ── Step 2: Create the serving endpoint ──────────────────────────────
    create_gpu_serving_endpoint(reranker_ep, UC_RERANKER_MODEL, model_ver)
    poll_serving_ready(reranker_ep)
    print(f"Reranker endpoint READY ✓")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. OTel LLM — `otel-llm-1b-it`
# MAGIC
# MAGIC Deploys **`farbodtavakkoli/OTel-LLM-1.2B-IT`** (Gemma-3-1b-it fine-tuned on telco data)
# MAGIC using the MLflow `transformers` flavor with the `llm/v1/chat` task.
# MAGIC
# MAGIC - **Input:** `{"messages": [{"role": "user", "content": "..."}]}` — OpenAI chat format
# MAGIC - **Output:** `{"choices": [{"message": {"role": "assistant", "content": "..."}}]}`
# MAGIC - **Key property:** trained to **abstain** when retrieved context is insufficient,
# MAGIC   preventing hallucinated answers in the RAG sub-agent pipeline

# COMMAND ----------

UC_LLM_MODEL = f"{catalog}.{schema}.otel_llm_12b_it"

exists, state = get_serving_state(llm_ep)
if exists:
    print(f"Endpoint '{llm_ep}' already exists (ready={state}) — skipping ✓")
else:
    # ── Step 1: Register in UC if not already there ───────────────────────
    model_ver = get_latest_uc_model_version(UC_LLM_MODEL)
    if model_ver:
        print(f"UC model '{UC_LLM_MODEL}' already registered (v{model_ver})")
    else:
        print(f"Registering '{UC_LLM_MODEL}' from HuggingFace...")
        import torch
        from transformers import pipeline as hf_pipeline

        # Load on CPU for logging — inference uses GPU at serving time
        llm_pipe = hf_pipeline(
            "text-generation",
            model="farbodtavakkoli/OTel-LLM-1.2B-IT",
            torch_dtype=torch.float16,
            device_map="cpu",
        )

        with mlflow.start_run(run_name="register_otel_llm_12b_it"):
            # signature is auto-set by MLflow when task="llm/v1/chat"; do NOT pass it manually
            mlflow.transformers.log_model(
                transformers_model=llm_pipe,
                task="llm/v1/chat",
                artifact_path="model",
                registered_model_name=UC_LLM_MODEL,
                model_config={
                    "max_new_tokens": 512,
                    "temperature":    0.1,
                    "do_sample":      True,
                },
                pip_requirements=[
                    "transformers>=4.44.0",
                    "torch>=2.0.0",
                    "accelerate>=0.30.0",
                ],
            )
        model_ver = get_latest_uc_model_version(UC_LLM_MODEL)
        print(f"  Registered as v{model_ver} ✓")

    # ── Step 2: Create the serving endpoint ──────────────────────────────
    create_gpu_serving_endpoint(llm_ep, UC_LLM_MODEL, model_ver)
    poll_serving_ready(llm_ep)
    print(f"LLM endpoint READY ✓")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Supervisor External Model + AI Gateway  ⚠ PLACEHOLDER
# MAGIC
# MAGIC This section creates an **Anthropic Claude external model endpoint** behind a
# MAGIC **Databricks AI Gateway** as a drop-in replacement for `databricks-claude-sonnet-4`.
# MAGIC
# MAGIC ### When to use this
# MAGIC Use this when `databricks-claude-sonnet-4` is **not available** in the target workspace
# MAGIC (e.g., no Foundation Model API access, or a bring-your-own-key deployment). The
# MAGIC resulting endpoint speaks the same OpenAI-compatible chat API — only `LLM_ENDPOINT`
# MAGIC in `app.yaml` / `databricks.yml` needs updating, no agent code changes required.
# MAGIC
# MAGIC ### Pre-requisites
# MAGIC 1. Store your Anthropic API key in Databricks Secrets:
# MAGIC    ```bash
# MAGIC    databricks secrets create-scope <scope>   # if the scope doesn't exist yet
# MAGIC    databricks secrets put-secret <scope> anthropic_api_key --string-value sk-ant-...
# MAGIC    ```
# MAGIC 2. Set the **`secret_scope`** widget to `<scope>` and re-run this section.
# MAGIC
# MAGIC ### AI Gateway features applied
# MAGIC | Feature | Config |
# MAGIC |---------|--------|
# MAGIC | Usage tracking | Enabled — token/request counters in workspace |
# MAGIC | Inference table | Enabled — every request/response logged to Delta |
# MAGIC | Rate limits | 60 req/min per user |
# MAGIC | PII guardrails | Input PII blocked |

# COMMAND ----------

if not secret_scope.strip():
    # ── Informational output when skipped ────────────────────────────────
    print("=" * 65)
    print("SUPERVISOR GATEWAY — SKIPPED (secret_scope widget is empty)")
    print("=" * 65)
    print()
    print("If 'databricks-claude-sonnet-4' is available in this workspace,")
    print("no action needed — the agent uses it by default.")
    print()
    print("To deploy the external Claude endpoint + AI Gateway instead:")
    print()
    print("  1. Store your Anthropic API key in Databricks Secrets:")
    print(f"       databricks secrets create-scope <scope>")
    print(f"       databricks secrets put-secret <scope> {secret_key} --string-value sk-ant-...")
    print()
    print("  2. Set the 'secret_scope' widget to <scope> and re-run this cell.")
    print()
    print("  3. After deployment, update LLM_ENDPOINT in app.yaml / databricks.yml:")
    print(f'       LLM_ENDPOINT: "{gateway_ep}"')

else:
    exists, state = get_serving_state(gateway_ep)

    if exists:
        print(f"Gateway endpoint '{gateway_ep}' already exists (ready={state}) — skipping ✓")
    else:
        # ── Create the external model endpoint ───────────────────────────
        print(f"Creating external model endpoint: {gateway_ep}")
        secret_ref = "{{" + f"secrets/{secret_scope}/{secret_key}" + "}}"
        body = {
            "name": gateway_ep,
            "config": {
                "served_entities": [{
                    "external_model": {
                        "name":     claude_model_name,
                        "provider": "anthropic",
                        "task":     "llm/v1/chat",
                        "anthropic_config": {
                            "anthropic_api_key": secret_ref,
                        },
                    }
                }]
            },
            "tags": [{"key": "project", "value": "otel-slm-agent-demo"}],
        }
        r = api("POST", "api/2.0/serving-endpoints", body)
        if r.status_code not in (200, 201):
            raise RuntimeError(
                f"Failed to create gateway endpoint ({r.status_code}): {r.text[:500]}"
            )
        print("  External model endpoint created.")
        poll_serving_ready(gateway_ep)
        print(f"  External model endpoint READY ✓")

    # ── Apply AI Gateway configuration ───────────────────────────────────
    # This is idempotent — safe to re-apply even if already configured.
    print(f"Configuring AI Gateway on '{gateway_ep}'...")
    gw_body = {
        "usage_tracking_config": {
            "enabled": True,
        },
        "inference_table_config": {
            "enabled":           True,
            "catalog_name":      catalog,
            "schema_name":       schema,
            "table_name_prefix": "telco_gateway_logs",
        },
        "rate_limits": [{
            "calls":          60,
            "renewal_period": "minute",
            "key":            "user",
        }],
        "guardrails": {
            "input": {
                "pii": {"behavior": "BLOCK"},
            },
        },
    }
    r = api("PUT", f"api/2.0/serving-endpoints/{gateway_ep}/ai-gateway", gw_body)
    if r.status_code == 200:
        print(f"  AI Gateway configured ✓")
        print()
        print("  Update LLM_ENDPOINT in app.yaml and databricks.yml:")
        print(f'    LLM_ENDPOINT: "{gateway_ep}"')
    else:
        print(f"  WARNING: AI Gateway config returned {r.status_code}: {r.text[:300]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Summary

# COMMAND ----------

print("=" * 65)
print("ENDPOINT PROVISIONING COMPLETE")
print("=" * 65)
print()

rows = [
    ("VS endpoint",        vs_endpoint,  "vs"),
    ("Embedding",          embedding_ep, "serving"),
    ("Reranker",           reranker_ep,  "serving"),
    ("LLM",                llm_ep,       "serving"),
    ("Supervisor gateway", gateway_ep,   "serving"),
]

for label, ep_name, kind in rows:
    if kind == "vs":
        try:
            s    = vsc.get_endpoint(ep_name).get("endpoint_status", {}).get("state", "?")
            icon = "✓" if s == "ONLINE" else "⚠"
        except Exception:
            s, icon = "ERROR", "✗"
    else:
        _, s = get_serving_state(ep_name)
        icon = "✓" if s == "READY" else ("—" if not s else "⚠")
    print(f"  {label:<22} {ep_name:<35} {s} {icon}")

print()
print("Next steps:")
print("  Run the data setup job (tasks 01–07 automatically use these endpoints):")
print("    databricks bundle run data_setup --profile <your-profile>")
print()
print("  Or trigger the full job:")
print("    databricks jobs run-now --job-name otel-demo-data-setup --profile <your-profile>")
