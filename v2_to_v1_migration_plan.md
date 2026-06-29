# Plan: Migrate Selected V2 Changes Back Into V1

## Goal

Safely move shared hardening improvements from the V2 branch into the V1 app
without introducing V2-only resources, prompt registry dependencies, or AI
Gateway routing into the current production-style V1 path.

## Keep V2-Only Changes Out of V1

Do not migrate these into V1 yet:

- `telco_agent_v2` app resource in `databricks.yml`
- `app_v2/`
- `v2/`
- `SUPERVISOR_PROMPT_URI`
- `LLM_ENDPOINT=telcogpt-v2-supervisor`
- `telcogpt-v2-*` AI Gateway resources
- V2 optimizer notebooks/jobs

These remain isolated on the V2 branch until promotion and rollback guardrails
are designed.

## Candidate Changes to Backport to V1

### 1. UC Tool Robustness

Backport from `agent_app/tools.py`:

- Custom UC SQL function tools via SQL Statement API.
- Boolean coercion for arguments like `unresolved_only`.
- Row and character caps for large tool outputs.
- Truncation metadata in tool outputs.
- Aggregate-first tool descriptions.

Why:

- Fixes the observed `"FALSE"` string boolean failure.
- Reduces context-window pressure from large raw CSV tool outputs.
- Improves broad regional questions by steering to aggregate tools.

Validation:

- Local backend smoke:
  - broad latency question uses `compare_regions`
  - unresolved events query does not crash
  - empty one-hour dropped-call query returns no-data answer
- Databricks App smoke against V1 app.

### 2. Model History Compaction

Backport from `agent_app/agent.py`:

- LangGraph `pre_model_hook` that compacts model input history.
- Keep recent turns verbatim.
- Summarize older user/assistant messages.
- Preserve full Lakebase checkpoint state.

Why:

- Reduces risk of long conversations exceeding model context.
- Avoids sending large tool-heavy histories back to the model.

Validation:

- Same-thread memory recall still works.
- Long multi-turn conversation does not duplicate or corrupt tool history.
- No regression in tool-calling behavior.

### 3. Prompt Guardrails

Backport from `agent_app/prompts.py`:

- Boolean arguments must be real JSON booleans.
- Prefer aggregate/targeted tools before raw row dumps.
- Warn when tool outputs are truncated.
- Respect requested time windows.

Why:

- Reinforces the runtime guardrails at the model behavior layer.

Validation:

- Confirm V1 app no longer silently widens time windows.
- Confirm answers mention truncation when present.

### 4. Separate UC Permission Notebook

Backport:

- `notebooks/08_grant_app_uc_permissions.py`
- `databricks.yml` task dependency changes that run UC grants after:
  - UC functions exist
  - VS indexes exist
  - app SP exists

Why:

- Keeps `04_create_vs_indexes.py` focused on index creation.
- Avoids granting permissions before objects exist.
- Provides a reusable one-off permissions repair notebook.

Validation:

- Run `08` for V1 app SP in a test workspace.
- Confirm UC functions and VS indexes can be queried by V1 app.

### 5. Lakebase `memory_schema` Parameter

Backport carefully:

- `memory_schema` widget in `07_provision_lakebase_app.py`
- Keep default as `agent_memory` for V1.
- Ensure any pgvector setup is validated before relying on it.

Why:

- Allows future isolated schemas without creating a separate Lakebase project.

Risk:

- Current V2 testing showed `vector` extension visibility is still not fully
  resolved. Do not make V1 depend on this until verified.

Validation:

- Rerun `07` with default `agent_memory` in a test workspace.
- Confirm V1 Lakebase checkpointing still initializes cleanly.

## Proposed Backport Sequence

1. Create a V1 backport branch from `main`.
2. Cherry-pick or manually copy only the shared files:
   - `agent_app/tools.py`
   - `agent_app/agent.py`
   - `agent_app/prompts.py`
   - `notebooks/08_grant_app_uc_permissions.py`
   - selected `databricks.yml` task dependency updates
   - selected `notebooks/07_provision_lakebase_app.py` default-safe changes
3. Exclude all V2-only files/resources.
4. Run local syntax/lint checks.
5. Deploy to a non-CMEG FEVM workspace first.
6. Run full smoke:
   - KPI aggregate
   - empty data period
   - unresolved events boolean path
   - RAG runbook
   - Lakebase memory recall
7. If clean, deploy to CMEG V1 app.

## Do Not Promote Yet

Do not route V1 through `telcogpt-v2-supervisor` or Prompt Registry until:

- AI Gateway app SP permission behavior is fully understood.
- Prompt Registry app SP read behavior is fixed.
- V2 Lakebase schema/vector extension issue is resolved.
- Promotion/rollback plan is implemented.
