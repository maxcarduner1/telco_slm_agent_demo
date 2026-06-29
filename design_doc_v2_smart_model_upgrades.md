# TelcoGPT V2 Design: Smart Model Upgrades

## Purpose

V2 adds an optimization control plane around the existing TelcoGPT solution accelerator. The goal is to keep the Databricks App stable while periodically improving prompt versions and model choices using evaluation data, MLflow traces, AI Gateway V2, and the `smart-model-upgrades` library.

This is not intended to run inside the chat app as a background thread. It should run as a separate Databricks Job or notebook workflow that evaluates candidate prompts/models and promotes winners only when they beat the current production baseline.

## Current V1 Architecture

The current app is a Databricks App with:

- React chat UI and FastAPI `ResponseAgent` backend.
- LangGraph supervisor using `LLM_ENDPOINT` directly, currently `databricks-claude-sonnet-4`.
- UC SQL function tools for structured KPI/event/churn analysis.
- Vector Search RAG over runbooks, standards, and incidents.
- OTel embedding endpoint for query vectors.
- Lakebase-backed LangGraph checkpoints and user memory.
- MLflow tracing through a workspace-portable experiment name.

V1 treats model endpoints and prompts mostly as static deployment config. Changing model choices typically means editing app env vars and redeploying.

## V2 Architecture

V2 introduces a separate optimization layer:

```text
                 Scheduled / Manual Databricks Job
                               |
                               v
                  smart-model-upgrades optimizer
                               |
          +--------------------+--------------------+
          |                                         |
          v                                         v
  MLflow Prompt Registry                    AI Gateway V2 endpoints
  prompts:/...@production                   telcogpt-supervisor
  prompts:/...@candidate                    telcogpt-query-worker
                                               |
                                               v
                                  Foundation / served model choices


Databricks App runtime:
  TelcoGPT -> Prompt Registry @production + AI Gateway prod endpoint names
```

The app continues to serve users. The optimizer runs out of band against eval sets and only changes production behavior through promotion:

- Prompt registry aliases move to the winning prompt versions.
- AI Gateway prod endpoints are patched to point at the winning model choices.
- The app keeps using the same prompt URIs and gateway endpoint names, so no app redeploy is required for most upgrades.

## Key Architecture Changes From V1

### 1. Route LLM Calls Through AI Gateway V2

Current:

- `LLM_ENDPOINT=databricks-claude-sonnet-4`
- App calls the foundation model endpoint directly.

V2:

- `LLM_ENDPOINT=telcogpt-supervisor`
- `telcogpt-supervisor` is an AI Gateway V2 endpoint.
- Candidate models might include `databricks-claude-sonnet-4`, smaller Claude variants, GPT variants, or other workspace-approved models.

This is required because Smart Model Upgrades hot-swaps model destinations behind gateway endpoints.

### 2. Move Prompts to MLflow Prompt Registry

Current:

- Main system prompt lives in `agent_app/prompts.py`.

V2:

- Register prompts in MLflow Prompt Registry, for example:
  - `prompts:/<catalog>.<schema>.telcogpt_supervisor@production`
  - `prompts:/<catalog>.<schema>.telcogpt_rag_synthesis@production`
  - `prompts:/<catalog>.<schema>.telcogpt_sql_tool_policy@production`
- App loads prompt versions at startup or per request and formats them through `PromptVersion.format(...)`.

Prompt registry lets the optimizer generate candidate prompt versions and promote winners without editing app source.

### 3. Add Eval Datasets and Scorers

Current:

- Smoke tests validate basic app behavior.
- MLflow traces exist, but no formal recurring eval loop is wired in.

V2:

- Add curated evaluation datasets for:
  - Network health summary.
  - Latency and dropped call investigation.
  - Empty data windows.
  - Runbook remediation guidance.
  - Memory recall.
  - Tool argument correctness.
- Add scorers for:
  - Correctness / expected answer.
  - Groundedness against tool outputs.
  - No silent widening of requested time windows.
  - No unmentioned truncation.
  - Tool-call validity.
  - Latency and estimated cost.

### 4. Add a Model/Prompt Optimization Job

Add a Databricks Job, initially manual, later scheduled:

1. Load train and validation examples.
2. Load prompt URIs and gateway endpoint candidate model lists.
3. Run `smu.optimize_prompts_and_models(...)`.
4. Compare winner to baseline.
5. If threshold passes, optionally call `smu.promote_to_prod(result)`.
6. Write an MLflow run summary with baseline score, candidate score, selected prompts, selected models, cost, and latency.

### 5. Add Promotion Guardrails

Do not auto-promote solely on a tiny score improvement. Require:

- Minimum quality improvement threshold.
- No regression on empty-data behavior.
- No regression on tool-call validity.
- Maximum latency gate.
- Maximum cost gate.
- Optional human approval for demo or customer-facing workspaces.

## Proposed Repository Changes

### New Files

```text
v2/
  README.md
  eval_sets/
    telcogpt_core_eval.yaml
    telcogpt_empty_data_eval.yaml
    telcogpt_tool_validity_eval.yaml
  prompts/
    supervisor.yaml
    sql_tool_policy.yaml
    rag_synthesis.yaml
  notebooks/
    00_setup_prompt_registry.py
    01_setup_gateway_endpoints.py
    02_run_smart_model_upgrade.py
    03_review_and_promote.py
  jobs/
    smart_model_upgrade_job.yml
```

### Existing Files to Modify

- `agent_app/prompts.py`
  - Keep local defaults, but allow loading registered prompts when prompt URIs are configured.

- `agent_app/agent.py`
  - Construct `ChatDatabricks(endpoint=<gateway endpoint name>)`.
  - Ensure MLflow autologging captures latency/tokens.

- `databricks.yml`
  - Add env vars:
    - `SUPERVISOR_PROMPT_URI`
    - `SQL_POLICY_PROMPT_URI`
    - `RAG_SYNTHESIS_PROMPT_URI`
    - `LLM_ENDPOINT=telcogpt-supervisor`
  - Add optional job resource for Smart Model Upgrades.

- `README.md`
  - Add V2 instructions for setup, evals, optimizer job, and promotion workflow.

## Smart Model Upgrades Integration Contract

The app must satisfy the `smart-model-upgrades` BYOA contract:

1. Expose or wrap a `predict(inputs: dict)` function for eval.
2. Load prompts through MLflow Prompt Registry and format with `PromptVersion.format(...)`.
3. Route model calls through AI Gateway endpoint names.
4. Enable MLflow tracing/autologging so latency and token usage are available.

For TelcoGPT, `predict` can be a thin wrapper around the local agent invocation path, using the same request shape as `/invocations`.

## Release Plan

### V2.0: Manual Optimizer Loop

Scope:

- Add prompt registry setup notebook.
- Add AI Gateway setup notebook for `telcogpt-supervisor`.
- Add small eval set with 20-30 rows.
- Add manual optimizer notebook.
- No scheduled promotion.

Exit criteria:

- Optimizer can run against TelcoGPT locally or from a Databricks notebook.
- Results are logged to MLflow.
- Manual promotion works in a dev workspace.

### V2.1: Promotion Guardrails

Scope:

- Add quality, latency, and cost gates.
- Add regression tests for:
  - Empty data periods.
  - Boolean tool arguments.
  - Truncation warnings.
  - Memory recall.
- Add promotion report artifact.

Exit criteria:

- Optimizer refuses promotion when guardrails fail.
- Report explains why a candidate won or lost.

### V2.2: Scheduled Optimization Job

Scope:

- Add Databricks Job for weekly or on-demand optimizer runs.
- Add job parameters for candidate model lists and promotion mode.
- Add Slack/email notification placeholder.

Exit criteria:

- Scheduled job can run without app downtime.
- Default mode is "evaluate only".
- Promotion requires explicit parameter or approval.

### V2.3: Multi-Component Optimization

Scope:

- Split prompts by component:
  - Supervisor planning.
  - SQL tool policy.
  - RAG synthesis.
  - Empty-data response policy.
- Add multiple AI Gateway endpoints if needed:
  - `telcogpt-supervisor`
  - `telcogpt-sql-reasoner`
  - `telcogpt-rag-synthesizer`

Exit criteria:

- Smart Model Upgrades can independently choose models/prompts per component.
- App still uses stable endpoint names.

### V2.4: Production Hardening

Scope:

- Add rollback to previous prompt aliases and gateway destinations.
- Add audit table for optimizer decisions.
- Add approval workflow for customer-facing demos.
- Add cost budget limits per optimizer run.

Exit criteria:

- Every promotion has a reversible audit record.
- Production behavior can roll back without redeploy.

## Operational Guidance

Start with manual optimization runs. Scheduled auto-promotion should wait until:

- Eval coverage is broad enough to represent demo-critical behavior.
- Scorers catch the failures we have already seen in V1.
- Cost and latency estimates are stable.
- Rollback is proven.

The first scheduled job should be "evaluate only." Promotion should initially remain manual.

## Open Questions

- Which model candidates are approved for the supervisor endpoint in each workspace?
- Should OTel RAG generation remain fixed, or should it also become a gateway-routed component?
- Should prompt aliases live in the demo UC schema or a shared evaluation schema?
- Should optimizer results promote directly to `@production` or first to `@staging`?
- What minimum score delta is required to justify a model/prompt change?

## Success Criteria

- TelcoGPT app can use gateway endpoints without code changes for model swaps.
- Prompts can be updated via MLflow Prompt Registry aliases.
- Optimizer can produce a measurable improvement over baseline on a held-out eval set.
- Promotion does not require app redeploy.
- Empty-data, tool-validity, memory, and truncation behavior do not regress.
