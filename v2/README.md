# TelcoGPT V2.0: Smart Model Upgrades

V2.0 adds an evaluate-only Smart Model Upgrades workflow around TelcoGPT. It is
isolated from the live V1 app and uses `telcogpt-v2-*` resource names.

## Scope

- Optimize the supervisor prompt and supervisor model choice only.
- Use AI Gateway V2 endpoint `telcogpt-v2-supervisor`.
- Register prompt `telcogpt_v2_supervisor` in MLflow Prompt Registry.
- Run evaluate-only optimization. Do not call `promote_to_prod` in V2.0.
- Keep V1 app, serving endpoints, Vector Search endpoint, and Lakebase project
  untouched.

## Local Checks

Before running Databricks notebooks:

```bash
python -m compileall agent_app v2
```

Set env vars for local predict testing:

```bash
export DATABRICKS_CONFIG_PROFILE=DEFAULT
export DATABRICKS_HOST=https://fevm-cmegdemos.cloud.databricks.com
export UC_CATALOG=cmegdemos_catalog
export UC_SCHEMA=network_analytics_enablement
export DATABRICKS_WAREHOUSE_ID=7b65956f30d66feb
export LLM_ENDPOINT=databricks-claude-sonnet-4
```

Then run:

```bash
python - <<'PY'
from v2.predict import predict
print(predict({"question": "How does latency look across all regions over the last 12 hours?"}))
PY
```

## Databricks Notebook Flow

Run in order:

1. `v2/notebooks/00_setup_prompt_registry.py`
2. `v2/notebooks/01_setup_gateway_endpoints.py`
3. `v2/notebooks/02_run_smart_model_upgrade.py`

The optimizer notebook logs results to MLflow and intentionally does not promote.

## Resource Naming

Use these defaults unless explicitly changing the namespace:

| Resource | Name |
| --- | --- |
| Gateway endpoint | `telcogpt-v2-supervisor` |
| Prompt name | `<catalog>.<schema>.telcogpt_v2_supervisor` |
| MLflow experiment | `/Shared/telcogpt-v2-smart-model-upgrades` |

## Promotion Policy

V2.0 is evaluate-only. Promotion is deferred to V2.1 after guardrails and
rollback are implemented.
