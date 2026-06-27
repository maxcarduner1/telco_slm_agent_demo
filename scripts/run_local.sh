#!/usr/bin/env bash
set -euo pipefail

# One-command local runner for fast troubleshooting loops.
# Override any value by exporting it before running this script.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export DATABRICKS_AUTH_STORAGE="${DATABRICKS_AUTH_STORAGE:-plaintext}"
export DATABRICKS_CONFIG_PROFILE="${DATABRICKS_CONFIG_PROFILE:-fevm-y353qx}"
export DATABRICKS_HOST="${DATABRICKS_HOST:-https://fevm-serverless-stable-y353qx.cloud.databricks.com}"

export UC_CATALOG="${UC_CATALOG:-serverless_stable_y353qx_catalog}"
export UC_SCHEMA="${UC_SCHEMA:-otel_rag_agent_demo}"
export DATABRICKS_WAREHOUSE_ID="${DATABRICKS_WAREHOUSE_ID:-c17949ecf3c38481}"

export LLM_ENDPOINT="${LLM_ENDPOINT:-databricks-claude-sonnet-4}"
export EMBEDDING_ENDPOINT="${EMBEDDING_ENDPOINT:-otel-embedding2-300m}"
export VS_ENDPOINT="${VS_ENDPOINT:-demo_telco_vs_endpoint}"

export CHAT_APP_PORT="${CHAT_APP_PORT:-3000}"
export BACKEND_PORT="${BACKEND_PORT:-8000}"
export API_PROXY="http://localhost:${BACKEND_PORT}/invocations"
export LOCAL_BACKEND_ONLY="${LOCAL_BACKEND_ONLY:-0}"

echo "== Local app config =="
echo "PROFILE:   ${DATABRICKS_CONFIG_PROFILE}"
echo "HOST:      ${DATABRICKS_HOST}"
echo "CATALOG:   ${UC_CATALOG}.${UC_SCHEMA}"
echo "WAREHOUSE: ${DATABRICKS_WAREHOUSE_ID}"
echo "FRONTEND:  http://localhost:${CHAT_APP_PORT}"
echo "BACKEND:   http://localhost:${BACKEND_PORT}"
echo "MODE:      $([ "${LOCAL_BACKEND_ONLY}" = "1" ] && echo "backend-only" || echo "full-stack")"
echo

if ! command -v databricks >/dev/null 2>&1; then
  echo "Databricks CLI not found. Install CLI first." >&2
  exit 1
fi

if ! databricks auth token --profile "${DATABRICKS_CONFIG_PROFILE}" >/dev/null 2>&1; then
  echo "Databricks auth token check failed for profile '${DATABRICKS_CONFIG_PROFILE}'." >&2
  echo "Run: databricks auth login --host ${DATABRICKS_HOST} --profile ${DATABRICKS_CONFIG_PROFILE}" >&2
  exit 1
fi

if [ "${LOCAL_BACKEND_ONLY}" = "1" ]; then
  exec uvicorn agent_app.server:app --host 0.0.0.0 --port "${BACKEND_PORT}" --reload
fi

python scripts/start_app.py
