"""Tool definitions for the Telco Network Analytics Agent.

UC Function tools are loaded via UCFunctionToolkit.
Vector Search RAG tools are defined here for document retrieval.
"""

import logging
import os
import re
import requests
import json
import csv
import io
import time
from typing import Any, Optional

from langchain_core.tools import tool
from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)

# Configuration — all required, no hardcoded defaults
CATALOG            = os.environ["UC_CATALOG"]
SCHEMA             = os.environ["UC_SCHEMA"]
WAREHOUSE_ID       = os.environ["DATABRICKS_WAREHOUSE_ID"]
VS_ENDPOINT        = os.environ.get("VS_ENDPOINT", "demo_telco_vs_endpoint")
EMBEDDING_ENDPOINT = os.environ.get("EMBEDDING_ENDPOINT", "otel-embedding2-300m")
EMBEDDING_DIM = 768
MAX_UC_RESULT_ROWS = int(os.environ.get("MAX_UC_RESULT_ROWS", "60"))
MAX_UC_RESULT_CHARS = int(os.environ.get("MAX_UC_RESULT_CHARS", "12000"))


def _coerce_bool(value: Any) -> bool:
    """Normalize model-generated bool-ish values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "1", "yes"}:
            return True
        if normalized in {"false", "f", "0", "no"}:
            return False
    return bool(value)


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def _csv_result(columns: list[str], rows: list[list[Any]], truncated: bool) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows[:MAX_UC_RESULT_ROWS])
    return _compact_tool_output(
        json.dumps(
            {
                "format": "CSV",
                "value": buffer.getvalue(),
                "truncated": truncated,
            }
        )
    )


def _compact_csv_payload(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("value")
    if payload.get("format") != "CSV" or not isinstance(value, str):
        return payload

    rows = value.splitlines()
    if len(rows) <= MAX_UC_RESULT_ROWS + 1:
        return payload

    compacted = dict(payload)
    kept_rows = rows[: MAX_UC_RESULT_ROWS + 1]
    compacted["value"] = "\n".join(kept_rows) + "\n"
    compacted["truncated"] = True
    compacted["truncated_note"] = (
        f"Returned first {MAX_UC_RESULT_ROWS} data rows out of {len(rows) - 1}. "
        "Ask for a narrower filter or an aggregate view for more detail."
    )
    return compacted


def _compact_tool_output(output: Any) -> str:
    """Bound tool output size before it is added to model history."""
    text = output if isinstance(output, str) else json.dumps(output)

    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None

    if isinstance(parsed, dict):
        parsed = _compact_csv_payload(parsed)
        text = json.dumps(parsed)

    if len(text) <= MAX_UC_RESULT_CHARS:
        return text

    return (
        text[:MAX_UC_RESULT_CHARS]
        + f"\n\n[truncated to {MAX_UC_RESULT_CHARS} characters; ask for a narrower filter or aggregate.]"
    )


def _run_sql(statement: str) -> dict[str, Any]:
    wc = WorkspaceClient()
    response = wc.api_client.do(
        "POST",
        "/api/2.0/sql/statements",
        body={
            "statement": statement,
            "warehouse_id": WAREHOUSE_ID,
            "wait_timeout": "30s",
            "format": "JSON_ARRAY",
        },
    )
    statement_id = response.get("statement_id")
    for _ in range(30):
        status = response.get("status", {})
        state = status.get("state")
        if state == "SUCCEEDED":
            return response
        if state in {"FAILED", "CANCELED", "CLOSED"}:
            error = status.get("error", {})
            raise RuntimeError(error.get("message") or f"SQL statement {state}")
        if not statement_id:
            raise RuntimeError(f"SQL statement did not return an id: {response}")
        time.sleep(1)
        response = wc.api_client.do(
            "GET",
            f"/api/2.0/sql/statements/{statement_id}",
        )
    raise TimeoutError(f"SQL statement did not finish within 60s: {statement_id}")


def _call_uc_function(function_name: str, args: list[Any]) -> str:
    fqn = f"`{CATALOG}`.`{SCHEMA}`.`{function_name}`"
    args_sql = ", ".join(_sql_literal(arg) for arg in args)
    limit = MAX_UC_RESULT_ROWS + 1
    statement = f"SELECT * FROM {fqn}({args_sql}) LIMIT {limit}"
    response = _run_sql(statement)
    manifest = response.get("manifest", {})
    schema = manifest.get("schema", {})
    columns = [col.get("name", f"col_{i}") for i, col in enumerate(schema.get("columns", []))]
    rows = response.get("result", {}).get("data_array", [])
    truncated = len(rows) > MAX_UC_RESULT_ROWS
    return _csv_result(columns, rows, truncated)


def get_uc_function_tools():
    """Return UC SQL function tools with app-side arg normalization."""
    return [
        get_kpi_metrics,
        get_threshold_breaches,
        compare_regions,
        get_network_events,
        get_churn_risk,
    ]


@tool
def get_kpi_metrics(
    metric_name: str,
    region: Optional[str] = None,
    site_id: Optional[str] = None,
    hours_back: int = 168,
) -> str:
    """Query raw KPI rows for a metric, optionally filtered by region/site.

    Prefer aggregate tools such as compare_regions or get_threshold_breaches for
    broad questions. Use this when the user asks for row-level detail or after
    narrowing to a specific region, site, metric, and time window.
    """
    return _call_uc_function("get_kpi_metrics", [metric_name, region, site_id, hours_back])


@tool
def get_threshold_breaches(
    metric_name: str,
    threshold: float,
    direction: str = "above",
    region: Optional[str] = None,
    hours_back: int = 168,
) -> str:
    """Find KPI readings above or below a threshold.

    Prefer this over raw KPI rows when investigating whether there are issues,
    violations, or problematic sites.
    """
    return _call_uc_function(
        "get_threshold_breaches",
        [metric_name, threshold, direction, region, hours_back],
    )


@tool
def compare_regions(
    metric_name: str,
    hours_back: int = 168,
    agg: str = "avg",
) -> str:
    """Compare a KPI metric across all regions with an aggregation.

    Prefer this for broad regional or fleet-wide questions before requesting
    raw KPI rows.
    """
    return _call_uc_function("compare_regions", [metric_name, hours_back, agg])


@tool
def get_network_events(
    region: Optional[str] = None,
    severity: Optional[str] = None,
    event_type: Optional[str] = None,
    hours_back: int = 168,
    unresolved_only: Any = False,
) -> str:
    """Query network events by region, severity, event type, and resolution state."""
    return _call_uc_function(
        "get_network_events",
        [
            region,
            severity,
            event_type,
            hours_back,
            _coerce_bool(unresolved_only),
        ],
    )


@tool
def get_churn_risk(
    region: Optional[str] = None,
    segment: Optional[str] = None,
    min_churn_rate: float = 0.0,
    days_back: int = 30,
) -> str:
    """Query customer churn risk by region and segment."""
    return _call_uc_function(
        "get_churn_risk",
        [region, segment, min_churn_rate, days_back],
    )


def _embed_query(text: str) -> list[float]:
    """Embed a query using the OTel embedding endpoint."""
    wc = WorkspaceClient()
    host = wc.config.host
    auth_header = wc.config.authenticate().get("Authorization", "")

    url = f"{host}/serving-endpoints/{EMBEDDING_ENDPOINT}/invocations"
    headers = {"Authorization": auth_header, "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json={"input": [text]}, timeout=30)
    resp.raise_for_status()

    result = resp.json()
    if isinstance(result, list):
        return result[0]
    elif isinstance(result, dict) and "data" in result:
        return result["data"][0]["embedding"]
    else:
        raise ValueError(f"Unexpected embedding response format: {str(result)[:200]}")


def _vs_search(index_name: str, query: str, num_results: int = 5) -> list[dict]:
    """Search a Vector Search index with a pre-computed query embedding."""
    from databricks.vector_search.client import VectorSearchClient

    wc = WorkspaceClient()
    auth_header = wc.config.authenticate().get("Authorization", "")
    token = auth_header.removeprefix("Bearer ")
    vsc = VectorSearchClient(
        workspace_url=wc.config.host,
        personal_access_token=token,
        disable_notice=True,
    )
    index = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=index_name)

    query_vector = _embed_query(query)
    try:
        results = index.similarity_search(
            query_vector=query_vector,
            num_results=num_results,
            columns=["chunk_id", "chunk_text", "source_path", "doc_type"],
        )
    except Exception as e:
        # Some workspaces return 1024-d vectors from the embedding endpoint while
        # indexes were created with 768-d vectors. Retry with adjusted dimensions.
        msg = str(e)
        match = re.search(
            r"query vector dimension (\d+) does not match index vector dimension (\d+)",
            msg,
        )
        if not match:
            raise

        index_dim = int(match.group(2))
        if len(query_vector) > index_dim:
            adjusted_vector = query_vector[:index_dim]
        elif len(query_vector) < index_dim:
            adjusted_vector = query_vector + [0.0] * (index_dim - len(query_vector))
        else:
            adjusted_vector = query_vector

        logger.warning(
            "Adjusted embedding vector size %s -> %s for index %s",
            len(query_vector),
            index_dim,
            index_name,
        )
        results = index.similarity_search(
            query_vector=adjusted_vector,
            num_results=num_results,
            columns=["chunk_id", "chunk_text", "source_path", "doc_type"],
        )

    docs = []
    for row in results.get("result", {}).get("data_array", []):
        docs.append({
            "chunk_id": row[0],
            "text": row[1],
            "doc_title": row[2],
            "source_type": row[3],
        })
    return docs


def _safe_vs_search(index_name: str, query: str, num_results: int = 5) -> tuple[list[dict], Optional[str]]:
    """Run VS search without raising, so the agent can always complete a turn."""
    try:
        return _vs_search(index_name, query, num_results=num_results), None
    except Exception as e:
        msg = str(e)
        logger.exception("RAG retrieval failed for index %s", index_name)
        return [], msg


def _runbook_fallback(query: str, error_msg: str) -> str:
    """Fallback guidance when VS/embedding retrieval is unavailable."""
    return (
        "I couldn't access the runbook retrieval backend right now, so I'll give a safe "
        "manual troubleshooting flow for VoLTE quality issues.\n\n"
        "1. Confirm symptom scope: affected regions/sites, time window, device segment, and "
        "whether the issue is low MOS, drops, or one-way audio.\n"
        "2. Check radio quality indicators around impacted cells (coverage, interference, "
        "handover behavior, and congestion signals).\n"
        "3. Validate core path health for IMS/voice signaling and media path latency/loss.\n"
        "4. Compare current metrics against recent baseline to isolate a sudden regression.\n"
        "5. Correlate with active outages/degradations/maintenance in the same window.\n"
        "6. Prioritize remediation by customer impact and re-test MOS after each change.\n\n"
        f"Retrieval error: {error_msg[:240]}"
    )


@tool
def search_runbooks(query: str) -> str:
    """Search operational runbooks for troubleshooting procedures, remediation steps, and best practices.

    Use this when the user asks how to fix an issue, what the procedure is for something,
    or needs operational guidance.

    Args:
        query: Natural language description of what you're looking for.
    """
    docs, error = _safe_vs_search(f"{CATALOG}.{SCHEMA}.otel_runbooks_vs_index", query)
    if error:
        return _runbook_fallback(query, error)
    if not docs:
        return "No relevant runbook content found."

    results = []
    for d in docs:
        results.append(f"**{d['doc_title']}**\n{d['text']}")
    return "\n\n---\n\n".join(results)


@tool
def search_standards(query: str) -> str:
    """Search network standards and specifications for compliance requirements, thresholds, and technical specifications.

    Use this when the user asks about standards, compliance, acceptable thresholds,
    or technical specifications.

    Args:
        query: Natural language description of what you're looking for.
    """
    docs, error = _safe_vs_search(f"{CATALOG}.{SCHEMA}.otel_standards_vs_index", query)
    if error:
        return (
            "I couldn't access standards retrieval right now due to a backend error, so I can't "
            "quote the indexed standards document for this query.\n\n"
            f"Retrieval error: {error[:240]}"
        )
    if not docs:
        return "No relevant standards content found."

    results = []
    for d in docs:
        results.append(f"**{d['doc_title']}**\n{d['text']}")
    return "\n\n---\n\n".join(results)


@tool
def search_incidents(query: str) -> str:
    """Search historical incident reports for past outages, root cause analyses, and resolution patterns.

    Use this when the user asks about past incidents, wants to know if something
    has happened before, or needs historical context.

    Args:
        query: Natural language description of what you're looking for.
    """
    docs, error = _safe_vs_search(f"{CATALOG}.{SCHEMA}.otel_incidents_vs_index", query)
    if error:
        return (
            "I couldn't access incident retrieval right now due to a backend error, so I can't "
            "search similar historical incidents at the moment.\n\n"
            f"Retrieval error: {error[:240]}"
        )
    if not docs:
        return "No relevant incident reports found."

    results = []
    for d in docs:
        results.append(f"**{d['doc_title']}**\n{d['text']}")
    return "\n\n---\n\n".join(results)


def get_rag_tools():
    """Return the RAG document search tools."""
    return [search_runbooks, search_standards, search_incidents]
