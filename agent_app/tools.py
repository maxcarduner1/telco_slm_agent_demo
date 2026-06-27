"""Tool definitions for the Telco Network Analytics Agent.

UC Function tools are loaded via UCFunctionToolkit.
Vector Search RAG tools are defined here for document retrieval.
"""

import logging
import os
import re
import requests
from typing import Optional

from langchain_core.tools import tool
from databricks_langchain import UCFunctionToolkit
from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)

# Configuration — all required, no hardcoded defaults
CATALOG            = os.environ["UC_CATALOG"]
SCHEMA             = os.environ["UC_SCHEMA"]
VS_ENDPOINT        = os.environ.get("VS_ENDPOINT", "demo_telco_vs_endpoint")
EMBEDDING_ENDPOINT = os.environ.get("EMBEDDING_ENDPOINT", "otel-embedding2-300m")
EMBEDDING_DIM = 768


def get_uc_function_tools():
    """Load UC functions as LangGraph tools via UCFunctionToolkit."""
    toolkit = UCFunctionToolkit(
        warehouse_id=os.environ["DATABRICKS_WAREHOUSE_ID"],
        function_names=[
            f"{CATALOG}.{SCHEMA}.get_kpi_metrics",
            f"{CATALOG}.{SCHEMA}.get_threshold_breaches",
            f"{CATALOG}.{SCHEMA}.compare_regions",
            f"{CATALOG}.{SCHEMA}.get_network_events",
            f"{CATALOG}.{SCHEMA}.get_churn_risk",
        ],
    )
    return toolkit.tools


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
