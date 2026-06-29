"""Thin wrapper over the AI Gateway V2 REST API.

Provides idempotent CRUD helpers (`create_endpoint`, `get_endpoint`,
`update_endpoint`, `delete_endpoint`, `list_endpoints`) plus shape constructors
(`destination`, `tag`, `traffic_config`, etc.) that match the Gateway V2
request schema. All calls hit `<DATABRICKS_HOST>/api/ai-gateway/v2/endpoints`.
Authenticates via `DATABRICKS_TOKEN` from the environment.
"""
import os
import requests
from typing import List, Optional

BASE_PATH = "/api/ai-gateway/v2/endpoints"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_headers(token: Optional[str] = None) -> dict:
    token = token or os.environ["DATABRICKS_TOKEN"]
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _get_base_url(host: Optional[str] = None) -> str:
    host = host or os.environ["DATABRICKS_HOST"]
    return host.rstrip("/") + BASE_PATH


def _raise_for_status(response: requests.Response) -> None:
    if not response.ok:
        raise requests.HTTPError(
            f"{response.status_code} {response.reason}: {response.text}",
            response=response,
        )


# ---------------------------------------------------------------------------
# Object builders — construct the dicts the API expects
# ---------------------------------------------------------------------------

def destination(name: str, destination_type: str, traffic_percentage: Optional[int] = None) -> dict:
    """Build a DestinationConfig object.

    Args:
        name: Model name. PPT/PT names must start with 'system.ai.'.
        destination_type: 'PAY_PER_TOKEN_FOUNDATION_MODEL' or 'EXTERNAL_FOUNDATION_MODEL'.
        traffic_percentage: 0–100. All destinations in a config must sum to 100.
            Omit for fallback destinations.
    """
    d = {"name": name, "type": destination_type}
    if traffic_percentage is not None:
        d["traffic_percentage"] = traffic_percentage
    return d


def rate_limit(
    key: str,
    requests: Optional[int] = None,
    tokens: Optional[int] = None,
    principal: Optional[str] = None,
    renewal_period: str = "MINUTE",
) -> dict:
    """Build a RateLimit object.

    Args:
        key: 'USER', 'USER_GROUP', 'SERVICE_PRINCIPAL', 'ENDPOINT', or 'USER_DEFAULT'.
        requests: Max requests per renewal period. Exactly one of requests/tokens required.
        tokens: Max tokens per renewal period. Exactly one of requests/tokens required.
        principal: Required for USER, USER_GROUP, SERVICE_PRINCIPAL keys.
        renewal_period: Only 'MINUTE' is supported.
    """
    if requests is None and tokens is None:
        raise ValueError("Exactly one of 'requests' or 'tokens' must be provided.")
    if requests is not None and tokens is not None:
        raise ValueError("Exactly one of 'requests' or 'tokens' must be provided, not both.")
    rl = {"key": key, "renewal_period": renewal_period}
    if requests is not None:
        rl["requests"] = requests
    if tokens is not None:
        rl["tokens"] = tokens
    if principal is not None:
        rl["principal"] = principal
    return rl


def fallback_config(
    destinations: List[dict],
    strategy: str = "ROUND_ROBIN",
    max_attempts: int = 2,
) -> dict:
    """Build a FallbackConfig object.

    Args:
        destinations: List of destination() objects. Must not overlap with primary destinations.
            Do not set traffic_percentage on these.
        strategy: Only 'ROUND_ROBIN' is supported.
        max_attempts: 1 or 2.
    """
    return {"strategy": strategy, "max_attempts": max_attempts, "destinations": destinations}


def tag(key: str, value: str) -> dict:
    """Build an EndpointTag object."""
    return {"key": key, "value": value}


def inference_table(
    catalog_name: str,
    schema_name: str,
    table_name_prefix: str,
    enabled: bool = True,
) -> dict:
    """Build an InferenceTableConfig object."""
    return {
        "catalog_name": catalog_name,
        "schema_name": schema_name,
        "table_name_prefix": table_name_prefix,
        "enabled": enabled,
    }


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------

def create_endpoint(
    name: str,
    destinations: List[dict],
    routing_strategy: str = "REQUEST_BASED_TRAFFIC_SPLIT",
    task_type: Optional[str] = None,
    fallback: Optional[dict] = None,
    rate_limits: Optional[List[dict]] = None,
    tags: Optional[List[dict]] = None,
    usage_tracking_enabled: Optional[bool] = None,
    inference_table_config: Optional[dict] = None,
    host: Optional[str] = None,
    token: Optional[str] = None,
) -> dict:
    """Create an AI Gateway V2 endpoint.

    Args:
        name: Endpoint name (3–254 chars, pattern ^[a-z0-9_-]+$, cannot start with 'databricks-').
        destinations: List of destination() objects. Traffic percentages must sum to 100.
        routing_strategy: Only 'REQUEST_BASED_TRAFFIC_SPLIT' is supported.
        task_type: Optional task type. Immutable after create.
            One of: 'llm/v1/chat', 'llm/v1/completions', 'llm/v1/embeddings',
            'llm/v1/responses', 'transparent'.
        fallback: Optional fallback_config() object.
        rate_limits: Optional list of rate_limit() objects (max 20).
        tags: Optional list of tag() objects (max 20).
        usage_tracking_enabled: Enable usage tracking if True.
        inference_table_config: Optional inference_table() object.
        host: Databricks workspace URL. Defaults to DATABRICKS_HOST env var.
        token: Databricks PAT. Defaults to DATABRICKS_TOKEN env var.

    Returns:
        Operation dict with the created Endpoint in 'response'.
    """
    config = {"destinations": destinations, "routing_strategy": routing_strategy}
    if fallback is not None:
        config["fallback"] = fallback
    if rate_limits is not None:
        config["rate_limits"] = rate_limits
    if tags is not None:
        config["tags"] = tags
    if usage_tracking_enabled is not None:
        config["usage_tracking"] = {"enabled": usage_tracking_enabled}
    if inference_table_config is not None:
        config["inference_table"] = inference_table_config

    body = {"name": name, "config": config}
    if task_type is not None:
        body["task_type"] = task_type

    response = requests.post(_get_base_url(host), headers=_get_headers(token), json=body)
    _raise_for_status(response)
    return response.json()


def get_endpoint(name: str, host: Optional[str] = None, token: Optional[str] = None) -> dict:
    """Get an AI Gateway V2 endpoint by name.

    Args:
        name: Endpoint name.
        host: Databricks workspace URL. Defaults to DATABRICKS_HOST env var.
        token: Databricks PAT. Defaults to DATABRICKS_TOKEN env var.

    Returns:
        Endpoint dict.
    """
    response = requests.get(f"{_get_base_url(host)}/{name}", headers=_get_headers(token))
    _raise_for_status(response)
    return response.json()


def list_endpoints(
    page_size: int = 1000,
    page_token: Optional[str] = None,
    host: Optional[str] = None,
    token: Optional[str] = None,
) -> dict:
    """List AI Gateway V2 endpoints.

    Args:
        page_size: Number of results per page (1–1000, default 1000).
        page_token: Token for the next page of results.
        host: Databricks workspace URL. Defaults to DATABRICKS_HOST env var.
        token: Databricks PAT. Defaults to DATABRICKS_TOKEN env var.

    Returns:
        Dict with 'endpoints' list and optional 'next_page_token'.
    """
    params = {"page_size": page_size}
    if page_token:
        params["page_token"] = page_token
    response = requests.get(_get_base_url(host), headers=_get_headers(token), params=params)
    _raise_for_status(response)
    return response.json()


def update_endpoint(
    name: str,
    destinations: Optional[List[dict]] = None,
    routing_strategy: Optional[str] = None,
    fallback: Optional[dict] = None,
    rate_limits: Optional[List[dict]] = None,
    tags: Optional[List[dict]] = None,
    inference_table_config: Optional[dict] = None,
    host: Optional[str] = None,
    token: Optional[str] = None,
) -> dict:
    """Partially update an AI Gateway V2 endpoint.

    Only the fields you pass will be updated; the update_mask is built automatically.
    Pass any combination of the config fields below.

    Args:
        name: Endpoint name.
        destinations: New list of destination() objects.
        routing_strategy: New routing strategy.
        fallback: New fallback_config() object.
        rate_limits: New list of rate_limit() objects (replaces existing).
        tags: New list of tag() objects (replaces existing).
        inference_table_config: New inference_table() object.
        host: Databricks workspace URL. Defaults to DATABRICKS_HOST env var.
        token: Databricks PAT. Defaults to DATABRICKS_TOKEN env var.

    Returns:
        Operation dict with the updated Endpoint in 'response'.
    """
    config = {}
    mask_parts = []

    if destinations is not None:
        config["destinations"] = destinations
        mask_parts.append("config.destinations")
    if routing_strategy is not None:
        config["routing_strategy"] = routing_strategy
        mask_parts.append("config.routing_strategy")
    if fallback is not None:
        config["fallback"] = fallback
        mask_parts.append("config.fallback")
    if rate_limits is not None:
        config["rate_limits"] = rate_limits
        mask_parts.append("config.rate_limits")
    if tags is not None:
        config["tags"] = tags
        mask_parts.append("config.tags")
    if inference_table_config is not None:
        config["inference_table"] = inference_table_config
        mask_parts.append("config.inference_table")

    if not mask_parts:
        raise ValueError("At least one field to update must be provided.")

    url = f"{_get_base_url(host)}/{name}"
    response = requests.patch(
        url,
        headers=_get_headers(token),
        params={"update_mask": ",".join(mask_parts)},
        json={"config": config},
    )
    _raise_for_status(response)
    return response.json()


def delete_endpoint(name: str, host: Optional[str] = None, token: Optional[str] = None) -> None:
    """Delete an AI Gateway V2 endpoint.

    Note: System-managed (pay-per-token default) endpoints cannot be deleted.

    Args:
        name: Endpoint name.
        host: Databricks workspace URL. Defaults to DATABRICKS_HOST env var.
        token: Databricks PAT. Defaults to DATABRICKS_TOKEN env var.
    """
    response = requests.delete(f"{_get_base_url(host)}/{name}", headers=_get_headers(token))
    _raise_for_status(response)
