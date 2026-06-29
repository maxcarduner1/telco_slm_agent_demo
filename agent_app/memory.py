"""Lakebase memory configuration for conversation persistence.

Follows the agent-langgraph-advanced template pattern:
- Long-lived resources opened at app startup (lifespan)
- Per-request acquire_lakebase_resources() yields those shared resources
"""

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional, Tuple

from databricks_langchain import AsyncCheckpointSaver, AsyncDatabricksStore

logger = logging.getLogger(__name__)

# Long-lived Lakebase resources, opened once at startup and reused across requests.
_lakebase_resources: Optional[Tuple[AsyncCheckpointSaver, AsyncDatabricksStore]] = None


def set_lakebase_resources(
    checkpointer: AsyncCheckpointSaver, store: AsyncDatabricksStore
) -> None:
    global _lakebase_resources
    _lakebase_resources = (checkpointer, store)


@dataclass(frozen=True)
class LakebaseConfig:
    """Configuration for Lakebase connection."""
    autoscaling_endpoint: Optional[str] = None
    autoscaling_project: Optional[str] = None
    autoscaling_branch: Optional[str] = None
    embedding_endpoint: str = "otel-embedding2-300m"
    embedding_dims: int = 768
    memory_schema: Optional[str] = None

    @property
    def description(self) -> str:
        return (
            self.autoscaling_endpoint
            or f"{self.autoscaling_project}/{self.autoscaling_branch}"
        )


def init_lakebase_config() -> LakebaseConfig:
    """Initialize Lakebase config from environment variables."""
    endpoint = os.environ.get("LAKEBASE_AUTOSCALING_ENDPOINT") or None
    project = os.environ.get("LAKEBASE_AUTOSCALING_PROJECT") or None
    branch = os.environ.get("LAKEBASE_AUTOSCALING_BRANCH") or None

    has_autoscaling = project and branch

    if not endpoint and not has_autoscaling:
        raise ValueError(
            "Lakebase configuration required. Set LAKEBASE_AUTOSCALING_PROJECT + "
            "LAKEBASE_AUTOSCALING_BRANCH or LAKEBASE_AUTOSCALING_ENDPOINT."
        )

    # Priority: endpoint > project+branch (mutually exclusive in the library)
    if endpoint:
        project = None
        branch = None
    else:
        endpoint = None

    return LakebaseConfig(
        autoscaling_endpoint=endpoint,
        autoscaling_project=project,
        autoscaling_branch=branch,
        embedding_endpoint=os.environ.get("EMBEDDING_ENDPOINT", "otel-embedding2-300m"),
        embedding_dims=768,
        memory_schema=os.environ.get("LAKEBASE_AGENT_MEMORY_SCHEMA") or None,
    )


@asynccontextmanager
async def lakebase_context(config: LakebaseConfig):
    """Context manager that yields (checkpointer, store) connected to Lakebase."""
    # Non-default schemas may not have the pgvector extension in their search
    # path. Keep checkpointing enabled, but only enable semantic long-term
    # memory indexing for the default V1 schema until per-schema pgvector is
    # fully validated.
    store_embedding_endpoint = (
        config.embedding_endpoint
        if config.memory_schema in (None, "agent_memory")
        else None
    )
    store_embedding_dims = (
        config.embedding_dims
        if store_embedding_endpoint is not None
        else None
    )

    async with AsyncCheckpointSaver(
        autoscaling_endpoint=config.autoscaling_endpoint,
        project=config.autoscaling_project,
        branch=config.autoscaling_branch,
        schema=config.memory_schema,
    ) as checkpointer, AsyncDatabricksStore(
        autoscaling_endpoint=config.autoscaling_endpoint,
        project=config.autoscaling_project,
        branch=config.autoscaling_branch,
        embedding_endpoint=store_embedding_endpoint,
        embedding_dims=store_embedding_dims,
        schema=config.memory_schema,
    ) as store:
        yield checkpointer, store


@asynccontextmanager
async def acquire_lakebase_resources(config: LakebaseConfig):
    """Yield (checkpointer, store) for use in a request handler.

    If the lifespan populated long-lived resources, yield those without closing.
    Otherwise fall back to opening a fresh per-call context.
    """
    if _lakebase_resources is not None:
        yield _lakebase_resources
    else:
        async with lakebase_context(config) as resources:
            yield resources
