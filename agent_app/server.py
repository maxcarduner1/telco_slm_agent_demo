"""Agent server entry point using LongRunningAgentServer.

Follows the agent-langgraph-advanced template pattern:
- Background task execution (agent completes even if client disconnects)
- Cursor-based SSE streaming for reliable delivery
- Lakebase checkpointing without corruption risk
"""

import logging
import os
from contextlib import asynccontextmanager

import mlflow
from databricks_ai_bridge.long_running import LongRunningAgentServer

mlflow.langchain.autolog()

logger = logging.getLogger(__name__)

# Import the agent module to register @invoke/@stream handlers
import agent_app.agent  # noqa: F401

from agent_app.memory import (
    init_lakebase_config,
    lakebase_context,
    set_lakebase_resources,
)

# MLflow experiment for traces
MLFLOW_EXPERIMENT_ID = os.environ.get("MLFLOW_EXPERIMENT_ID", "")
if MLFLOW_EXPERIMENT_ID:
    try:
        mlflow.set_experiment(experiment_id=MLFLOW_EXPERIMENT_ID)
    except Exception as e:
        logger.warning(f"Could not set MLflow experiment: {e}")

# Initialize Lakebase config
try:
    LAKEBASE_CONFIG = init_lakebase_config()
except ValueError as e:
    logger.warning(f"Lakebase not configured: {e}")
    LAKEBASE_CONFIG = None

# Create the LongRunningAgentServer
server_kwargs = {
    "enable_chat_proxy": True,
    "task_timeout_seconds": float(os.getenv("TASK_TIMEOUT_SECONDS", "300")),
    "poll_interval_seconds": float(os.getenv("POLL_INTERVAL_SECONDS", "1.0")),
}

if LAKEBASE_CONFIG:
    server_kwargs["db_autoscaling_endpoint"] = LAKEBASE_CONFIG.autoscaling_endpoint
    server_kwargs["db_project"] = LAKEBASE_CONFIG.autoscaling_project
    server_kwargs["db_branch"] = LAKEBASE_CONFIG.autoscaling_branch

agent_server = LongRunningAgentServer("ResponsesAgent", **server_kwargs)

# Expose the app for uvicorn
app = agent_server.app


# Override the lifespan to set up Lakebase resources
_original_lifespan = app.router.lifespan_context


@asynccontextmanager
async def _lifespan(app):
    if LAKEBASE_CONFIG:
        try:
            async with lakebase_context(LAKEBASE_CONFIG) as (checkpointer, store):
                await checkpointer.setup()
                await store.setup()
                logger.info("Lakebase setup complete")

                app.state.checkpointer = checkpointer
                app.state.store = store
                set_lakebase_resources(checkpointer, store)

                try:
                    async with _original_lifespan(app):
                        yield
                except Exception as exc:
                    logger.warning(
                        "Long-running DB init failed: %s. Background mode disabled.",
                        exc,
                    )
                    yield
        except Exception as exc:
            logger.error("Lakebase session setup failed: %s", exc)
            logger.warning("Running without Lakebase persistence.")
            async with _original_lifespan(app):
                yield
    else:
        logger.info("Running without Lakebase persistence (not configured).")
        async with _original_lifespan(app):
            yield


app.router.lifespan_context = _lifespan


def main():
    agent_server.run(app_import_string="agent_app.server:app")


if __name__ == "__main__":
    main()
