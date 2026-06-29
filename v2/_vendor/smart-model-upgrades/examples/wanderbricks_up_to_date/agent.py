"""WanderBricks 'up-to-date prompts' scenario.

Same agent code as examples/wanderbricks/. The variation is in the YAMLs in
this directory: richer initial prompts and older initial models -- the
"customer has invested in prompt engineering but their LLMs are a generation
behind" scenario.

Sets `AGENT_CONFIG_PATH` to this dir's `agent_config.yaml`, then re-exports
`predict` from the shared WanderBricks module.
"""
import os

os.environ["AGENT_CONFIG_PATH"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "agent_config.yaml",
)

from examples.wanderbricks.agent import predict, AGENT, graph, model_config  # noqa: E402

__all__ = ["predict", "AGENT", "graph", "model_config"]
