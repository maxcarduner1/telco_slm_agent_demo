"""
HotpotQA Agent

One LLM call behind one AI Gateway V2 endpoint. Used to demonstrate that
the optimization loop is agent-agnostic.
"""

import os

import mlflow
from databricks_openai import DatabricksOpenAI

mlflow.openai.autolog()

_DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "agent_config.yaml")
CONFIG_PATH = os.getenv("AGENT_CONFIG_PATH", _DEFAULT_CONFIG)
model_config = mlflow.models.ModelConfig(development_config=CONFIG_PATH)

_gw_endpoints = model_config.get("gateway_endpoints")
_prompt_names = model_config.get("prompt_registry")

_endpoint_name = _gw_endpoints["endpoint"]["smart_endpoint"]

_client = DatabricksOpenAI(use_ai_gateway=True)


def load_prompts():
    """Load the QA prompt from the registry. The optimizer's monkey-patch on
    PromptVersion.template transparently substitutes GEPA candidates.
    """
    return {
        key: mlflow.genai.load_prompt(f"prompts:/{name}@production", link_to_model=False)
        for key, name in _prompt_names.items()
    }


@mlflow.trace(name="endpoint")
def predict(inputs):
    """Single QA call. inputs: {"context": str, "question": str}."""
    pv = load_prompts()["endpoint"]
    user_message = pv.format(context=inputs["context"], question=inputs["question"])
    resp = _client.chat.completions.create(
        model=_endpoint_name,
        messages=[{"role": "user", "content": user_message}],
    )
    return resp.choices[0].message.content
