"""Toy translator agent: 1 component, multi-key input.

Tests the BYOA contract for an agent whose input dict has more than one field.
"""
import os

import mlflow
import yaml
from databricks_openai import DatabricksOpenAI

mlflow.openai.autolog()

CONFIG_PATH = os.getenv(
    "AGENT_CONFIG_PATH",
    os.path.join(os.path.dirname(__file__), "agent_config.yaml"),
)
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

_endpoint = cfg["gateway_endpoints"]["translator"]["smart_endpoint"]
_prompt_name = cfg["prompt_registry"]["translator"]

_client = DatabricksOpenAI(use_ai_gateway=True)


@mlflow.trace(name="translator")
def predict(inputs: dict) -> str:
    pv = mlflow.genai.load_prompt(f"prompts:/{_prompt_name}@production")
    msg = pv.format(text=inputs["text"], target_language=inputs["target_language"])
    resp = _client.chat.completions.create(
        model=_endpoint,
        messages=[{"role": "user", "content": msg}],
    )
    return resp.choices[0].message.content
