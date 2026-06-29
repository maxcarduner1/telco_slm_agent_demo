"""Minimal reference agent for smart-model-upgrades.

Satisfies the three BYOA requirements (see README):

1. Exposes predict(inputs: dict) -> str.
2. Loads its prompt via mlflow.genai.load_prompt() from the registry.
3. Enables MLflow autologging so token/latency are extractable from spans.

Drop this file next to a matching agent config YAML:

    gateway_endpoints:
      endpoint:
        smart_endpoint: my-endpoint
        initial_model: databricks-gpt-5-4-nano
    prompt_registry:
      endpoint: "my_catalog.my_schema.my_prompt"
"""
import os

import mlflow
import yaml
from databricks_openai import DatabricksOpenAI

# --- autologging: requirement (3) ------------------------------------------
mlflow.openai.autolog()

# --- config: load endpoint name + prompt name -------------------------------
CONFIG_PATH = os.getenv("AGENT_CONFIG_PATH", "agent_config.yaml")
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

_endpoint = cfg["gateway_endpoints"]["endpoint"]["smart_endpoint"]
_prompt_full_name = cfg["prompt_registry"]["endpoint"]

_client = DatabricksOpenAI(use_ai_gateway=True)


@mlflow.trace(name="endpoint")
def predict(inputs: dict) -> str:
    """Requirement (1) + (2): the signature the library expects, and prompts
    loaded from the registry so the optimizer can inject GEPA candidates."""
    pv = mlflow.genai.load_prompt(f"prompts:/{_prompt_full_name}@production")
    user_message = pv.format(**inputs)
    resp = _client.chat.completions.create(
        model=_endpoint,
        messages=[{"role": "user", "content": user_message}],
    )
    return resp.choices[0].message.content
