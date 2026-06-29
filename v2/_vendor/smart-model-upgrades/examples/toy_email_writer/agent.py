"""Toy email writer: 2 fan-out components.

Both components see the same input dict; their outputs are concatenated.
Tests a non-sequential topology -- subject_writer and body_writer don't
depend on each other.
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

_endpoints = {
    comp: ep["smart_endpoint"]
    for comp, ep in cfg["gateway_endpoints"].items()
}
_prompt_names = cfg["prompt_registry"]

_client = DatabricksOpenAI(use_ai_gateway=True)


def _call(comp: str, **vars) -> str:
    pv = mlflow.genai.load_prompt(f"prompts:/{_prompt_names[comp]}@production")
    msg = pv.format(**vars)
    resp = _client.chat.completions.create(
        model=_endpoints[comp],
        messages=[{"role": "user", "content": msg}],
    )
    return resp.choices[0].message.content


@mlflow.trace(name="subject_writer")
def _subject(recipient: str, topic: str) -> str:
    return _call("subject_writer", recipient=recipient, topic=topic)


@mlflow.trace(name="body_writer")
def _body(recipient: str, topic: str) -> str:
    return _call("body_writer", recipient=recipient, topic=topic)


def predict(inputs: dict) -> str:
    subject = _subject(inputs["recipient"], inputs["topic"])
    body = _body(inputs["recipient"], inputs["topic"])
    return f"Subject: {subject}\n\n{body}"
