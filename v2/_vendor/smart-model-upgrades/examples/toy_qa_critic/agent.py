"""Toy QA-critic agent: 2 sequential components, plain Python (no framework).

answerer drafts an answer, critic revises it. Each component has its own
gateway endpoint and prompt registry entry; the second call depends on the
first call's output.
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


@mlflow.trace(name="answerer")
def _answer(question: str) -> str:
    return _call("answerer", question=question)


@mlflow.trace(name="critic")
def _critique(question: str, draft: str) -> str:
    return _call("critic", question=question, draft=draft)


def predict(inputs: dict) -> str:
    draft = _answer(inputs["question"])
    return _critique(inputs["question"], draft)
