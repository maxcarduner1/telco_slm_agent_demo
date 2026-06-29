"""Toy 3-step research agent: 3 sequential components, plain Python.

clarifier -> answerer -> polisher. Same component count as WanderBricks but
no LangGraph. Tests that the optimization loop handles a 3-component agent
without any framework wiring.
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


@mlflow.trace(name="clarifier")
def _clarify(question: str) -> str:
    return _call("clarifier", question=question)


@mlflow.trace(name="answerer")
def _answer(question: str, clarification: str) -> str:
    return _call("answerer", question=question, clarification=clarification)


@mlflow.trace(name="polisher")
def _polish(question: str, draft: str) -> str:
    return _call("polisher", question=question, draft=draft)


def predict(inputs: dict) -> str:
    q = inputs["question"]
    clarification = _clarify(q)
    draft = _answer(q, clarification)
    return _polish(q, draft)
