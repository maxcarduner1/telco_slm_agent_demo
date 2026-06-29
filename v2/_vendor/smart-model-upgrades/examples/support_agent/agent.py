"""LangGraph support-agent example: classifier -> responder.

Two-node StateGraph. Each node loads its prompt fresh from MLflow Prompt
Registry on every invocation and calls its own AI Gateway endpoint via
ChatDatabricks (use_ai_gateway=True). Demonstrates the BYOA contract for a
real LangGraph agent.
"""
import os
from typing import TypedDict

import mlflow
import yaml
from databricks_langchain import ChatDatabricks
from langgraph.graph import StateGraph, START, END

mlflow.langchain.autolog()

CONFIG_PATH = os.getenv(
    "AGENT_CONFIG_PATH",
    os.path.join(os.path.dirname(__file__), "agent_config.yaml"),
)
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

llms = {
    comp: ChatDatabricks(endpoint=ep["smart_endpoint"], use_ai_gateway=True)
    for comp, ep in cfg["gateway_endpoints"].items()
}


class State(TypedDict):
    ticket: str
    category: str
    reply: str


def classifier_node(state: State) -> State:
    pv = mlflow.genai.load_prompt(
        f"prompts:/{cfg['prompt_registry']['classifier']}@production"
    )
    out = llms["classifier"].invoke(pv.format(ticket=state["ticket"]))
    return {"category": out.content}


def responder_node(state: State) -> State:
    pv = mlflow.genai.load_prompt(
        f"prompts:/{cfg['prompt_registry']['responder']}@production"
    )
    out = llms["responder"].invoke(
        pv.format(ticket=state["ticket"], category=state["category"])
    )
    return {"reply": out.content}


graph = (
    StateGraph(State)
    .add_node("classifier", classifier_node)
    .add_node("responder", responder_node)
    .add_edge(START, "classifier")
    .add_edge("classifier", "responder")
    .add_edge("responder", END)
    .compile()
)


@mlflow.trace(name="support_agent")
def predict(inputs: dict) -> str:
    return graph.invoke({"ticket": inputs["ticket"]})["reply"]
