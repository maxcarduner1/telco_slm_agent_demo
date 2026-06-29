"""Prediction wrapper for Smart Model Upgrades.

The optimizer calls `predict(inputs: dict)` repeatedly. This wrapper reuses the
TelcoGPT LangGraph agent while keeping evaluation isolated from the live app.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from agent_app.agent import create_telco_agent


def _final_text(messages: list[Any]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def predict(inputs: dict) -> str:
    """Run TelcoGPT for one eval row and return the final assistant text.

    Expected input keys:
    - `question`: required user question.
    - `follow_up`: optional second user turn for memory-style evals.
    """
    question = inputs.get("question") or inputs.get("query")
    if not question:
        raise ValueError("predict inputs must include 'question' or 'query'")

    agent = create_telco_agent()
    state = {"messages": [HumanMessage(content=str(question))]}
    result = agent.invoke(state)

    follow_up = inputs.get("follow_up")
    if follow_up:
        prior_messages = list(result.get("messages", []))
        state = {"messages": [*prior_messages, HumanMessage(content=str(follow_up))]}
        result = agent.invoke(state)

    return _final_text(result.get("messages", []))
