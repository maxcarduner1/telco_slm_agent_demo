"""
WanderBricks Travel Agent

Architecture:
  supervisor -> [query_rewriter -> genie | enrichment (tool-calling agent)] -> supervisor -> FINISH

Three LLM calls, each behind its own AI Gateway endpoint:
  1. supervisor:      routes + synthesizes final answer (fast, cheap model)
  2. query_rewriter:  rewrites user question for Genie SQL agent (medium model)
  3. enrichment:      tool-calling agent for weather lookup (standard ToolNode pattern)
"""

import operator
import os
from typing import Annotated, Any, Callable, Dict, Generator, List, Literal

import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)

from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, ToolMessage,
)
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from .tools import all_tools

# ---------------------------------------------------------------------------
# State + models (pure, no side effects)
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages(format="langchain-openai")]
    genie_query: str                   # rewritten query for Genie
    genie_results: Annotated[List[str], operator.add]        # accumulated Genie responses
    enrichment_results: Annotated[List[str], operator.add]  # accumulated weather data
    supervisor_reasoning: str          # latest supervisor thought


class SupervisorDecision(BaseModel):
    reasoning: str = Field(description="Brief, friendly status update for the user (e.g. 'Let me search for places in Paris!')")
    next_step: Literal["query_rewriter", "enrichment", "FINISH"] = Field(
        description="Route to query_rewriter for DB data, enrichment for weather, or FINISH to respond"
    )
    response: str = Field(
        default="",
        description="Final user-facing response (only when next_step=FINISH)",
    )


class EnrichmentState(TypedDict):
    messages: Annotated[list, add_messages(format="langchain-openai")]


# ---------------------------------------------------------------------------
# Helpers (pure, only use state)
# ---------------------------------------------------------------------------

def get_user_question(state: AgentState) -> str:
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            return m.content
    return ""


def build_scratchpad(state: AgentState) -> str:
    parts = []
    genie = state["genie_results"]
    weather = state["enrichment_results"]
    if genie:
        parts.append("Property data:\n" + "\n---\n".join(
            f"[Query {i+1}]: {r}" for i, r in enumerate(genie)
        ))
    if weather:
        parts.append("Weather data:\n" + "\n".join(weather))
    return "\n\n".join(parts) if parts else "(nothing gathered yet)"


def message_text(msg) -> str:
    """Flatten BaseMessage.content to a string.

    ChatOpenAI against a V2 gateway may return content as a list of content
    blocks (e.g. from thinking-capable models) instead of a plain string.
    """
    content = getattr(msg, "content", msg)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


# ---------------------------------------------------------------------------
# ResponsesAgent wrapper for MLflow deployment
# ---------------------------------------------------------------------------

class LangGraphResponsesAgent(ResponsesAgent):
    """Wraps a compiled LangGraph workflow for MLflow serving."""

    def __init__(self, agent: CompiledStateGraph, recursion_limit: int = 25):
        self.agent = agent
        self.recursion_limit = recursion_limit

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        last_item = None
        for event in self.predict_stream(request):
            if event.type == "response.output_item.done":
                last_item = event.item
        outputs = [last_item] if last_item else []
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        cc_msgs = to_chat_completions_input([i.model_dump() for i in request.input])
        try:
            for _, events in self.agent.stream(
                {"messages": cc_msgs},
                stream_mode=["updates"],
                config={"recursion_limit": self.recursion_limit},
            ):
                for node_data in events.values():
                    if "messages" in node_data:
                        yield from output_to_responses_items_stream(node_data["messages"])
        except GraphRecursionError:
            # Last-resort safety net: the in-graph supervisor.must_finish path
            # should normally produce a clean final answer before this fires.
            # If it does fire (e.g. enrichment subgraph recurses past its
            # limit), surface a graceful partial-answer rather than aborting
            # the whole `predict` call -- callers (e.g. an optimizer) can score
            # this as a low-quality eval and route around it.
            yield from output_to_responses_items_stream([
                AIMessage(content=(
                    "I wasn't able to fully complete this request within the "
                    "available reasoning budget. Try a more specific question, "
                    "or increase recursion_limit / max_worker_rounds."
                ))
            ])


# ---------------------------------------------------------------------------
# Parameterized graph builder
# ---------------------------------------------------------------------------

def build_graph(
    llms: dict,
    load_prompts_fn: Callable,
    genie_agent,
    max_worker_rounds: int = 4,
    enrichment_recursion_limit: int = 10,
) -> CompiledStateGraph:
    """Build the WanderBricks agent graph.

    Args:
        llms: {"supervisor": ChatDatabricks, "query_rewriter": ..., "enrichment": ...}
        load_prompts_fn: () -> {"supervisor": str/PromptVersion, ...}
        genie_agent: GenieAgent instance
        max_worker_rounds: force FINISH after this many worker dispatches
        enrichment_recursion_limit: max tool-calling loops for enrichment
    """
    supervisor_structured_llm = llms["supervisor"].with_structured_output(
        SupervisorDecision, method="json_schema"
    )
    enrichment_llm_with_tools = llms["enrichment"].bind_tools(all_tools)

    # -- Nodes (closures over llms, load_prompts_fn, etc.) --

    def supervisor_node(state: AgentState) -> Command[Literal["query_rewriter", "enrichment", "__end__"]]:
        """Route to a worker or finish with a direct response."""
        prompts = load_prompts_fn()
        genie = state["genie_results"]
        weather = state["enrichment_results"]
        calls_used = len(genie) + len(weather)
        must_finish = calls_used >= max_worker_rounds

        prompt = prompts["supervisor"].format(
            user_question=get_user_question(state),
            scratchpad=build_scratchpad(state),
            calls_used=str(calls_used),
        )

        decision = supervisor_structured_llm.invoke([HumanMessage(content=prompt)])

        if decision.next_step == "FINISH" or must_finish:
            response = decision.response
            # If the supervisor wanted to keep routing but we forced a finish,
            # `response` is empty/garbage. Re-prompt for a clean synthesis from
            # the scratchpad so the user gets a real answer instead of "".
            if must_finish and (not response or decision.next_step != "FINISH"):
                synthesis_prompt = (
                    f"You have used your full search budget. Based ONLY on what "
                    f"you've gathered, write a final user-facing answer to the "
                    f"original question. Do not request more searches.\n\n"
                    f"Original question: {get_user_question(state)}\n\n"
                    f"Gathered so far:\n{build_scratchpad(state)}"
                )
                final = llms["supervisor"].invoke([HumanMessage(content=synthesis_prompt)])
                response = message_text(final).strip() or response
            return Command(
                goto=END,
                update={
                    "messages": [AIMessage(content=response)],
                    "supervisor_reasoning": decision.reasoning,
                },
            )

        return Command(
            goto=decision.next_step,
            update={
                "messages": [AIMessage(content=decision.reasoning)],
                "supervisor_reasoning": decision.reasoning,
            },
        )

    def query_rewriter_node(state: AgentState) -> Command[Literal["genie"]]:
        """Rewrite user question into a precise Genie query."""
        prompts = load_prompts_fn()
        genie_results = state.get("genie_results", [])
        previous = "\n---\n".join(
            f"[Query {i+1}]: {r}" for i, r in enumerate(genie_results)
        ) if genie_results else "(none yet)"

        prompt = prompts["query_rewriter"].format(
            user_question=get_user_question(state),
            supervisor_reasoning=state.get("supervisor_reasoning", ""),
            previous_queries=previous,
        )

        response = llms["query_rewriter"].invoke([HumanMessage(content=prompt)])

        return Command(
            goto="genie",
            update={"genie_query": message_text(response).strip()},
        )

    def genie_node(state: AgentState) -> Command[Literal["supervisor"]]:
        """Send rewritten query to Genie, collect results."""
        query = state.get("genie_query", "")
        if not query:
            return Command(
                goto="supervisor",
                update={"genie_results": ["(empty query)"]},
            )

        result = genie_agent.invoke({"messages": [HumanMessage(content=query)]})
        text = "\n".join(
            m.content for m in result.get("messages", [])
            if hasattr(m, "content") and m.content
        ).strip() or "(no results)"

        return Command(
            goto="supervisor",
            update={"genie_results": [text]},
        )

    # -- Enrichment subgraph (tool-calling loop) --
    #
    # Magnum-style budget guard: the subgraph counts tool turns by tallying
    # ToolMessages on its state; once we're near the recursion limit and the
    # LLM still wants to call tools, we make a second LLM call asking for a
    # tool-free final answer based on what's been gathered, then scrub
    # tool_calls so `tools_condition` routes to END.

    def enrichment_agent_node(state: EnrichmentState) -> dict:
        """LLM decides whether to call a weather tool or respond."""
        messages = state["messages"]
        tool_turns_used = sum(1 for m in messages if isinstance(m, ToolMessage))
        # Two extra recursion steps per tool turn (agent + tools); leave
        # headroom of one full turn for the forced-finish call below.
        budget_low = tool_turns_used >= max(1, (enrichment_recursion_limit // 2) - 1)

        response = enrichment_llm_with_tools.invoke(messages)

        if response.tool_calls and budget_low:
            forced = llms["enrichment"].invoke(
                messages + [HumanMessage(content=(
                    "You have used your full tool-call budget. Provide a final "
                    "answer based ONLY on the data you have already gathered. "
                    "Do not call any more tools."
                ))]
            )
            forced.tool_calls = []
            forced.additional_kwargs.pop("tool_calls", None)
            return {"messages": [forced]}

        return {"messages": [response]}

    _enrichment_builder = StateGraph(EnrichmentState)
    _enrichment_builder.add_node("agent", enrichment_agent_node)
    _enrichment_builder.add_node("tools", ToolNode(all_tools))
    _enrichment_builder.add_edge(START, "agent")
    _enrichment_builder.add_conditional_edges("agent", tools_condition, ["tools", "__end__"])
    _enrichment_builder.add_edge("tools", "agent")
    enrichment_subgraph = _enrichment_builder.compile()

    def enrichment_node(state: AgentState) -> Command[Literal["supervisor"]]:
        """Wrapper that invokes the enrichment subgraph from the main graph."""
        prompts = load_prompts_fn()
        user_q = get_user_question(state)
        reasoning = state.get("supervisor_reasoning", "")

        enrichment_prompt = prompts["enrichment"].format(
            user_question=user_q,
            supervisor_reasoning=reasoning,
        )

        subgraph_input = {
            "messages": [
                HumanMessage(content=enrichment_prompt),
            ]
        }

        try:
            result = enrichment_subgraph.invoke(
                subgraph_input,
                {"recursion_limit": enrichment_recursion_limit},
            )
            final_msg = result["messages"][-1]
            text = message_text(final_msg)
        except GraphRecursionError:
            text = "(weather lookup didn't terminate within the available steps)"
            return Command(
                goto="supervisor",
                update={"enrichment_results": [text]},
            )

        return Command(
            goto="supervisor",
            update={"enrichment_results": [text]},
        )

    # -- Build main graph --

    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("query_rewriter", query_rewriter_node)
    graph.add_node("genie", genie_node)
    graph.add_node("enrichment", enrichment_node)

    graph.add_edge(START, "supervisor")
    # All other routing via Command returns

    return graph.compile()


# ---------------------------------------------------------------------------
# Default serving setup (standard customer pattern)
# ---------------------------------------------------------------------------

from databricks_langchain import ChatDatabricks
from databricks_langchain.genie import GenieAgent
from databricks.sdk import WorkspaceClient

mlflow.langchain.autolog()

_DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "agent_config.yaml")
CONFIG_PATH = os.getenv("AGENT_CONFIG_PATH", _DEFAULT_CONFIG)
model_config = mlflow.models.ModelConfig(development_config=CONFIG_PATH)

gateway = model_config.get("gateway_endpoints")
resources = model_config.get("databricks_resources")
prompt_names = model_config.get("prompt_registry")


def load_prompts():
    """Load all prompts from MLflow Prompt Registry (cached 60s per alias)."""
    return {
        key: mlflow.genai.load_prompt(f"prompts:/{name}@production", link_to_model=False)
        for key, name in prompt_names.items()
    }


# V2 gateway endpoints aren't reachable via ChatDatabricks -- use ChatOpenAI
from langchain_openai import ChatOpenAI
from smart_model_upgrades import ai_gateway as gw

_gw_endpoints = {
    comp: gateway[comp]["smart_endpoint"]
    for comp in gateway
}
_any_ep = list(_gw_endpoints.values())[0]
_gw_base_url = gw.get_endpoint(_any_ep).get("ai_gateway_url", "").rstrip("/") + "/mlflow/v1"

llms = {
    comp: ChatOpenAI(
        model=ep_name,
        base_url=_gw_base_url,
        api_key=os.environ["DATABRICKS_TOKEN"],
        disabled_params={"parallel_tool_calls": None},
    )
    for comp, ep_name in _gw_endpoints.items()
}

genie_agent = GenieAgent(
    genie_space_id=resources["genie_space_id"],
    genie_agent_name="WanderBricks",
    client=WorkspaceClient(),
)

graph: CompiledStateGraph = build_graph(
    llms=llms,
    load_prompts_fn=load_prompts,
    genie_agent=genie_agent,
    max_worker_rounds=model_config.get("max_worker_rounds"),
    enrichment_recursion_limit=model_config.get("enrichment_recursion_limit"),
)

AGENT = LangGraphResponsesAgent(
    graph,
    recursion_limit=model_config.get("recursion_limit") or 25,
)
mlflow.models.set_model(AGENT)


def predict(inputs: dict) -> str:
    """BYOA-contract entry point: dict in, string out.

    Wraps AGENT.predict with the ResponsesAgent request shape so the generic
    optimization loop can call this agent the same way it calls hotpotqa_agent
    or any toy agent.
    """
    request = ResponsesAgentRequest(
        input=[{"role": "user", "content": inputs["question"]}]
    )
    response = AGENT.predict(request)
    if not response.output:
        return ""
    block = response.output[-1].content[0]
    return block["text"] if isinstance(block, dict) else block.text
