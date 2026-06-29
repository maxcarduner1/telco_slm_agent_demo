"""LangGraph agent with Responses API handlers.

Uses the agent-langgraph-advanced template pattern:
- @invoke() and @stream() handlers for the LongRunningAgentServer
- Per-request agent creation with acquire_lakebase_resources
- astream with stream_mode=["updates", "messages"] for proper checkpointing
"""

import json
import logging
import os
import time
from typing import Any, AsyncGenerator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata

try:
    import mlflow
except Exception:  # pragma: no cover - local fallback when mlflow deps are unavailable
    mlflow = None
import uuid_utils
from databricks_langchain import ChatDatabricks
from langchain_core.messages import AIMessageChunk, ToolMessage
from langgraph.prebuilt import create_react_agent
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    create_function_call_item,
    create_function_call_output_item,
    create_text_output_item,
    to_chat_completions_input,
)

from agent_app.memory import (
    acquire_lakebase_resources,
    init_lakebase_config,
)
from agent_app.prompts import SYSTEM_PROMPT
from agent_app.tools import get_uc_function_tools, get_rag_tools

logger = logging.getLogger(__name__)
if mlflow is not None:
    mlflow.langchain.autolog()

LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT", "databricks-claude-sonnet-4")
MAX_MODEL_HISTORY_MESSAGES = int(os.environ.get("MAX_MODEL_HISTORY_MESSAGES", "18"))
MAX_SUMMARY_SOURCE_MESSAGES = int(os.environ.get("MAX_SUMMARY_SOURCE_MESSAGES", "16"))

# Initialize config (may be None if Lakebase is not configured)
try:
    LAKEBASE_CONFIG = init_lakebase_config()
except ValueError:
    LAKEBASE_CONFIG = None


def create_telco_agent(checkpointer=None, store=None):
    """Create the LangGraph telco agent."""
    model = ChatDatabricks(endpoint=LLM_ENDPOINT)
    tools = get_uc_function_tools() + get_rag_tools()

    return create_react_agent(
        model=model,
        tools=tools,
        prompt=SYSTEM_PROMPT,
        pre_model_hook=_compact_model_history,
        checkpointer=checkpointer,
        store=store,
    )


def _message_excerpt(msg: Any, max_chars: int = 220) -> str:
    content = getattr(msg, "content", "")
    if not isinstance(content, str):
        try:
            content = json.dumps(content)
        except Exception:
            content = str(content)
    content = " ".join(content.split())
    return content[:max_chars] + ("..." if len(content) > max_chars else "")


def _summarize_older_messages(messages: list[Any]) -> SystemMessage | None:
    if not messages:
        return None

    lines = [
        "Earlier conversation summary (compacted to protect the context window):"
    ]
    summarized = 0
    for msg in messages:
        if summarized >= MAX_SUMMARY_SOURCE_MESSAGES:
            break
        if isinstance(msg, HumanMessage):
            lines.append(f"- User: {_message_excerpt(msg)}")
            summarized += 1
        elif isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            lines.append(f"- Assistant: {_message_excerpt(msg)}")
            summarized += 1

    if summarized == 0:
        return None
    if len(messages) > summarized:
        lines.append(f"- Additional older tool/detail messages omitted: {len(messages) - summarized}.")
    return SystemMessage(content="\n".join(lines))


def _safe_recent_boundary(messages: list[Any], start: int) -> int:
    """Avoid starting model input with an orphan ToolMessage."""
    while start > 0 and isinstance(messages[start], ToolMessage):
        start -= 1
    return start


def _compact_model_history(state: dict[str, Any]) -> dict[str, Any]:
    """Bound model input while preserving full checkpoint state in Lakebase."""
    messages = list(state.get("messages", []))
    if len(messages) <= MAX_MODEL_HISTORY_MESSAGES:
        return {"llm_input_messages": messages}

    start = _safe_recent_boundary(
        messages,
        max(0, len(messages) - MAX_MODEL_HISTORY_MESSAGES),
    )
    older = messages[:start]
    recent = messages[start:]
    summary = _summarize_older_messages(older)
    compacted = ([summary] if summary else []) + recent
    logger.info(
        "Compacted model input history from %s to %s messages",
        len(messages),
        len(compacted),
    )
    return {"llm_input_messages": compacted}


def _get_thread_id(request: ResponsesAgentRequest) -> str:
    """Extract thread_id from request custom_inputs or generate one."""
    ci = dict(request.custom_inputs or {})
    if "thread_id" in ci and ci["thread_id"]:
        return str(ci["thread_id"])
    if request.context and getattr(request.context, "conversation_id", None):
        return str(request.context.conversation_id)
    return str(uuid_utils.uuid7())


def _latest_user_messages(request: ResponsesAgentRequest) -> list:
    """Extract only the new (trailing) user messages from the request input.

    The frontend sends the full conversation history on every request.
    The checkpointer already stores prior turns, so we only pass the
    messages that come after the last assistant turn — avoiding duplicates
    that corrupt the checkpoint state.
    """
    items = [i.model_dump() for i in request.input]
    # Find the index right after the last assistant message
    last_assistant_idx = -1
    for i, item in enumerate(items):
        if item.get("role") == "assistant":
            last_assistant_idx = i
    new_items = items[last_assistant_idx + 1 :]
    return to_chat_completions_input(new_items) if new_items else to_chat_completions_input(items)


@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    """Non-streaming invocation handler."""
    outputs = [
        event.item
        async for event in stream_handler(request)
        if event.type == "response.output_item.done"
    ]
    return ResponsesAgentResponse(output=outputs)


@stream()
async def stream_handler(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    """Streaming handler using astream for reliable checkpointing."""
    thread_id = _get_thread_id(request)
    if mlflow is not None:
        mlflow.update_current_trace(metadata={"mlflow.trace.session": thread_id})

    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

    # When using a checkpointer, only pass the latest user message(s).
    # The checkpoint already has the prior conversation history; sending
    # the full history would duplicate messages and corrupt the state.
    if LAKEBASE_CONFIG:
        messages = _latest_user_messages(request)
    else:
        messages = to_chat_completions_input([i.model_dump() for i in request.input])

    input_state: dict[str, Any] = {"messages": messages}

    if LAKEBASE_CONFIG:
        async with acquire_lakebase_resources(LAKEBASE_CONFIG) as (
            checkpointer,
            store,
        ):
            agent = create_telco_agent(checkpointer=checkpointer, store=store)
            try:
                async for event in _process_stream(agent, input_state, config):
                    yield event
            except Exception as e:
                err = str(e)
                is_invalid_tool_history = "tool_calls" in err and (
                    "ToolMessage" in err or "INVALID_CHAT_HISTORY" in err
                )
                if not is_invalid_tool_history:
                    raise

                logger.warning(
                    "Corrupted checkpoint for thread %s — clearing and retrying. Error: %s",
                    thread_id, e,
                )
                # Overwrite the corrupted checkpoint with a clean empty state.
                # Use uuid7 (time-ordered) so this checkpoint is always fetched
                # as "latest" (lexicographically greater than any prior uuid4).
                clean_config = {
                    "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
                }
                await checkpointer.aput(
                    clean_config,
                    Checkpoint(
                        v=1,
                        id=str(uuid_utils.uuid7()),
                        ts="1970-01-01T00:00:00+00:00",
                        channel_values={},
                        channel_versions={},
                        versions_seen={},
                        pending_sends=[],
                    ),
                    CheckpointMetadata(source="update", step=-1, writes=None, parents={}),
                    {},
                )
                full_messages = to_chat_completions_input(
                    [i.model_dump() for i in request.input]
                )
                async for event in _process_stream(
                    agent, {"messages": full_messages}, config
                ):
                    yield event
    else:
        agent = create_telco_agent()
        async for event in _process_stream(agent, input_state, config):
            yield event


async def _process_stream(
    agent, input_state: dict, config: dict
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    """Process agent astream events into Responses API stream events.

    Ported from agent-langgraph-advanced template's process_agent_astream_events.
    Handles text streaming, tool call streaming, and per-turn response wrapping.
    """
    response_id = f"resp_placeholder_{uuid_utils.uuid7().hex[:16]}"
    in_turn = False
    turn_output_items: list[dict] = []
    output_index = 0
    active_text_item_id: str | None = None
    active_text_content = ""
    active_tool_calls: dict[int, dict] = {}
    saw_text_output = False
    tool_outputs_for_fallback: list[str] = []

    def _response_obj(output: list | None = None) -> dict:
        return {
            "id": response_id,
            "created_at": time.time(),
            "object": "response",
            "output": output or [],
            "status": None,
        }

    def _start_turn():
        nonlocal in_turn, turn_output_items
        in_turn = True
        turn_output_items = []

    def _end_turn():
        nonlocal in_turn, active_text_item_id, active_text_content
        in_turn = False
        active_text_item_id = None
        active_text_content = ""

    def _tool_output_looks_empty(output: str) -> bool:
        try:
            parsed = json.loads(output)
        except Exception:
            parsed = None

        if isinstance(parsed, dict) and parsed.get("format") == "CSV":
            value = str(parsed.get("value", ""))
            rows = [line for line in value.splitlines() if line.strip()]
            return len(rows) <= 1

        normalized = output.strip().lower()
        return normalized.startswith("no relevant") or normalized in {
            "",
            "no data",
            "no results",
        }

    def _fallback_text() -> str:
        if tool_outputs_for_fallback and all(
            _tool_output_looks_empty(output) for output in tool_outputs_for_fallback
        ):
            return (
                "No data was returned for the requested filters and time period. "
                "I did not find rows to summarize. Try a wider lookback window or "
                "different region, site, metric, severity, or event type."
            )

        return (
            "I ran the requested tool, but did not receive a final synthesized answer "
            "from the orchestrator. Please retry or ask for a narrower summary."
        )

    async for event in agent.astream(
        input_state, config, stream_mode=["updates", "messages"]
    ):
        if event[0] == "messages":
            try:
                chunk = event[1][0]
                if not isinstance(chunk, AIMessageChunk):
                    continue

                if not in_turn:
                    _start_turn()
                    yield ResponsesAgentStreamEvent(
                        type="response.created",
                        response=_response_obj(),
                    )

                # Tool call chunks
                if chunk.tool_call_chunks:
                    for tc_chunk in chunk.tool_call_chunks:
                        idx = tc_chunk.get("index", 0)
                        name = tc_chunk.get("name") or ""
                        tc_id = tc_chunk.get("id") or ""
                        args = tc_chunk.get("args") or ""

                        if idx not in active_tool_calls:
                            item_id = str(uuid_utils.uuid7())
                            active_tool_calls[idx] = {
                                "item_id": item_id,
                                "name": name,
                                "args": "",
                                "call_id": tc_id,
                                "output_index": output_index,
                            }
                            output_index += 1
                            yield ResponsesAgentStreamEvent(
                                type="response.output_item.added",
                                item={
                                    "type": "function_call",
                                    "id": item_id,
                                    "call_id": tc_id,
                                    "name": name,
                                    "arguments": "",
                                },
                                output_index=active_tool_calls[idx]["output_index"],
                            )
                        else:
                            tc_info = active_tool_calls[idx]
                            if name and not tc_info["name"]:
                                tc_info["name"] = name
                            if tc_id and not tc_info["call_id"]:
                                tc_info["call_id"] = tc_id

                        if args:
                            active_tool_calls[idx]["args"] += args
                            yield ResponsesAgentStreamEvent(
                                type="response.function_call_arguments.delta",
                                delta=args,
                                item_id=active_tool_calls[idx]["item_id"],
                                output_index=active_tool_calls[idx]["output_index"],
                            )

                # Text content
                elif chunk.content:
                    content = chunk.content
                    if not active_text_item_id:
                        active_text_item_id = str(uuid_utils.uuid7())
                        active_text_content = ""
                        yield ResponsesAgentStreamEvent(
                            type="response.output_item.added",
                            item={
                                "type": "message",
                                "id": active_text_item_id,
                                "role": "assistant",
                                "status": "in_progress",
                                "content": [],
                            },
                            output_index=output_index,
                        )
                        yield ResponsesAgentStreamEvent(
                            type="response.content_part.added",
                            item_id=active_text_item_id,
                            output_index=output_index,
                            content_index=0,
                            part={"type": "output_text", "text": "", "annotations": []},
                        )

                    active_text_content += content
                    yield ResponsesAgentStreamEvent(
                        type="response.output_text.delta",
                        delta=content,
                        item_id=active_text_item_id,
                        content_index=0,
                        output_index=output_index,
                    )

            except Exception as e:
                logger.exception("Error processing agent stream event: %s", e)

        elif event[0] == "updates":
            for node_data in event[1].values():
                messages = node_data.get("messages", [])
                if not messages:
                    continue

                has_ai_message = False

                for j, msg in enumerate(messages):
                    if isinstance(msg, ToolMessage):
                        content = (
                            msg.content
                            if isinstance(msg.content, str)
                            else json.dumps(msg.content)
                        )
                        tool_outputs_for_fallback.append(content)
                        item = create_function_call_output_item(
                            call_id=msg.tool_call_id,
                            output=content,
                        )
                        yield ResponsesAgentStreamEvent(
                            type="response.output_item.done",
                            item=item,
                        )

                    elif hasattr(msg, "tool_calls") and msg.tool_calls:
                        has_ai_message = True
                        if not in_turn:
                            _start_turn()
                            yield ResponsesAgentStreamEvent(
                                type="response.created",
                                response=_response_obj(),
                            )

                        for k, tc in enumerate(msg.tool_calls):
                            call_id = tc.get("id", "")
                            name = tc.get("name", "")
                            args = tc.get("args", {})
                            args_str = (
                                json.dumps(args) if isinstance(args, dict) else str(args)
                            )
                            tc_info = active_tool_calls.get(k)
                            if tc_info:
                                item_id = tc_info["item_id"]
                                matched_oi = tc_info["output_index"]
                            else:
                                item_id = str(uuid_utils.uuid7())
                                matched_oi = output_index
                                output_index += 1

                            item = create_function_call_item(
                                id=item_id,
                                call_id=call_id,
                                name=name,
                                arguments=args_str,
                            )
                            turn_output_items.append(item)
                            yield ResponsesAgentStreamEvent(
                                type="response.output_item.done",
                                item=item,
                                output_index=matched_oi,
                            )

                        active_tool_calls.clear()

                    elif hasattr(msg, "content") and msg.content:
                        saw_text_output = True
                        has_ai_message = True
                        if not in_turn:
                            _start_turn()
                            yield ResponsesAgentStreamEvent(
                                type="response.created",
                                response=_response_obj(),
                            )

                        text = msg.content
                        item_id = active_text_item_id or str(uuid_utils.uuid7())

                        if not active_text_item_id:
                            yield ResponsesAgentStreamEvent(
                                type="response.output_item.added",
                                item={
                                    "type": "message",
                                    "id": item_id,
                                    "role": "assistant",
                                    "status": "in_progress",
                                    "content": [],
                                },
                                output_index=output_index,
                            )
                            yield ResponsesAgentStreamEvent(
                                type="response.content_part.added",
                                item_id=item_id,
                                output_index=output_index,
                                content_index=0,
                                part={"type": "output_text", "text": "", "annotations": []},
                            )

                        yield ResponsesAgentStreamEvent(
                            type="response.content_part.done",
                            item_id=item_id,
                            output_index=output_index,
                            content_index=0,
                            part={"type": "output_text", "text": text, "annotations": []},
                        )

                        item = create_text_output_item(text=text, id=item_id)
                        item["status"] = "completed"
                        turn_output_items.append(item)
                        yield ResponsesAgentStreamEvent(
                            type="response.output_item.done",
                            item=item,
                            output_index=output_index,
                        )
                        output_index += 1
                        active_text_item_id = None
                        active_text_content = ""

                if has_ai_message and in_turn:
                    yield ResponsesAgentStreamEvent(
                        type="response.completed",
                        response=_response_obj(turn_output_items),
                    )
                    _end_turn()

    if tool_outputs_for_fallback and not saw_text_output:
        if not in_turn:
            _start_turn()
            yield ResponsesAgentStreamEvent(
                type="response.created",
                response=_response_obj(),
            )

        item_id = str(uuid_utils.uuid7())
        text = _fallback_text()
        yield ResponsesAgentStreamEvent(
            type="response.output_item.added",
            item={
                "type": "message",
                "id": item_id,
                "role": "assistant",
                "status": "in_progress",
                "content": [],
            },
            output_index=output_index,
        )
        yield ResponsesAgentStreamEvent(
            type="response.content_part.done",
            item_id=item_id,
            output_index=output_index,
            content_index=0,
            part={"type": "output_text", "text": text, "annotations": []},
        )

        item = create_text_output_item(text=text, id=item_id)
        item["status"] = "completed"
        turn_output_items.append(item)
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=item,
            output_index=output_index,
        )
        yield ResponsesAgentStreamEvent(
            type="response.completed",
            response=_response_obj(turn_output_items),
        )
        _end_turn()
