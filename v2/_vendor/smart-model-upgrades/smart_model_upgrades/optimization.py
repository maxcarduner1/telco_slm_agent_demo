"""Agent-agnostic optimization core.

Public surface:
    smu.optimize_prompts_and_models(predict_fn, train_data, val_data, *,
                 prompt_uris=[], gateway_endpoints={},
                 scorers, max_metric_calls, ...)  -> Result
    smu.score(predict_fn, val_data, *, scorers)   -> float
    smu.promote_to_prod(result)
    smu.setup_endpoints(endpoints)

Everything else is internal (`_`-prefixed) and may move without warning.
"""

import json
import os
import re
import time
import warnings
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import gepa
import mlflow
from mlflow.entities import Feedback
from mlflow.genai.judges import CategoricalRating

from . import ai_gateway as gw

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")


# Default DBU rates per 1M tokens (Global / Short Context).
# Source: databricks.com/product/pricing/proprietary-foundation-model-serving
# Hidden behind the public API; will be replaced by gateway-provided cost data.
_DEFAULT_TOKEN_COSTS = {
    "databricks-claude-3-7-sonnet":     {"input": 42.857, "output": 214.286},
    "databricks-claude-sonnet-4":       {"input": 42.857, "output": 214.286},
    "databricks-claude-sonnet-4-6":     {"input": 42.857, "output": 214.286},
    "databricks-claude-opus-4-6":       {"input": 71.429, "output": 357.143},
    "databricks-claude-opus-4-7":       {"input": 71.429, "output": 357.143},
    "databricks-claude-haiku-4-5":      {"input": 14.286, "output": 71.429},
    "databricks-gpt-5-4":               {"input": 17.857, "output": 142.857},
    "databricks-gpt-5-4-mini":          {"input": 3.571,  "output": 28.571},
    "databricks-gpt-5-4-nano":          {"input": 0.714,  "output": 5.714},
    "databricks-gpt-5-nano":            {"input": 0.714,  "output": 5.714},
    "databricks-llama-4-maverick":      {"input": 1.429,  "output": 7.143},
    "databricks-gemini-3-1-flash-lite": {"input": 8.929,  "output": 53.571},
    "databricks-gemini-2-5-flash":      {"input": 8.929,  "output": 53.571},
}
_DBU_TO_USD = 0.07
_FALLBACK_TOKEN_RATE = {"input": 10.0, "output": 50.0}
_FALLBACK_INPUT_TOKENS = 500
_FALLBACK_OUTPUT_TOKENS = 200


# ---------------------------------------------------------------------------
# URI parsing + var extraction
# ---------------------------------------------------------------------------

_VAR_PATTERN = re.compile(r"\{\{\s*(\w+)\s*\}\}|\{(\w+)\}")


def _parse_prompt_uri(uri):
    """Parse 'prompts:/cat.schema.name@alias' or 'prompts:/cat.schema.name/version'.

    Returns (full_name, alias_or_none, version_or_none, short_name).
    """
    if not uri.startswith("prompts:/"):
        raise ValueError(f"Invalid prompt URI {uri!r} (must start with 'prompts:/')")
    rest = uri[len("prompts:/"):]
    if "@" in rest:
        name, alias = rest.rsplit("@", 1)
        version = None
    elif "/" in rest:
        name, version = rest.rsplit("/", 1)
        alias = None
    else:
        name, alias, version = rest, "production", None
    short_name = name.rsplit(".", 1)[-1]
    return name, alias, version, short_name


def _extract_required_vars(template):
    """Find every `{var}` or `{{ var }}` placeholder. Returns deduped list in order."""
    seen, out = set(), []
    for jinja, python in _VAR_PATTERN.findall(template):
        v = jinja or python
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

@dataclass
class _PromptTarget:
    uri: str
    name: str
    alias: Optional[str]
    version: Optional[str]
    short_name: str
    template: str
    required_vars: list
    prior_version: int


@dataclass
class _EndpointTarget:
    name: str
    candidate_models: list
    initial_model: str

    @property
    def exp_name(self):
        return f"{self.name}-exp"


@dataclass
class _State:
    predict_fn: Callable
    prompt_targets: List[_PromptTarget]
    endpoint_targets: List[_EndpointTarget]
    scorers: list
    weight_quality: float
    weight_latency: float
    weight_cost: float
    latency_hard_gate: float
    cost_soft_gate: float
    reflection_model: str
    token_costs: dict
    last_gateway_synced: dict = field(default_factory=dict)
    scorer_attempted: int = 0
    scorer_succeeded: int = 0


def _prompt_key(pt): return f"prompt:{pt.short_name}"
def _model_key(et):  return f"model:{et.name}"


# ---------------------------------------------------------------------------
# Public Result
# ---------------------------------------------------------------------------

@dataclass
class Result:
    """Output of `optimize_prompts_and_models`.

    `gepa_result` is non-picklable, but the rest of the object is -- if you
    need to defer promotion across kernels, drop `gepa_result` and pickle
    the rest, or just persist `best_candidate` + `prompt_uris` +
    `gateway_endpoints` + the seed `initial_model` per endpoint.

    Attributes:
        best_candidate: GEPA-flat dict, keys `prompt:<short>` / `model:<endpoint>`.
        best_score: mean composite score on val_data after optimization.
        baseline_score: mean composite score on val_data before optimization.
        prompt_uris: input prompt URIs (echoed for `promote_to_prod`).
        gateway_endpoints: input endpoint dict (echoed for `promote_to_prod`).
        prompt_targets: resolved prompt metadata used by `promote_to_prod`.
        endpoint_targets: resolved endpoint metadata used by `promote_to_prod`.
        gepa_result: raw `gepa.GEPAResult` (advanced/debug use, non-picklable).
    """
    best_candidate: dict
    best_score: float
    baseline_score: float
    prompt_uris: list
    gateway_endpoints: dict
    prompt_targets: List[_PromptTarget] = field(repr=False)
    endpoint_targets: List[_EndpointTarget] = field(repr=False)
    gepa_result: object = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# State construction
# ---------------------------------------------------------------------------

def _build_prompt_target(uri):
    name, alias, version, short = _parse_prompt_uri(uri)
    pv = mlflow.genai.load_prompt(uri)
    return _PromptTarget(
        uri=uri,
        name=name,
        alias=alias,
        version=version,
        short_name=short,
        template=pv.template,
        required_vars=_extract_required_vars(pv.template),
        prior_version=int(pv.version),
    )


def _read_endpoint_destination(ep_name):
    ep = gw.get_endpoint(ep_name)
    dests = ep.get("config", {}).get("destinations", [])
    if not dests:
        raise ValueError(f"Endpoint {ep_name!r} has no destinations")
    return dests[0]["name"].removeprefix("system.ai.")


def _build_endpoint_target(name, candidate_models):
    if not candidate_models:
        raise ValueError(f"Endpoint {name!r} has no candidate_models")
    return _EndpointTarget(
        name=name,
        candidate_models=list(candidate_models),
        initial_model=_read_endpoint_destination(name),
    )


def _build_state(predict_fn, prompt_uris, gateway_endpoints, scorers,
                 weight_quality, weight_latency, weight_cost,
                 latency_hard_gate, cost_soft_gate, reflection_model,
                 token_costs):
    if not prompt_uris and not gateway_endpoints:
        raise ValueError("Pass at least one of prompt_uris or gateway_endpoints")

    weight_sum = weight_quality + weight_latency + weight_cost
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(
            f"weights must sum to 1.0 (got quality={weight_quality} + "
            f"latency={weight_latency} + cost={weight_cost} = {weight_sum})"
        )
    for name, w in (("weight_quality", weight_quality),
                    ("weight_latency", weight_latency),
                    ("weight_cost", weight_cost)):
        if w < 0:
            raise ValueError(f"{name} must be >= 0 (got {w})")

    if cost_soft_gate <= 0:
        raise ValueError(f"cost_soft_gate must be > 0 (got {cost_soft_gate})")
    if latency_hard_gate <= 0:
        raise ValueError(f"latency_hard_gate must be > 0 (got {latency_hard_gate})")

    merged_costs = dict(_DEFAULT_TOKEN_COSTS)
    if token_costs:
        merged_costs.update(token_costs)

    prompt_targets = [_build_prompt_target(u) for u in prompt_uris]
    endpoint_targets = [
        _build_endpoint_target(name, candidates)
        for name, candidates in gateway_endpoints.items()
    ]

    if weight_cost > 0 and endpoint_targets:
        unknown = set()
        for et in endpoint_targets:
            for model in [et.initial_model, *et.candidate_models]:
                if model not in merged_costs:
                    unknown.add(model)
        if unknown:
            raise ValueError(
                f"weight_cost > 0 but no token cost data for {sorted(unknown)}. "
                f"Either pass token_costs={{<model>: {{'input': <DBU per 1M input tokens>, "
                f"'output': <DBU per 1M output tokens>}}, ...}}, or set weight_cost=0 "
                f"to disable the cost component. See databricks.com/product/pricing/"
                f"proprietary-foundation-model-serving for current rates."
            )

    counts = Counter(pt.short_name for pt in prompt_targets)
    dupes = [name for name, n in counts.items() if n > 1]
    if dupes:
        raise ValueError(
            f"prompt_uris share short names {sorted(dupes)}; rename or use distinct UC paths"
        )

    return _State(
        predict_fn=predict_fn,
        prompt_targets=prompt_targets,
        endpoint_targets=endpoint_targets,
        scorers=scorers,
        weight_quality=weight_quality,
        weight_latency=weight_latency,
        weight_cost=weight_cost,
        latency_hard_gate=latency_hard_gate,
        cost_soft_gate=cost_soft_gate,
        reflection_model=reflection_model,
        token_costs=merged_costs,
    )


def _seed_candidate(state):
    seed = {}
    for pt in state.prompt_targets:
        seed[_prompt_key(pt)] = pt.template
    for et in state.endpoint_targets:
        seed[_model_key(et)] = et.initial_model
    return seed


def _preflight(state, seed, train_data, val_data):
    """Smoke-test predict_fn once before gepa.optimize launches.

    Surfaces obvious failures (broken auth, missing endpoint, import errors)
    early but does NOT abort -- a per-input failure (recursion limit on a hard
    record, transient API error) shouldn't kill a run that the scorer-success
    ratchet would otherwise let GEPA work around. Prints a loud warning if
    the warmup raises and lets gepa.optimize proceed; the ratchet at the end
    catches the case where every iteration fails consistently.

    The patches (`_patched_endpoints` + `_patched_prompts`) must already be
    active when this is called.
    """
    sample = next(iter(train_data or val_data or []), None)
    if sample is None:
        return
    print("Running pre-flight smoke test on first record...")
    with _patched_prompts(seed, state.prompt_targets):
        _sync_destinations(seed, state, use_exp=True)
        try:
            state.predict_fn(sample["inputs"])
        except Exception as e:
            print(
                f"  WARN: predict_fn raised {type(e).__name__} on the first record: "
                f"{e}. Continuing -- GEPA will treat this as a low-quality "
                f"iteration. If every iteration fails the same way the scorer-"
                f"success ratchet will surface it at the end."
            )
            return
    print("  Pre-flight passed.")


# ---------------------------------------------------------------------------
# Exp endpoint lifecycle (internal)
# ---------------------------------------------------------------------------

def _ensure_exp_endpoints(state):
    for et in state.endpoint_targets:
        try:
            gw.get_endpoint(et.exp_name)
            print(f"  {et.exp_name}: already exists")
            continue
        except Exception:
            pass
        gw.create_endpoint(
            name=et.exp_name,
            destinations=[_ppt_destination(et.initial_model)],
            task_type="llm/v1/chat",
            tags=[
                gw.tag("managed_by", "smart-model-upgrades"),
                gw.tag("role", "experimental"),
            ],
        )
        print(f"  {et.exp_name}: created with {et.initial_model}")


def _cleanup_exp_endpoints(state):
    for et in state.endpoint_targets:
        try:
            gw.delete_endpoint(et.exp_name)
            print(f"  {et.exp_name}: deleted")
        except Exception as e:
            print(f"  {et.exp_name}: delete failed ({type(e).__name__}: {e}) -- "
                  f"clean up manually with `databricks api delete /api/2.0/ai-gateway/v2/endpoints/{et.exp_name}`")


_model_info_cache = {}

_PPT = "PAY_PER_TOKEN_FOUNDATION_MODEL"


def _ppt_destination(model):
    """Build a 100% PAY_PER_TOKEN destination for `system.ai.<model>`."""
    return gw.destination(_resolve_system_ai_name(model), _PPT, 100)


def _resolve_model_info(fmapi_endpoint):
    """Return {name, display_name, description} for an FMAPI endpoint. Cached.

    Falls back to {name: f"system.ai.{fmapi_endpoint}", display_name: <endpoint>}
    when the endpoint metadata is missing or unreachable -- the gateway sync
    still works since `system.ai.<endpoint>` is the conventional destination.
    """
    if fmapi_endpoint in _model_info_cache:
        return _model_info_cache[fmapi_endpoint]
    fallback = {
        "name": f"system.ai.{fmapi_endpoint}",
        "display_name": fmapi_endpoint,
        "description": "",
    }
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        ep = w.serving_endpoints.get(fmapi_endpoint)
        if ep.config is None or not ep.config.served_entities:
            info = fallback
        else:
            fm = ep.config.served_entities[0].foundation_model
            info = {
                "name": fm.name or fallback["name"],
                "display_name": fm.display_name or fmapi_endpoint,
                "description": fm.description or "",
            }
    except Exception:
        info = fallback
    _model_info_cache[fmapi_endpoint] = info
    return info


def _resolve_system_ai_name(fmapi_endpoint):
    return _resolve_model_info(fmapi_endpoint)["name"]


def _sync_destinations(candidate, state, *, use_exp):
    """Push candidate model choices to gateway endpoints.

    Skips endpoints already at the requested model via `state.last_gateway_synced`,
    which is per-_State so consecutive optimize calls in the same kernel don't
    skip a needed update because of stale cross-call state.
    """
    cache = state.last_gateway_synced
    updated = False
    for et in state.endpoint_targets:
        ep_name = et.exp_name if use_exp else et.name
        model = candidate.get(_model_key(et))
        if model is None:
            continue
        if model != cache.get(ep_name):
            gw.update_endpoint(ep_name, destinations=[_ppt_destination(model)])
            cache[ep_name] = model
            print(f"  {ep_name} -> {model}")
            updated = True
    if not updated and not use_exp:
        print("  All endpoints already up to date.")


# ---------------------------------------------------------------------------
# Patched prompts (internal)
# ---------------------------------------------------------------------------

@contextmanager
def _patched_prompts(candidate, prompt_targets):
    """Inject GEPA candidate prompts by patching `PromptVersion.template`.

    Why this works without the customer's predict_fn reloading prompts:

    `PromptVersion.template` is a `@property` -- reading `pv.template` invokes
    the getter every access, there's no cached instance attribute. We replace
    the property descriptor at the class level for the duration of the block,
    so every `.template` access during one evaluation goes through our patched
    getter. That getter checks if the prompt's full UC name matches one of the
    prompts being optimized -- if so, returns the GEPA candidate string from
    `overrides`; otherwise falls through to the original `@production` lookup.

    The customer's predict_fn can therefore do either:

        # fresh load per call
        pv = mlflow.genai.load_prompt("prompts:/cat.schema.foo@production")
        return client.create(messages=[{"role": "user", "content": pv.format(**x)}])

        # cached at module level (more common)
        _pv = mlflow.genai.load_prompt("prompts:/cat.schema.foo@production")
        ... pv.format(**x) ...

    Both call `pv.format(...)` which internally reads `self.template` -- and
    that read goes through the patched class-level descriptor regardless of
    when the `PromptVersion` instance was constructed. No live-reload needed.

    Contract violation: caching the *rendered string itself* (`tmpl_str =
    pv.template` once, then `tmpl_str.format(...)` forever) skips the property
    descriptor on subsequent accesses, so the patch never fires. The agent
    would silently optimize against the seed prompt the entire run. Rare in
    practice -- `pv.format(...)` is the idiomatic MLflow API.

    The original descriptor is restored in `finally` even on exception, so
    two optimizations in the same kernel don't cross-contaminate.
    """
    from mlflow.entities.model_registry.prompt_version import PromptVersion
    original = PromptVersion.template
    original_getter = original.fget
    overrides = {
        pt.name: candidate[_prompt_key(pt)]
        for pt in prompt_targets
        if _prompt_key(pt) in candidate
    }

    @property
    def _patched(self):
        if self.name in overrides:
            return overrides[self.name]
        return original_getter(self)

    PromptVersion.template = _patched
    try:
        yield
    finally:
        PromptVersion.template = original


# ---------------------------------------------------------------------------
# Transparent endpoint routing (internal)
# ---------------------------------------------------------------------------

def _patch_create(import_path, cls_name, is_async, rewrites):
    """Monkeypatch `<cls>.create` to rewrite `model=<endpoint>` to `<endpoint>-exp`.

    Returns `(cls, "create", original)` for the caller to restore on exit, or
    `None` if the import isn't available (e.g. an older `openai` SDK without the
    Responses API).
    """
    import importlib
    try:
        module = importlib.import_module(import_path)
        cls = getattr(module, cls_name)
    except (ImportError, AttributeError):
        return None
    original = cls.create

    if is_async:
        async def patched(self, *args, **kwargs):
            model = kwargs.get("model")
            if model in rewrites:
                kwargs["model"] = rewrites[model]
            return await original(self, *args, **kwargs)
    else:
        def patched(self, *args, **kwargs):
            model = kwargs.get("model")
            if model in rewrites:
                kwargs["model"] = rewrites[model]
            return original(self, *args, **kwargs)

    cls.create = patched
    return (cls, "create", original)


@contextmanager
def _patched_endpoints(endpoint_targets):
    """Rewrite `model=<prod_endpoint>` to `<prod_endpoint>-exp` on outbound LLM calls.

    Patches the OpenAI Python client's chat.completions and responses APIs (sync
    + async) so any agent routing through OpenAI / databricks_openai /
    langchain-openai / dspy-openai transparently hits the experimental clones
    during optimization. Customer's agent code keeps using the prod endpoint
    name and is not touched.

    Agents that bypass the OpenAI client (raw requests/httpx to the gateway) are
    not covered; document and add a separate intercept if it comes up.
    """
    if not endpoint_targets:
        yield
        return

    rewrites = {et.name: et.exp_name for et in endpoint_targets}
    targets = [
        ("openai.resources.chat.completions", "Completions", False),
        ("openai.resources.chat.completions", "AsyncCompletions", True),
        ("openai.resources.responses", "Responses", False),
        ("openai.resources.responses", "AsyncResponses", True),
    ]
    patches = [
        p for (path, cls_name, is_async) in targets
        if (p := _patch_create(path, cls_name, is_async, rewrites)) is not None
    ]

    try:
        yield
    finally:
        for cls, attr, original in patches:
            setattr(cls, attr, original)


# ---------------------------------------------------------------------------
# Trace summary (internal, simplified)
# ---------------------------------------------------------------------------

_TOKEN_USAGE_KEY = "mlflow.chat.tokenUsage"


def _read_span_tokens(span):
    usage = None
    if hasattr(span, "get_attribute"):
        try:
            usage = span.get_attribute(_TOKEN_USAGE_KEY)
        except Exception:
            usage = None
    if usage is None:
        attrs = getattr(span, "attributes", None) or {}
        usage = attrs.get(_TOKEN_USAGE_KEY)
    if isinstance(usage, str):
        try:
            usage = json.loads(usage)
        except Exception:
            usage = None
    if not isinstance(usage, dict):
        return 0, 0
    try:
        return int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)
    except (TypeError, ValueError):
        return 0, 0


def _get_active_trace():
    """Return the latest active MLflow Trace object, or None if no trace exists.

    Used by `_run_scorers` so MLflow `Scorer` objects (e.g. `make_judge` judges
    with `Trace` template fields) get the per-component span data they need;
    smu would otherwise pass only the final answer.
    """
    try:
        trace_id = mlflow.get_last_active_trace_id()
        if not trace_id:
            return None
        return mlflow.get_trace(trace_id)
    except Exception:
        return None


def _extract_trace_summary():
    """Pull total tokens + per-span debug info from the latest active trace.

    Returns {} if no trace is active. We no longer try to attribute spans to
    specific endpoints -- caller uses the totals + per-endpoint defaults.
    """
    try:
        trace = _get_active_trace()
        if not trace or not trace.data:
            return {}
        total_in, total_out = 0, 0
        spans = []
        for span in trace.data.spans:
            in_t, out_t = _read_span_tokens(span)
            total_in += in_t
            total_out += out_t
            dur = 0.0
            if hasattr(span, "end_time_ns") and hasattr(span, "start_time_ns"):
                if span.end_time_ns and span.start_time_ns:
                    dur = (span.end_time_ns - span.start_time_ns) / 1e9
            spans.append({
                "name": span.name or "",
                "type": span.span_type,
                "duration_s": round(dur, 2),
                "input": str(span.inputs or "")[:300],
                "output": str(span.outputs or "")[:500],
            })
        return {"total_tokens": {"input": total_in, "output": total_out}, "spans": spans}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Scoring (internal)
# ---------------------------------------------------------------------------

def _convert_to_numeric(score):
    if isinstance(score, Feedback):
        score = score.value
    if score == CategoricalRating.YES:
        return 1.0
    if score == CategoricalRating.NO:
        return 0.0
    if isinstance(score, (int, float, bool)):
        return float(score)
    return None


_SCORER_WARNINGS_SEEN = set()


def _scorer_name(s):
    return getattr(s, "name", None) or getattr(s, "__name__", None) or type(s).__name__


def _warn_once(scorer_name, kind, detail):
    """Emit a UserWarning once per (scorer_name, kind) for the lifetime of the process.

    Cross-call dedup is intentional -- same scorer error in score() then optimize()
    shouldn't fire twice. The set is module-global by design.
    """
    key = (scorer_name, kind)
    if key in _SCORER_WARNINGS_SEEN:
        return
    _SCORER_WARNINGS_SEEN.add(key)
    warnings.warn(f"[scorer:{scorer_name}] {kind}: {detail}", stacklevel=2)


def _run_scorers(scorers, inputs, expectations, answer, *, trace=None, state=None):
    """Run `scorers` and average their numeric outputs.

    MLflow `Scorer` objects get the structured kwargs they expect (`inputs`,
    `outputs`, `expectations`, `trace`) directly. The `trace` arg lets
    trace-aware judges (e.g. `make_judge` with a `Trace` template field) read
    per-component span data; without it they would see only the final answer.
    Plain callables are invoked as `(inputs, expectations, answer)` -- the
    trace is intentionally not threaded through, since plain scorers have no
    standard way to consume it. Failures and non-numeric returns are
    warned-once per (scorer, error-kind) so silent zero-quality runs are
    diagnosable.

    If `state` is passed, increments `state.scorer_attempted` / `_succeeded`
    so optimize_prompts_and_models can ratchet on a low success rate.
    """
    from mlflow.genai.scorers.base import Scorer as MlflowScorer
    numeric = []
    if scorers and state is not None:
        state.scorer_attempted += 1
    for s in scorers:
        name = _scorer_name(s)
        try:
            if isinstance(s, MlflowScorer):
                result = s.run(
                    inputs=inputs,
                    outputs=answer,
                    expectations=expectations,
                    trace=trace,
                )
            else:
                result = s(inputs, expectations, answer)
        except Exception as e:
            _warn_once(name, "raised", f"{type(e).__name__}: {e}")
            continue
        val = _convert_to_numeric(result)
        if val is None:
            _warn_once(name, "returned non-numeric", repr(result))
            continue
        numeric.append(val)
    if numeric:
        if state is not None:
            state.scorer_succeeded += 1
        return sum(numeric) / len(numeric)
    return 0.0


def _estimate_cost_usd(candidate, endpoint_targets, total_tokens, token_costs):
    """Rough $-per-call estimate using state-resolved rates and a 500/200 fallback.

    Real per-endpoint token attribution will land when gateway exposes it.
    """
    if not endpoint_targets:
        return 0.0
    n = len(endpoint_targets)
    total_in = total_tokens.get("input") or 0
    total_out = total_tokens.get("output") or 0
    if not total_in and not total_out:
        in_per, out_per = _FALLBACK_INPUT_TOKENS, _FALLBACK_OUTPUT_TOKENS
    else:
        in_per, out_per = total_in / n, total_out / n
    total = 0.0
    for et in endpoint_targets:
        model = candidate.get(_model_key(et), et.initial_model)
        rate = token_costs.get(model, _FALLBACK_TOKEN_RATE)
        dbu = (in_per * rate["input"] + out_per * rate["output"]) / 1_000_000
        total += dbu * _DBU_TO_USD
    return total


# ---------------------------------------------------------------------------
# Adapter (internal)
# ---------------------------------------------------------------------------

def _format_full_trace(spans):
    if not spans:
        return "(no spans captured)"
    return "\n".join(
        f"[{s['name']}] ({s.get('duration_s', 0)}s)\n  input: {s.get('input', '')}\n  output: {s.get('output', '')}"
        for s in spans
    )


class _AgentAdapter(gepa.GEPAAdapter):
    """Bridges GEPA with any predict_fn via the state's exp endpoints."""

    def __init__(self, state):
        self.state = state
        self._model_history = {et.name: defaultdict(list) for et in state.endpoint_targets}

    def _validate(self, candidate):
        for pt in self.state.prompt_targets:
            txt = candidate.get(_prompt_key(pt), "")
            for var in pt.required_vars:
                if var not in txt:
                    return False, f"prompt:{pt.short_name} missing var {{{var}}}"
        for et in self.state.endpoint_targets:
            model = candidate.get(_model_key(et), "")
            allowed = set(et.candidate_models) | {et.initial_model}
            if model not in allowed:
                return False, f"model:{et.name} '{model[:80]}' not in candidate list"
        return True, ""

    def _run_one(self, candidate, inputs, expectations):
        """Score a single record. Caller owns prompt-patching + endpoint sync."""
        state = self.state
        start = time.perf_counter()
        try:
            answer = state.predict_fn(inputs)
        except Exception as e:
            answer = f"ERROR: {e}"
        latency = time.perf_counter() - start
        mlflow_trace = _get_active_trace()
        trace = _extract_trace_summary()

        if latency > state.latency_hard_gate:
            return 0.0, {"quality": 0.0, "latency": 0.0, "cost": 0.0}, answer, \
                   f"REJECTED: latency {latency:.1f}s", trace

        quality = _run_scorers(
            state.scorers, inputs, expectations, answer,
            trace=mlflow_trace, state=state,
        )
        lat_score = max(0.0, 1.0 - latency / state.latency_hard_gate)

        total_in = trace.get("total_tokens", {}).get("input") or 0
        if state.weight_cost > 0 and not total_in:
            _warn_once(
                "tracing", "no_token_usage",
                "no token usage in active trace; cost component is using a "
                "500/200-token fallback x per-model rates, which means cost "
                "score depends only on model choice. Enable "
                "`mlflow.<framework>.autolog()` at agent import, or pass "
                "`weight_cost=0` to disable the cost component.",
            )
        cost_usd = _estimate_cost_usd(candidate, state.endpoint_targets,
                                      trace.get("total_tokens", {}),
                                      state.token_costs)
        cost_score = max(0.0, 1.0 - cost_usd / state.cost_soft_gate)

        score = (state.weight_quality * quality
                 + state.weight_latency * lat_score
                 + state.weight_cost * cost_score)
        objectives = {"quality": quality, "latency": lat_score, "cost": cost_score}
        feedback = (
            f"quality={quality:.2f} latency={latency:.1f}s "
            f"(score={lat_score:.2f}) cost=${cost_usd:.4f}"
        )

        for et in state.endpoint_targets:
            m = candidate.get(_model_key(et), "")
            if m:
                self._model_history[et.name][m].append(score)

        return score, objectives, answer, feedback, trace

    def evaluate(self, batch, candidate, capture_traces=False):
        ok, msg = self._validate(candidate)
        if not ok:
            zero_obj = {"quality": 0.0, "latency": 0.0, "cost": 0.0}
            return gepa.EvaluationBatch(
                outputs=[""] * len(batch),
                scores=[0.0] * len(batch),
                trajectories=None,
                objective_scores=[zero_obj] * len(batch),
            )

        outputs, scores, trajectories, all_obj = [], [], [], []
        with _patched_prompts(candidate, self.state.prompt_targets):
            _sync_destinations(candidate, self.state, use_exp=True)
            for record in batch:
                score, obj, answer, feedback, trace = self._run_one(
                    candidate, record["inputs"], record["expectations"],
                )
                scores.append(score)
                outputs.append(answer)
                all_obj.append(obj)
                if capture_traces:
                    trajectories.append({
                        "inputs": record["inputs"],
                        "outputs": answer,
                        "expectations": record["expectations"],
                        "score": score,
                        "feedback": feedback,
                        "trace": trace,
                    })
        return gepa.EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
            objective_scores=all_obj,
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        datasets = {}
        for key in components_to_update:
            history_str = self._history_str(key)
            comp_data = []
            for traj, score in zip(eval_batch.trajectories or [], eval_batch.scores):
                trace = traj.get("trace", {})
                record = {
                    "component_name": key,
                    "current_text": candidate.get(key, ""),
                    "Inputs": traj.get("inputs", {}),
                    "Full Agent Answer": traj.get("outputs", ""),
                    "Expected Answer": traj.get("expectations", ""),
                    "Score": f"{score:.2f}",
                    "Feedback": traj.get("feedback", ""),
                    "Full Trace": _format_full_trace(trace.get("spans", [])),
                }
                if key.startswith("model:"):
                    record["Models Tried (aggregate across all iterations)"] = history_str
                comp_data.append(record)
            datasets[key] = comp_data
        return datasets

    def _history_str(self, key):
        if not key.startswith("model:"):
            return ""
        ep_name = key[len("model:"):]
        history = self._model_history.get(ep_name, {})
        et = next((e for e in self.state.endpoint_targets if e.name == ep_name), None)
        if et is None:
            return ""
        lines = []
        for model, scores in sorted(history.items()):
            mean = sum(scores) / len(scores)
            lines.append(
                f"- {model}: mean={mean:.3f} range=[{min(scores):.3f}, {max(scores):.3f}] trials={len(scores)}"
            )
        for model in et.candidate_models:
            if model not in history:
                lines.append(f"- {model}: not yet evaluated")
        return "\n".join(lines) if lines else "(none yet)"


# ---------------------------------------------------------------------------
# Reflection templates (internal)
# ---------------------------------------------------------------------------

_MODEL_SELECTION_TEMPLATE = """The current model endpoint is:
```
<curr_param>
```

Evaluation results with this model:
```
<side_info>
```

Select the best model endpoint from this list:
{candidates}

Rules:
- You MUST pick one of the exact names listed above. Do NOT invent names.
- Always try a new model.
- Prefer cheaper/faster models when quality is similar.

Provide your chosen model name within ``` blocks."""


_PROMPT_REFLECTION_TEMPLATE = """The current prompt is:
```
<curr_param>
```

Evaluation results with this prompt:
```
<side_info>
```

Write an improved version of this prompt.

Hard requirements:
- The new prompt MUST contain these placeholder variables: {required_vars}
- Preserve the exact placeholder syntax (including braces) as it appears in the current prompt. If the current prompt uses `{{{{ var }}}}`, keep that form; if it uses `{{var}}`, keep that form. Do not change brace style.
- Do NOT output meta-instructions about "how to write a prompt". Output the prompt itself, as it will be used directly by the agent at runtime.
- Do NOT add commentary, explanation, or headers outside the prompt.

Return the new prompt within ``` blocks."""


def _build_reflection_templates(state):
    out = {}
    for pt in state.prompt_targets:
        out[_prompt_key(pt)] = _PROMPT_REFLECTION_TEMPLATE.format(
            required_vars=", ".join(f"`{{{v}}}`" for v in pt.required_vars),
        )
    for et in state.endpoint_targets:
        lines = []
        for endpoint in et.candidate_models:
            try:
                info = _resolve_model_info(endpoint)
                display = info.get("display_name") or info.get("name") or endpoint
            except Exception:
                display = endpoint
            lines.append(f"- {endpoint} ({display})")
        out[_model_key(et)] = _MODEL_SELECTION_TEMPLATE.format(
            candidates="\n".join(lines),
        )
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_endpoints(endpoints, *, agent_tag=None, extra_tags=None):
    """Create or sync production gateway endpoints.

    Idempotent: existing endpoints with a matching destination are left alone;
    mismatched destinations are updated to `system.ai.<initial_model>`. Use
    this once per agent before the first `optimize_prompts_and_models` call.

    Endpoint names cannot start with `databricks-` (the gateway reserves that
    prefix for system endpoints) -- pick an agent-scoped namespace like
    `wanderbricks-supervisor` or `toy-translator-endpoint`.

    Args:
        endpoints: dict mapping `endpoint_name -> initial_model`. Same shape
            convention as `optimize_prompts_and_models`'s `gateway_endpoints`.
        agent_tag: optional value for the `agent` tag.
        extra_tags: optional list of `(key, value)` tuples for additional tags.

    Returns:
        dict {endpoint_name: status} where status is "created", "exists:<dest>",
        or "updated:<old>-><new>".
    """
    tags = [gw.tag("managed_by", "smart-model-upgrades")]
    if agent_tag:
        tags.append(gw.tag("agent", agent_tag))
    if extra_tags:
        tags.extend(gw.tag(k, v) for k, v in extra_tags)

    statuses = {}
    for name, initial_model in endpoints.items():
        dest = _ppt_destination(initial_model)
        target = dest["name"]
        try:
            existing = gw.get_endpoint(name)
        except Exception:
            existing = None

        if existing is not None:
            dests = existing.get("config", {}).get("destinations", [])
            current = dests[0]["name"] if dests else "unknown"
            if current == target:
                statuses[name] = f"exists:{current}"
                print(f"  {name}: already exists ({current})")
                continue
            gw.update_endpoint(name, destinations=[dest])
            statuses[name] = f"updated:{current}->{target}"
            print(f"  {name}: updated {current} -> {target}")
            continue

        gw.create_endpoint(
            name=name,
            destinations=[dest],
            task_type="llm/v1/chat",
            tags=tags,
        )
        statuses[name] = "created"
        print(f"  {name}: created with {target}")
    return statuses


def score(predict_fn, val_data, *, scorers):
    """Run `predict_fn` over `val_data` and average `scorers` outputs.

    A thin convenience -- equivalent to a small `mlflow.genai.evaluate` loop.

    Args:
        predict_fn: agent's `predict(inputs: dict)` callable. The return value
            is passed straight to scorers; it can be a string, dict, structured
            response object, or anything else your scorers know how to handle.
        val_data: list of `{"inputs": dict, "expectations": dict}` following
            the MLflow evaluation schema. `expectations` carries the ground
            truth -- typical reserved keys are `expected_response` (used by
            `Correctness`), `expected_facts`, and `guidelines`.
        scorers: required keyword arg. MLflow `Scorer` objects or plain
            `(inputs, expected, answer) -> float` callables.

    Returns:
        Mean scorer score across val_data, as a float.
    """
    scores = []
    for record in val_data:
        try:
            answer = predict_fn(record["inputs"])
        except Exception as e:
            answer = f"ERROR: {e}"
        mlflow_trace = _get_active_trace()
        scores.append(
            _run_scorers(
                scorers, record["inputs"], record["expectations"], answer,
                trace=mlflow_trace,
            )
        )
    return sum(scores) / len(scores) if scores else 0.0


def optimize_prompts_and_models(
    predict_fn,
    train_data,
    val_data,
    *,
    prompt_uris: Optional[List[str]] = None,
    gateway_endpoints: Optional[dict] = None,
    scorers,
    max_metric_calls,
    weight_quality: float = 0.7,
    weight_latency: float = 0.2,
    weight_cost: float = 0.1,
    latency_hard_gate: float = 60.0,
    cost_soft_gate: float = 0.02,
    token_costs: Optional[dict] = None,
    reflection_model: str = "databricks-claude-sonnet-4-6",
    reflection_minibatch_size: int = 5,
    frontier_type: str = "hybrid",
    use_mlflow: bool = True,
    display_progress_bar: bool = True,
    **gepa_kwargs,
) -> Result:
    """Optimize prompts and/or model choices for an agent via GEPA.

    Mirrors `mlflow.genai.optimize_prompts()`'s shape: pass a `predict_fn`,
    train/val data, a list of `prompt_uris` to tune, and a dict of
    `gateway_endpoints` whose models to swap. Either may be empty (but
    at least one must be non-empty).

    Preconditions:
      - Every endpoint in `gateway_endpoints` must already exist on AI
        Gateway with at least one destination. Run `smu.setup_endpoints(...)`
        once per agent before the first call here, or stand the endpoints
        up by hand. Missing endpoints raise during state construction.
      - The agent must call `mlflow.<framework>.autolog()` at import for
        the cost component to use real token usage; without it, cost
        degenerates to a per-model constant via the 500/200 fallback and
        the library emits a warn-once.
      - At least 10% of evaluations must produce a numeric scorer score;
        below that the call raises at the end so a misconfigured scorer
        doesn't silently report `delta=0`.

    Internally this:
      1. Loads each prompt URI's current `@production` template and extracts
         its required template variables.
      2. Reads each gateway endpoint's current destination as the seed model.
      3. Creates `<endpoint>-exp` clones of every gateway endpoint and runs
         optimization against them. The exp endpoints are deleted on exit
         (success or failure).
      4. Pre-flight: runs `predict_fn` once on the first record with the seed
         candidate to catch obvious failures (broken agent, auth, missing
         endpoint) before burning the metric budget.
      5. Scores the seed candidate, runs `gepa.optimize`, and scores the
         winner. Returns both as `Result.baseline_score` / `.best_score`.

    The library transparently rewrites two things during each evaluation, so
    the agent code is not touched:

    - **Prompts**: `PromptVersion.template` is patched at the class level so
      any `pv.format(...)` / `pv.template` access during the eval returns the
      GEPA candidate string instead of the registry's `@production` template.
      Works whether the agent loads prompts fresh per call or caches the
      `PromptVersion` instance at module import (see `_patched_prompts`).
    - **Endpoints**: the OpenAI client's `model=<endpoint>` arg is rewritten
      to `<endpoint>-exp` so calls land on the experimental clones rather than
      prod. Covers `chat.completions.create` and `responses.create`, sync +
      async (see `_patched_endpoints`).

    Args:
        predict_fn: agent's `predict(inputs: dict)` callable. The return value
            is passed straight to scorers; it can be a string, dict, structured
            response object, or anything else your scorers know how to handle.
        train_data, val_data: lists of `{"inputs": dict, "expectations": dict}`
            following the MLflow evaluation schema. Scorers consume
            `expectations` directly; reserved keys include `expected_response`,
            `expected_facts`, and `guidelines`.
        prompt_uris: list of `prompts:/cat.schema.name@alias` strings. Default `[]`.
        gateway_endpoints: dict mapping `endpoint_name -> [candidate_models]`.
            Default `{}`. Pass only the endpoints whose models you want tuned.
        scorers: required. MLflow `Scorer` objects or `(inputs, expected, answer)
            -> float` callables.
        max_metric_calls: required. GEPA budget for metric evaluations. Rule
            of thumb: 4-8 x len(train_data) for an exploratory run.
        weight_quality, weight_latency, weight_cost: composite-score weights.
            Must be >= 0 and sum to 1.0.
        latency_hard_gate: candidates with total latency over this (seconds)
            score 0 on the latency component. Linear from 1.0 at 0s to 0.0
            at the gate. Must be > 0. Default 60.0.
        cost_soft_gate: candidates costing more than this (USD/call) score 0
            on the cost component. Linear from 1.0 at $0 to 0.0 at the gate.
            Must be > 0. Default 0.02 -- raise this for agents that make many
            LLM calls per request, otherwise the cost component saturates.
        token_costs: optional dict of `{model: {"input": <DBU per 1M input
            tokens>, "output": <DBU per 1M output tokens>}}` overrides /
            additions to the built-in cost table. Required (for the relevant
            models) when `weight_cost > 0` and a candidate model isn't in the
            built-in table; otherwise raises with the expected shape.
        reflection_model: model used by GEPA's reflection LM.
        reflection_minibatch_size, frontier_type, use_mlflow, display_progress_bar:
            forwarded to `gepa.optimize`.
        **gepa_kwargs: any other `gepa.optimize` kwarg (overrides the above).

    Returns:
        `Result` with `best_candidate`, `best_score`, `baseline_score`, plus
        the inputs needed by `promote_to_prod`.
    """
    state = _build_state(
        predict_fn,
        list(prompt_uris or []),
        dict(gateway_endpoints or {}),
        scorers,
        weight_quality, weight_latency, weight_cost,
        latency_hard_gate, cost_soft_gate, reflection_model,
        token_costs,
    )

    print("Creating experimental gateway endpoints...")
    _ensure_exp_endpoints(state)
    try:
        seed = _seed_candidate(state)
        adapter = _AgentAdapter(state)
        templates = _build_reflection_templates(state)
        defaults = {
            "seed_candidate": seed,
            "trainset": train_data,
            "valset": val_data,
            "adapter": adapter,
            "reflection_lm": f"databricks/{state.reflection_model}",
            "reflection_prompt_template": templates,
            "reflection_minibatch_size": reflection_minibatch_size,
            "frontier_type": frontier_type,
            "max_metric_calls": max_metric_calls,
            "display_progress_bar": display_progress_bar,
            "use_mlflow": use_mlflow,
        }
        defaults.update(gepa_kwargs)
        with _patched_endpoints(state.endpoint_targets):
            _preflight(state, seed, train_data, val_data)
            gepa_result = gepa.optimize(**defaults)

        # GEPA already evaluates the seed (index 0) and tracks the best on val,
        # so we read its scores directly instead of re-evaluating.
        baseline_score = gepa_result.val_aggregate_scores[0]
        best_score = gepa_result.val_aggregate_scores[gepa_result.best_idx]
        print(
            f"baseline {baseline_score:.3f} -> best {best_score:.3f} "
            f"(delta {best_score - baseline_score:+.3f})"
        )

        attempted, succeeded = state.scorer_attempted, state.scorer_succeeded
        if scorers and attempted > 0 and (succeeded / attempted) < 0.10:
            raise RuntimeError(
                f"Only {succeeded}/{attempted} ({succeeded / attempted:.0%}) "
                f"evaluations produced a numeric scorer score. The quality "
                f"component is effectively zero across the run, so the "
                f"reported delta reflects only latency/cost movement. Check "
                f"the warnings above for the underlying scorer error (a "
                f"common cause is a malformed scorer URI like "
                f"'databricks/<model>' instead of 'databricks:/<model>')."
            )

        return Result(
            best_candidate=gepa_result.best_candidate,
            best_score=best_score,
            baseline_score=baseline_score,
            prompt_uris=list(prompt_uris or []),
            gateway_endpoints=dict(gateway_endpoints or {}),
            prompt_targets=state.prompt_targets,
            endpoint_targets=state.endpoint_targets,
            gepa_result=gepa_result,
        )
    finally:
        print("Cleaning up experimental endpoints...")
        _cleanup_exp_endpoints(state)


def promote_to_prod(result: Result, *, rollback_alias="production_previous", dry_run=False):
    """Apply the winner to production.

    For each gateway endpoint whose chosen model differs from its seed,
    repoints the *production* endpoint to `system.ai.<new_model>`.

    For each prompt URI whose template was rewritten, aliases the prior
    production version to `@production_previous`, registers the new template
    as a new version, and re-points the URI's alias (typically
    `@production`) to the new version. Unchanged prompts are left alone.

    Endpoint and prompt updates are applied with rollback semantics: all
    destinations are pre-resolved before any update fires, and if any
    endpoint update or prompt registration step fails the library attempts
    to restore the already-mutated endpoints to their seed destinations and
    re-point any rewritten `@<alias>` prompts back to their prior version
    before re-raising. Note: `register_prompt` versions are immutable in
    MLflow, so rolled-back prompts may leave an unused new version in the
    registry -- the alias rollback is what matters for runtime behavior.

    Args:
        result: `Result` from `optimize_prompts_and_models`.
        rollback_alias: alias for the prior production version of any rewritten
            prompt. Default "production_previous". Underscores only -- MLflow
            rejects hyphens in alias names.
        dry_run: print the planned changes without applying them. Default False.

    Returns:
        dict {prompt_name: PromptVersion or None, endpoint_name: status_str}.
    """
    candidate = result.best_candidate
    out = {}

    endpoint_changes = []
    for et in result.endpoint_targets:
        new_model = candidate.get(_model_key(et))
        if not new_model or new_model == et.initial_model:
            continue
        try:
            new_dest = _ppt_destination(new_model)
        except Exception as e:
            raise RuntimeError(
                f"Cannot resolve destination for model {new_model!r} on endpoint "
                f"{et.name!r}: {type(e).__name__}: {e}. No endpoint updates applied."
            ) from e
        endpoint_changes.append((et, new_model, new_dest))

    prompt_changes = []
    for pt in result.prompt_targets:
        new_template = candidate.get(_prompt_key(pt))
        if new_template is None or new_template == pt.template:
            continue
        prompt_changes.append((pt, new_template))

    print("Planned changes:")
    if not endpoint_changes and not prompt_changes:
        print("  (none -- best candidate matches the seed)")
    for et, new_model, _ in endpoint_changes:
        print(f"  endpoint {et.name}: {et.initial_model} -> {new_model}")
    for pt, _ in prompt_changes:
        print(f"  prompt {pt.name}@{pt.alias}: new version (rollback to v{pt.prior_version} via @{rollback_alias})")
    for et in result.endpoint_targets:
        if not any(c[0] is et for c in endpoint_changes):
            print(f"  endpoint {et.name}: unchanged ({et.initial_model})")
            out[et.name] = f"unchanged:{et.initial_model}"
    for pt in result.prompt_targets:
        if not any(c[0] is pt for c in prompt_changes):
            print(f"  prompt {pt.name}@{pt.alias}: unchanged")
            out[pt.name] = None

    if dry_run:
        print("dry_run=True -- nothing applied.")
        for et, new_model, _ in endpoint_changes:
            out[et.name] = f"would-update:{et.initial_model}->{new_model}"
        for pt, _ in prompt_changes:
            out[pt.name] = "would-register"
        return out

    applied_endpoints = []
    repointed_prompts = []

    def _rollback(reason):
        if applied_endpoints:
            print(f"Rolling back {len(applied_endpoints)} endpoint(s) ({reason})...")
            for et in applied_endpoints:
                try:
                    gw.update_endpoint(et.name, destinations=[_ppt_destination(et.initial_model)])
                    print(f"  {et.name}: rolled back to {et.initial_model}")
                except Exception as rb:
                    print(f"  {et.name}: ROLLBACK FAILED ({rb}) -- manual intervention required")
        if repointed_prompts:
            print(f"Rolling back {len(repointed_prompts)} prompt alias(es) ({reason})...")
            for pt in repointed_prompts:
                try:
                    mlflow.genai.set_prompt_alias(
                        name=pt.name, alias=pt.alias, version=pt.prior_version,
                    )
                    print(f"  {pt.name}@{pt.alias}: rolled back to v{pt.prior_version}")
                except Exception as rb:
                    print(f"  {pt.name}@{pt.alias}: ROLLBACK FAILED ({rb}) -- manual intervention required")

    print("Updating production gateway endpoints...")
    try:
        for et, new_model, new_dest in endpoint_changes:
            gw.update_endpoint(et.name, destinations=[new_dest])
            applied_endpoints.append(et)
            out[et.name] = f"updated:{et.initial_model}->{new_model}"
            print(f"  {et.name}: {et.initial_model} -> {new_model}")
    except Exception as e:
        print(f"Endpoint update failed: {e}")
        _rollback("endpoint update failed")
        raise

    print("Registering optimized prompts...")
    try:
        for pt, new_template in prompt_changes:
            if pt.alias:
                mlflow.genai.set_prompt_alias(
                    name=pt.name, alias=rollback_alias, version=pt.prior_version,
                )
            new_pv = mlflow.genai.register_prompt(
                name=pt.name,
                template=new_template,
                commit_message="GEPA-optimized via smart_model_upgrades",
            )
            if pt.alias:
                mlflow.genai.set_prompt_alias(
                    name=pt.name, alias=pt.alias, version=new_pv.version,
                )
                repointed_prompts.append(pt)
            out[pt.name] = new_pv
            print(f"  {pt.name}@{pt.alias}: registered v{new_pv.version}")
    except Exception as e:
        print(f"Prompt registration failed: {e}")
        _rollback("prompt registration failed")
        raise

    return out
