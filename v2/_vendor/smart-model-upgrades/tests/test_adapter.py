"""End-to-end tests for the scoring path inside `_AgentAdapter` and its helpers.

These tests exercise the actual scoring formula, scorer dispatch, latency
gate, cost estimation, warn-once paths, and the scorer-success ratchet --
behaviors that the optimize_prompts_and_models tests mock out.
"""
import warnings

import pytest

from smart_model_upgrades.optimization import (
    _AgentAdapter,
    _EndpointTarget,
    _PromptTarget,
    _SCORER_WARNINGS_SEEN,
    _State,
    _convert_to_numeric,
    _estimate_cost_usd,
    _run_scorers,
    _warn_once,
    optimize_prompts_and_models,
)


def _state(*, predict_fn=None, scorers=(), endpoint_targets=(), prompt_targets=(),
           weight_quality=0.7, weight_latency=0.2, weight_cost=0.1,
           latency_hard_gate=60.0, cost_soft_gate=0.02, token_costs=None):
    return _State(
        predict_fn=predict_fn or (lambda x: "ok"),
        prompt_targets=list(prompt_targets),
        endpoint_targets=list(endpoint_targets),
        scorers=list(scorers),
        weight_quality=weight_quality,
        weight_latency=weight_latency,
        weight_cost=weight_cost,
        latency_hard_gate=latency_hard_gate,
        cost_soft_gate=cost_soft_gate,
        reflection_model="reflection",
        token_costs=token_costs or {},
    )


@pytest.fixture(autouse=True)
def _clear_warning_dedup():
    """Each test sees fresh warning dedup; warn_once is module-global by design."""
    _SCORER_WARNINGS_SEEN.clear()


# ---------------------------------------------------------------------------
# _run_scorers matrix
# ---------------------------------------------------------------------------

def test_run_scorers_numeric_callable_averages():
    s1 = lambda inputs, expected, answer: 0.6
    s2 = lambda inputs, expected, answer: 0.8
    state = _state(scorers=[s1, s2])
    assert _run_scorers([s1, s2], {}, "x", "y", state=state) == pytest.approx(0.7)
    assert state.scorer_attempted == 1
    assert state.scorer_succeeded == 1


def test_run_scorers_raising_callable_warns_and_skips(recwarn):
    def bad(inputs, expected, answer):
        raise TypeError("boom")
    s_ok = lambda inputs, expected, answer: 1.0
    state = _state(scorers=[bad, s_ok])

    assert _run_scorers([bad, s_ok], {}, "x", "y", state=state) == pytest.approx(1.0)
    assert state.scorer_attempted == 1
    assert state.scorer_succeeded == 1  # at least one numeric came back
    msgs = [str(w.message) for w in recwarn.list]
    assert any("TypeError: boom" in m for m in msgs)


def test_run_scorers_all_fail_returns_zero_and_does_not_increment_succeeded(recwarn):
    def bad(inputs, expected, answer):
        raise RuntimeError("nope")
    state = _state(scorers=[bad])
    assert _run_scorers([bad], {}, "x", "y", state=state) == 0.0
    assert state.scorer_attempted == 1
    assert state.scorer_succeeded == 0


def test_run_scorers_categorical_rating_yes_no_coerced():
    from mlflow.genai.judges import CategoricalRating
    yes = lambda inputs, expected, answer: CategoricalRating.YES
    no = lambda inputs, expected, answer: CategoricalRating.NO
    state = _state(scorers=[yes, no])
    assert _run_scorers([yes, no], {}, "x", "y", state=state) == pytest.approx(0.5)


def test_run_scorers_feedback_value_unwrapped():
    from mlflow.entities import Feedback
    fb = lambda inputs, expected, answer: Feedback(name="s", value=0.42)
    state = _state(scorers=[fb])
    assert _run_scorers([fb], {}, "x", "y", state=state) == pytest.approx(0.42)


def test_run_scorers_non_numeric_warns_and_skips(recwarn):
    weird = lambda inputs, expected, answer: "not a number"
    state = _state(scorers=[weird])
    assert _run_scorers([weird], {}, "x", "y", state=state) == 0.0
    msgs = [str(w.message) for w in recwarn.list]
    assert any("returned non-numeric" in m for m in msgs)


def test_run_scorers_passes_trace_to_mlflow_scorer(mocker):
    """MLflow Scorer.run gets the trace kwarg so trace-aware judges (e.g.
    make_judge with a Trace template field) can read per-component spans."""
    from mlflow.genai.scorers.base import Scorer as MlflowScorer

    captured = {}

    class FakeScorer(MlflowScorer):
        name: str = "fake"

        def __call__(self, *args, **kwargs):  # pragma: no cover -- abstract guard
            return 1.0

        def run(self, *, inputs=None, outputs=None, expectations=None, trace=None):
            captured["inputs"] = inputs
            captured["outputs"] = outputs
            captured["expectations"] = expectations
            captured["trace"] = trace
            return 0.9

    sentinel_trace = object()
    score = _run_scorers(
        [FakeScorer()],
        inputs={"q": "?"}, expectations={"expected_response": "a"}, answer="answer",
        trace=sentinel_trace,
    )
    assert score == pytest.approx(0.9)
    assert captured["inputs"] == {"q": "?"}
    assert captured["outputs"] == "answer"
    assert captured["expectations"] == {"expected_response": "a"}
    assert captured["trace"] is sentinel_trace


def test_run_scorers_trace_defaults_to_none_for_mlflow_scorer():
    """If callers don't pass trace, MLflow Scorer.run still gets trace=None
    (not a missing kwarg) -- preserves the documented Scorer.run signature."""
    from mlflow.genai.scorers.base import Scorer as MlflowScorer

    captured = {}

    class FakeScorer(MlflowScorer):
        name: str = "fake"

        def __call__(self, *args, **kwargs):  # pragma: no cover
            return 1.0

        def run(self, *, inputs=None, outputs=None, expectations=None, trace=None):
            captured["trace"] = trace
            return 1.0

    _run_scorers(
        [FakeScorer()],
        inputs={}, expectations={}, answer="a",
    )
    assert "trace" in captured
    assert captured["trace"] is None


def test_warn_once_dedupes_within_process(recwarn):
    _warn_once("scorer_x", "raised", "first")
    _warn_once("scorer_x", "raised", "second")
    msgs = [str(w.message) for w in recwarn.list]
    relevant = [m for m in msgs if "scorer_x" in m]
    assert len(relevant) == 1


def test_convert_to_numeric_handles_all_types():
    from mlflow.entities import Feedback
    from mlflow.genai.judges import CategoricalRating
    assert _convert_to_numeric(0.5) == 0.5
    assert _convert_to_numeric(1) == 1.0
    assert _convert_to_numeric(True) == 1.0
    assert _convert_to_numeric(CategoricalRating.YES) == 1.0
    assert _convert_to_numeric(CategoricalRating.NO) == 0.0
    assert _convert_to_numeric(Feedback(name="s", value=0.3)) == pytest.approx(0.3)
    assert _convert_to_numeric("hi") is None
    assert _convert_to_numeric(None) is None


# ---------------------------------------------------------------------------
# _estimate_cost_usd
# ---------------------------------------------------------------------------

def test_estimate_cost_no_endpoints_is_zero():
    assert _estimate_cost_usd({}, [], {"input": 100, "output": 50}, {}) == 0.0


def test_estimate_cost_real_tokens_uses_per_model_rates():
    et = _EndpointTarget(name="ep1", candidate_models=["m1"], initial_model="m1")
    token_costs = {"m1": {"input": 1000.0, "output": 2000.0}}
    cost = _estimate_cost_usd(
        {"model:ep1": "m1"}, [et],
        {"input": 1_000_000, "output": 1_000_000},
        token_costs,
    )
    # 1M input * 1000/1M + 1M output * 2000/1M = 3000 DBU * 0.07 = 210 USD
    assert cost == pytest.approx(210.0)


def test_estimate_cost_no_tokens_uses_fallback():
    et = _EndpointTarget(name="ep1", candidate_models=["m1"], initial_model="m1")
    token_costs = {"m1": {"input": 1.0, "output": 1.0}}
    cost = _estimate_cost_usd({"model:ep1": "m1"}, [et], {}, token_costs)
    # 500 in * 1/1M + 200 out * 1/1M = 0.0007 DBU * 0.07 = ~4.9e-8
    assert cost > 0
    assert cost == pytest.approx(700 / 1_000_000 * 0.07)


def test_estimate_cost_zero_input_only_still_uses_real_tokens():
    """We previously fell back to 500/200 when input was 0. Now: only fallback
    if BOTH are zero. Verifies the simplify-pass C fix."""
    et = _EndpointTarget(name="ep1", candidate_models=["m1"], initial_model="m1")
    token_costs = {"m1": {"input": 1.0, "output": 1.0}}
    # input=0, output=200
    cost_real = _estimate_cost_usd(
        {"model:ep1": "m1"}, [et], {"input": 0, "output": 200}, token_costs,
    )
    cost_fallback = _estimate_cost_usd({"model:ep1": "m1"}, [et], {}, token_costs)
    assert cost_real != cost_fallback
    # cost_real used 0 input, 200 output: 200/1M * 0.07 = 1.4e-8
    assert cost_real == pytest.approx(200 / 1_000_000 * 0.07)


# ---------------------------------------------------------------------------
# _AgentAdapter._run_one composite-score formula
# ---------------------------------------------------------------------------

def test_run_one_composite_score(mocker):
    """Verify quality * 0.7 + latency * 0.2 + cost * 0.1 with mocked trace + scorers."""
    et = _EndpointTarget(name="ep1", candidate_models=["m1"], initial_model="m1")
    state = _state(
        predict_fn=lambda x: "answer",
        scorers=[lambda inputs, expected, answer: 1.0],   # quality = 1.0
        endpoint_targets=[et],
        token_costs={"m1": {"input": 0.0, "output": 0.0}},  # cost = 0 USD
        latency_hard_gate=10.0,
    )
    mocker.patch(
        "smart_model_upgrades.optimization._extract_trace_summary",
        return_value={"total_tokens": {"input": 100, "output": 50}, "spans": []},
    )

    adapter = _AgentAdapter(state)
    score, obj, answer, _, _ = adapter._run_one(
        {"model:ep1": "m1"}, inputs={}, expectations={"expected_response": "answer"},
    )
    # quality=1.0 (perfect scorer)
    # latency very small -> lat_score ~= 1.0
    # cost = 0 -> cost_score = 1.0
    # composite ~= 0.7*1 + 0.2*1 + 0.1*1 = 1.0
    assert obj["quality"] == 1.0
    assert obj["cost"] == pytest.approx(1.0)
    assert obj["latency"] > 0.99
    assert score == pytest.approx(1.0, abs=0.01)
    assert answer == "answer"


def test_run_one_latency_hard_gate(mocker):
    """Predict that takes longer than latency_hard_gate scores 0 across the board."""
    import time as _time

    def slow_predict(x):
        _time.sleep(0.05)
        return "ok"

    et = _EndpointTarget(name="ep1", candidate_models=["m1"], initial_model="m1")
    state = _state(
        predict_fn=slow_predict,
        scorers=[lambda inputs, expected, answer: 1.0],
        endpoint_targets=[et],
        token_costs={"m1": {"input": 0.0, "output": 0.0}},
        latency_hard_gate=0.01,  # 10ms gate -- the sleep blows past it
    )
    mocker.patch(
        "smart_model_upgrades.optimization._extract_trace_summary",
        return_value={"total_tokens": {}, "spans": []},
    )

    adapter = _AgentAdapter(state)
    score, obj, _, feedback, _ = adapter._run_one(
        {"model:ep1": "m1"}, inputs={}, expectations={"expected_response": "ok"},
    )
    assert score == 0.0
    assert obj == {"quality": 0.0, "latency": 0.0, "cost": 0.0}
    assert feedback.startswith("REJECTED: latency")


def test_run_one_predict_failure_yields_error_string(mocker):
    """Predict raising should produce 'ERROR: ...' answer for the scorer to see."""
    et = _EndpointTarget(name="ep1", candidate_models=["m1"], initial_model="m1")

    def boom(x):
        raise ValueError("agent broke")

    seen = []
    def scorer(inputs, expected, answer):
        seen.append(answer)
        return 0.0

    state = _state(
        predict_fn=boom,
        scorers=[scorer],
        endpoint_targets=[et],
        token_costs={"m1": {"input": 0.0, "output": 0.0}},
    )
    mocker.patch(
        "smart_model_upgrades.optimization._extract_trace_summary",
        return_value={"total_tokens": {}, "spans": []},
    )

    adapter = _AgentAdapter(state)
    _, _, answer, _, _ = adapter._run_one({"model:ep1": "m1"}, inputs={}, expectations={"expected_response": "x"})
    assert answer.startswith("ERROR:")
    assert "agent broke" in answer
    assert seen == [answer]


def test_run_one_cost_fallback_warns_when_no_tokens(mocker, recwarn):
    et = _EndpointTarget(name="ep1", candidate_models=["m1"], initial_model="m1")
    state = _state(
        predict_fn=lambda x: "ok",
        scorers=[lambda inputs, expected, answer: 1.0],
        endpoint_targets=[et],
        weight_cost=0.1,
        token_costs={"m1": {"input": 0.0, "output": 0.0}},
    )
    mocker.patch(
        "smart_model_upgrades.optimization._extract_trace_summary",
        return_value={"total_tokens": {}, "spans": []},
    )

    adapter = _AgentAdapter(state)
    adapter._run_one({"model:ep1": "m1"}, inputs={}, expectations={"expected_response": "ok"})

    msgs = [str(w.message) for w in recwarn.list]
    assert any("no_token_usage" in m for m in msgs)


def test_run_one_no_warn_when_weight_cost_zero(mocker, recwarn):
    """If the customer has explicitly disabled the cost component, no warn."""
    et = _EndpointTarget(name="ep1", candidate_models=["m1"], initial_model="m1")
    state = _state(
        predict_fn=lambda x: "ok",
        scorers=[lambda inputs, expected, answer: 1.0],
        endpoint_targets=[et],
        weight_quality=0.8, weight_latency=0.2, weight_cost=0.0,
        token_costs={"m1": {"input": 0.0, "output": 0.0}},
    )
    mocker.patch(
        "smart_model_upgrades.optimization._extract_trace_summary",
        return_value={"total_tokens": {}, "spans": []},
    )

    adapter = _AgentAdapter(state)
    adapter._run_one({"model:ep1": "m1"}, inputs={}, expectations={"expected_response": "ok"})

    msgs = [str(w.message) for w in recwarn.list]
    assert not any("no_token_usage" in m for m in msgs)


# ---------------------------------------------------------------------------
# Scorer-success ratchet (raises if <10% of evals produced a numeric score)
# ---------------------------------------------------------------------------

def test_optimize_raises_when_scorer_success_under_10pct(mocker, recwarn):
    """If almost every scorer call fails, the optimize call should surface
    the issue at the end rather than silently reporting delta=0."""
    pv = mocker.Mock(template="answer the {{question}}", version=3)
    mocker.patch("smart_model_upgrades.optimization.mlflow.genai.load_prompt", return_value=pv)
    mocker.patch(
        "smart_model_upgrades.optimization.gw.get_endpoint",
        return_value={"config": {"destinations": [{"name": "system.ai.m1"}]}},
    )
    mocker.patch("smart_model_upgrades.optimization.gw.create_endpoint")
    mocker.patch("smart_model_upgrades.optimization.gw.delete_endpoint")
    mocker.patch(
        "smart_model_upgrades.optimization._resolve_model_info",
        return_value={"name": "system.ai.m1", "display_name": "M1", "description": ""},
    )

    # Drive _AgentAdapter.evaluate via a fake gepa.optimize that pumps records
    # through the adapter so _run_scorers fires and increments the counters.
    def fake_optimize(*, adapter, valset, **kwargs):
        for _ in range(20):
            adapter.evaluate(valset, kwargs["seed_candidate"])
        return mocker.Mock(
            best_candidate=kwargs["seed_candidate"],
            val_aggregate_scores=[0.0, 0.0],
            best_idx=1,
        )

    mocker.patch("smart_model_upgrades.optimization.gepa.optimize", side_effect=fake_optimize)

    def always_fails(inputs, expected, answer):
        raise RuntimeError("scorer broken")

    val = [{"inputs": {"question": "q"}, "expectations": {"expected_response": "a"}}]
    with pytest.raises(RuntimeError, match="produced a numeric scorer score"):
        optimize_prompts_and_models(
            predict_fn=lambda inputs: "x",
            train_data=[],
            val_data=val,
            prompt_uris=["prompts:/cat.schema.foo@production"],
            scorers=[always_fails],
            max_metric_calls=10,
        )
