"""Tests for smu.score and smu.optimize_prompts_and_models wrappers."""
import sys
import types

import pytest

from smart_model_upgrades import score, optimize_prompts_and_models


@pytest.fixture
def fake_predict():
    return lambda inputs: f"answer for {inputs}"


@pytest.fixture
def fake_scorers():
    return [lambda inputs, expected, answer: 1.0]


def test_score_runs_predict_over_val_data(fake_predict, fake_scorers):
    val = [
        {"inputs": {"x": 1}, "expectations": {"expected_response": "y"}},
        {"inputs": {"x": 2}, "expectations": {"expected_response": "z"}},
    ]
    assert score(fake_predict, val, scorers=fake_scorers) == pytest.approx(1.0)


def test_score_returns_zero_on_empty(fake_predict, fake_scorers):
    assert score(fake_predict, [], scorers=fake_scorers) == 0.0


def test_score_requires_scorers(fake_predict):
    with pytest.raises(TypeError):
        score(fake_predict, [])


def test_optimize_prompts_and_models_requires_at_least_one_target(fake_predict, fake_scorers):
    with pytest.raises(ValueError, match="prompt_uris or gateway_endpoints"):
        optimize_prompts_and_models(
            fake_predict, [], [],
            scorers=fake_scorers, max_metric_calls=10,
        )


def test_optimize_prompts_and_models_requires_scorers(fake_predict):
    with pytest.raises(TypeError):
        optimize_prompts_and_models(fake_predict, [], [], max_metric_calls=10)


def test_optimize_prompts_and_models_rejects_unbalanced_weights(fake_predict, fake_scorers):
    with pytest.raises(ValueError, match="weights must sum to 1.0"):
        optimize_prompts_and_models(
            fake_predict, [], [],
            prompt_uris=["prompts:/cat.schema.foo@production"],
            scorers=fake_scorers, max_metric_calls=10,
            weight_quality=1.0, weight_latency=0.5, weight_cost=0.5,
        )


def test_optimize_prompts_and_models_rejects_negative_weights(fake_predict, fake_scorers):
    with pytest.raises(ValueError, match="weight_latency must be >= 0"):
        optimize_prompts_and_models(
            fake_predict, [], [],
            prompt_uris=["prompts:/cat.schema.foo@production"],
            scorers=fake_scorers, max_metric_calls=10,
            weight_quality=1.2, weight_latency=-0.1, weight_cost=-0.1,
        )


def test_optimize_prompts_and_models_rejects_unknown_model_when_cost_weighted(mocker, fake_predict, fake_scorers):
    mocker.patch(
        "smart_model_upgrades.optimization.gw.get_endpoint",
        return_value={"config": {"destinations": [{"name": "system.ai.unknown-model-x"}]}},
    )
    with pytest.raises(ValueError, match="no token cost data"):
        optimize_prompts_and_models(
            fake_predict, [], [],
            gateway_endpoints={"ep1": ["unknown-model-x", "another-unknown"]},
            scorers=fake_scorers, max_metric_calls=10,
        )


def test_optimize_prompts_and_models_accepts_token_costs_for_unknown_model(mocker, fake_predict, fake_scorers):
    mocker.patch(
        "smart_model_upgrades.optimization.gw.get_endpoint",
        return_value={"config": {"destinations": [{"name": "system.ai.custom-model"}]}},
    )
    mocker.patch("smart_model_upgrades.optimization.gw.create_endpoint")
    mocker.patch("smart_model_upgrades.optimization.gw.delete_endpoint")
    mocker.patch("smart_model_upgrades.optimization.gw.update_endpoint")
    mocker.patch(
        "smart_model_upgrades.optimization._resolve_model_info",
        return_value={"name": "system.ai.custom-model", "display_name": "X", "description": ""},
    )
    fake_gepa = mocker.patch("smart_model_upgrades.optimization.gepa")
    fake_gepa.optimize.return_value = mocker.Mock(
        best_candidate={"model:ep1": "custom-model"},
        val_aggregate_scores=[0.5, 0.6], best_idx=1,
    )
    mocker.patch("smart_model_upgrades.optimization._AgentAdapter")

    # Should not raise.
    optimize_prompts_and_models(
        fake_predict, [], [{"inputs": {}, "expectations": {"expected_response": "x"}}],
        gateway_endpoints={"ep1": ["custom-model"]},
        scorers=fake_scorers, max_metric_calls=10,
        token_costs={"custom-model": {"input": 1.0, "output": 5.0}},
    )


def test_optimize_prompts_and_models_threads_inputs_to_gepa_optimize(mocker, fake_predict, fake_scorers):
    """End-to-end mock: prompt loading, endpoint reads, exp lifecycle, gepa.optimize."""
    pv = mocker.Mock(template="Answer the {{question}} succinctly.", version=3)
    mocker.patch("smart_model_upgrades.optimization.mlflow.genai.load_prompt", return_value=pv)

    mocker.patch(
        "smart_model_upgrades.optimization.gw.get_endpoint",
        return_value={"config": {"destinations": [{"name": "system.ai.databricks-claude-sonnet-4"}]}},
    )
    mocker.patch("smart_model_upgrades.optimization.gw.create_endpoint")
    mocker.patch("smart_model_upgrades.optimization.gw.delete_endpoint")
    mocker.patch("smart_model_upgrades.optimization.gw.update_endpoint")
    mocker.patch(
        "smart_model_upgrades.optimization._resolve_model_info",
        return_value={"name": "system.ai.x", "display_name": "X", "description": ""},
    )

    fake_gepa = mocker.patch("smart_model_upgrades.optimization.gepa")
    fake_gepa.optimize.return_value = mocker.Mock(
        best_candidate={"prompt:foo": "Answer the {{question}}.", "model:ep1": "databricks-gpt-5-4-mini"},
        val_aggregate_scores=[0.50, 0.85],
        best_idx=1,
    )
    mocker.patch("smart_model_upgrades.optimization._AgentAdapter")

    result = optimize_prompts_and_models(
        fake_predict, [], [{"inputs": {"question": "q"}, "expectations": {"expected_response": "a"}}],
        prompt_uris=["prompts:/cat.schema.foo@production"],
        gateway_endpoints={"ep1": ["databricks-gpt-5-4-mini", "databricks-claude-sonnet-4"]},
        scorers=fake_scorers,
        max_metric_calls=10,
    )

    assert fake_gepa.optimize.call_count == 1
    kwargs = fake_gepa.optimize.call_args.kwargs
    assert kwargs["seed_candidate"] == {
        "prompt:foo": "Answer the {{question}} succinctly.",
        "model:ep1": "databricks-claude-sonnet-4",
    }
    assert kwargs["max_metric_calls"] == 10
    assert kwargs["reflection_lm"] == "databricks/databricks-claude-sonnet-4-6"
    assert "prompt:foo" in kwargs["reflection_prompt_template"]
    assert "model:ep1" in kwargs["reflection_prompt_template"]
    assert result.prompt_uris == ["prompts:/cat.schema.foo@production"]
    assert result.gateway_endpoints == {"ep1": ["databricks-gpt-5-4-mini", "databricks-claude-sonnet-4"]}
    assert result.baseline_score == 0.50
    assert result.best_score == 0.85


def test_patched_endpoints_rewrites_known_models_and_restores():
    """The context manager should rewrite `model=<ep>` to `<ep>-exp` and restore on exit."""
    from openai.resources.chat.completions import Completions
    from smart_model_upgrades.optimization import _patched_endpoints, _EndpointTarget

    seen = []
    original_create = Completions.create

    def fake_create(self, *args, **kwargs):
        seen.append(kwargs.get("model"))
        return "ok"

    Completions.create = fake_create
    try:
        targets = [
            _EndpointTarget(name="wb-supervisor", candidate_models=["m"], initial_model="m"),
            _EndpointTarget(name="wb-rewriter", candidate_models=["m"], initial_model="m"),
        ]
        with _patched_endpoints(targets):
            Completions.create(None, model="wb-supervisor", messages=[])
            Completions.create(None, model="wb-rewriter", messages=[])
            Completions.create(None, model="some-other-endpoint", messages=[])
        assert seen == ["wb-supervisor-exp", "wb-rewriter-exp", "some-other-endpoint"]
        # After exit, the patch is removed and our fake is back at the top.
        assert Completions.create is fake_create
    finally:
        Completions.create = original_create


def test_patched_endpoints_covers_responses_api():
    """Responses.create should be patched alongside Completions.create."""
    from openai.resources.responses import Responses
    from smart_model_upgrades.optimization import _patched_endpoints, _EndpointTarget

    seen = []
    original = Responses.create

    def fake_create(self, *args, **kwargs):
        seen.append(kwargs.get("model"))
        return "ok"

    Responses.create = fake_create
    try:
        targets = [_EndpointTarget(name="wb-supervisor", candidate_models=["m"], initial_model="m")]
        with _patched_endpoints(targets):
            Responses.create(None, model="wb-supervisor", input=[])
            Responses.create(None, model="some-other", input=[])
        assert seen == ["wb-supervisor-exp", "some-other"]
        assert Responses.create is fake_create
    finally:
        Responses.create = original


def test_preflight_runs_predict_once_before_gepa(mocker, fake_scorers):
    """Pre-flight should call predict_fn on the first record before launching gepa."""
    pv = mocker.Mock(template="answer the {{question}}", version=3)
    mocker.patch("smart_model_upgrades.optimization.mlflow.genai.load_prompt", return_value=pv)
    mocker.patch(
        "smart_model_upgrades.optimization.gw.get_endpoint",
        return_value={"config": {"destinations": [{"name": "system.ai.databricks-gpt-5-4-mini"}]}},
    )
    mocker.patch("smart_model_upgrades.optimization.gw.create_endpoint")
    mocker.patch("smart_model_upgrades.optimization.gw.delete_endpoint")
    mocker.patch("smart_model_upgrades.optimization.gw.update_endpoint")
    mocker.patch(
        "smart_model_upgrades.optimization._resolve_model_info",
        return_value={"name": "system.ai.x", "display_name": "X", "description": ""},
    )

    fake_gepa = mocker.patch("smart_model_upgrades.optimization.gepa")
    fake_gepa.optimize.return_value = mocker.Mock(
        best_candidate={"prompt:foo": "answer the {{question}}", "model:ep1": "databricks-gpt-5-4-mini"},
        val_aggregate_scores=[0.5, 0.5], best_idx=1,
    )
    mocker.patch("smart_model_upgrades.optimization._AgentAdapter")

    calls = []
    def predict(inputs):
        calls.append(inputs)
        return "ok"

    optimize_prompts_and_models(
        predict, [{"inputs": {"question": "first"}, "expectations": {"expected_response": "a"}}], [],
        prompt_uris=["prompts:/cat.schema.foo@production"],
        gateway_endpoints={"ep1": ["databricks-gpt-5-4-mini"]},
        scorers=fake_scorers, max_metric_calls=10,
    )
    assert calls == [{"question": "first"}]


def test_preflight_warns_but_does_not_abort_on_predict_failure(mocker, fake_scorers, capsys):
    """A predict failure on warmup is logged but doesn't kill the run -- GEPA's
    per-eval ERROR path + the scorer-success ratchet handle the rest."""
    pv = mocker.Mock(template="answer the {{question}}", version=3)
    mocker.patch("smart_model_upgrades.optimization.mlflow.genai.load_prompt", return_value=pv)
    mocker.patch(
        "smart_model_upgrades.optimization.gw.get_endpoint",
        return_value={"config": {"destinations": [{"name": "system.ai.databricks-gpt-5-4-mini"}]}},
    )
    mocker.patch("smart_model_upgrades.optimization.gw.create_endpoint")
    mocker.patch("smart_model_upgrades.optimization.gw.delete_endpoint")
    mocker.patch("smart_model_upgrades.optimization.gw.update_endpoint")
    mocker.patch(
        "smart_model_upgrades.optimization._resolve_model_info",
        return_value={"name": "system.ai.x", "display_name": "X", "description": ""},
    )
    fake_gepa = mocker.patch("smart_model_upgrades.optimization.gepa")
    fake_gepa.optimize.return_value = mocker.Mock(
        best_candidate={"prompt:foo": "answer the {{question}}", "model:ep1": "databricks-gpt-5-4-mini"},
        val_aggregate_scores=[0.0, 0.0], best_idx=1,
    )
    mocker.patch("smart_model_upgrades.optimization._AgentAdapter")

    def broken(inputs):
        raise ValueError("agent broken")

    optimize_prompts_and_models(
        broken, [{"inputs": {"question": "q"}, "expectations": {"expected_response": "a"}}], [],
        prompt_uris=["prompts:/cat.schema.foo@production"],
        gateway_endpoints={"ep1": ["databricks-gpt-5-4-mini"]},
        scorers=fake_scorers, max_metric_calls=10,
    )
    fake_gepa.optimize.assert_called_once()
    out = capsys.readouterr().out
    assert "WARN" in out and "ValueError" in out and "agent broken" in out


def test_patched_endpoints_no_targets_is_noop():
    """Passing an empty target list should not touch the OpenAI client."""
    from openai.resources.chat.completions import Completions
    from smart_model_upgrades.optimization import _patched_endpoints

    before = Completions.create
    with _patched_endpoints([]):
        assert Completions.create is before
    assert Completions.create is before


def test_optimize_prompts_and_models_cleans_up_exp_endpoints_on_failure(mocker, fake_predict, fake_scorers):
    pv = mocker.Mock(template="x {{var}}", version=1)
    mocker.patch("smart_model_upgrades.optimization.mlflow.genai.load_prompt", return_value=pv)
    mocker.patch(
        "smart_model_upgrades.optimization.gw.get_endpoint",
        return_value={"config": {"destinations": [{"name": "system.ai.databricks-gpt-5-4-mini"}]}},
    )
    mocker.patch("smart_model_upgrades.optimization.gw.create_endpoint")
    delete = mocker.patch("smart_model_upgrades.optimization.gw.delete_endpoint")

    fake_gepa = mocker.patch("smart_model_upgrades.optimization.gepa")
    fake_gepa.optimize.side_effect = RuntimeError("boom")
    mocker.patch("smart_model_upgrades.optimization._AgentAdapter")

    with pytest.raises(RuntimeError, match="boom"):
        optimize_prompts_and_models(
            fake_predict, [], [],
            prompt_uris=["prompts:/cat.schema.foo@production"],
            gateway_endpoints={"ep1": ["databricks-gpt-5-4-mini"]},
            scorers=fake_scorers,
            max_metric_calls=5,
        )
    delete.assert_called_with("ep1-exp")
