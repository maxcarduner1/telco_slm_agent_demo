"""smart_model_upgrades -- joint model + prompt optimization for agentic LLM apps.

Bring your own agent (BYOA contract: a `predict(inputs: dict)` callable that
loads its prompts via MLflow Prompt Registry and routes through AI Gateway V2
endpoints), then point this library at the prompts and gateway endpoints you
want tuned. The predict return value is passed straight to your scorers, so
strings, dicts, and structured responses all work.

    import smart_model_upgrades as smu
    from mlflow.genai.scorers import Correctness

    SCORERS = [Correctness(model="databricks:/databricks-gpt-5-4-nano")]

    train_data = [{"inputs": {...}, "expectations": {"expected_response": ...}}, ...]
    val_data   = [{"inputs": {...}, "expectations": {"expected_response": ...}}, ...]

    result = smu.optimize_prompts_and_models(
        predict_fn=predict,
        train_data=train_data,
        val_data=val_data,
        prompt_uris=["prompts:/cat.schema.supervisor@production"],
        gateway_endpoints={"wb-supervisor": ["databricks-claude-sonnet-4", ...]},
        scorers=SCORERS,
        max_metric_calls=200,
    )
    smu.promote_to_prod(result)
"""
from .optimization import (
    Result,
    promote_to_prod,
    optimize_prompts_and_models,
    score,
    setup_endpoints,
)

__all__ = [
    "Result",
    "promote_to_prod",
    "optimize_prompts_and_models",
    "score",
    "setup_endpoints",
]
