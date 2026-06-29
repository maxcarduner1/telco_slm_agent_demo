"""Internal MLflow Prompt Registry CRUD helpers.

Not part of the public smu API -- `promote_to_prod` handles the registration
flow during optimization via `mlflow.genai.*` directly. Kept here as
advanced/debug utilities.
"""
from typing import Any, List

import mlflow
from mlflow import MlflowClient


def _client() -> MlflowClient:
    return MlflowClient()


def search_prompt(catalog: str, schema: str) -> List[mlflow.entities.model_registry.prompt.Prompt]:
    """Search prompts in a catalog/schema, paginating through all results."""
    client = _client()
    all_prompts = []
    token = None
    while True:
        page = client.search_prompts(
            filter_string=f"catalog='{catalog}' schema='{schema}'",
            max_results=50,
            page_token=token,
        )
        all_prompts.extend(page)
        token = page.token
        if not token:
            break
    return all_prompts


def search_prompt_version(prompt_location: str) -> Any:
    return _client().search_prompt_versions(prompt_location)


def register_prompt(prompt_template: str, catalog: str, schema: str, name: str, commit_message: str):
    """Register a prompt and set the @latest alias to the new version."""
    prompt_location = f"{catalog}.{schema}.{name}"
    prompt = mlflow.genai.register_prompt(
        name=prompt_location,
        template=prompt_template,
        commit_message=commit_message,
    )
    mlflow.genai.set_prompt_alias(
        name=prompt_location,
        alias="latest",
        version=prompt.version,
    )
    print(f"Registered prompt: {prompt.name}  version={prompt.version} alias=latest uri={prompt.uri}")
    return prompt


def delete_prompt_version(name: str, version: int) -> None:
    _client().delete_prompt_version(name, version)


def delete_prompt(prompt_location: str) -> None:
    """Delete a prompt and all its versions. Best-effort -- silent on missing."""
    try:
        for pv in search_prompt_version(prompt_location).prompt_versions:
            delete_prompt_version(name=pv.name, version=pv.version)
            print(f"Deleted prompt version {pv.name}:{pv.version}")
        _client().delete_prompt(prompt_location)
    except Exception as e:
        print(f"Prompt {prompt_location} does not exist, skip delete: {e}")


def get_last_version(prompt_location: str) -> int:
    return int(search_prompt_version(prompt_location).prompt_versions[0].version)
