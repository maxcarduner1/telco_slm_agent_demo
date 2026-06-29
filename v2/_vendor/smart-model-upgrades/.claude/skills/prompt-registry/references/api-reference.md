# Prompt Registry API Reference

## Table of Contents

1. [High-Level SDK Functions](#high-level-sdk-functions)
2. [MlflowClient Methods](#mlflowclient-methods)
3. [URI Formats](#uri-formats)
4. [Pagination](#pagination)
5. [Tags and Metadata](#tags-and-metadata)

---

## High-Level SDK Functions

These are the primary functions in `mlflow.genai.*`:

### `mlflow.genai.register_prompt()`

Create a new prompt or add a version to an existing one.

```python
prompt = mlflow.genai.register_prompt(
    name="catalog.schema.prompt_name",       # 3-level UC name (required)
    template="Hello {{name}}, {{question}}",  # str or list[dict] (required)
    commit_message="Why this version exists", # str (required)
    tags={                                    # dict (optional)
        "author": "team@company.com",
        "use_case": "customer_support",
        "model_compatibility": "gpt-4",
    },
)
# Returns: mlflow.entities.model_registry.prompt.Prompt
# prompt.name, prompt.version, prompt.uri are the key attributes
```

If the prompt name already exists, a new version is created with an auto-incremented version number. Versions are immutable after creation.

### `mlflow.genai.load_prompt()`

Retrieve a prompt by version number or alias.

```python
# By version number (two equivalent syntaxes)
prompt = mlflow.genai.load_prompt(name_or_uri="prompts:/catalog.schema.name/3")
prompt = mlflow.genai.load_prompt(name_or_uri="catalog.schema.name", version="3")

# By alias
prompt = mlflow.genai.load_prompt(name_or_uri="prompts:/catalog.schema.name@production")

# Graceful fallback — returns None instead of raising
prompt = mlflow.genai.load_prompt(
    name_or_uri="catalog.schema.name",
    version="99",
    allow_missing=True,
)
```

The returned prompt object has a `.format(**kwargs)` method to render the template:

```python
rendered = prompt.format(name="Alice", question="How do I file a claim?")
```

### `mlflow.genai.search_prompts()`

Find prompts in a Unity Catalog schema.

```python
results = mlflow.genai.search_prompts(
    filter_string="catalog = 'my_catalog' AND schema = 'my_schema'"
)
# Returns a list of Prompt objects
```

The filter string requires both `catalog` and `schema`. You can further filter in Python:

```python
tagged = [p for p in results if p.tags.get("team") == "support"]
named = [p for p in results if "customer" in p.name.lower()]
```

### `mlflow.genai.set_prompt_alias()`

Create or update a mutable alias pointing to a specific version.

```python
mlflow.genai.set_prompt_alias(
    name="catalog.schema.prompt_name",  # 3-level UC name
    alias="production",                  # alias name (no @ prefix)
    version=3,                           # version number (int)
)
```

Common aliases: `production`, `staging`, `latest`, `production-previous`, `feature-xyz`.

### `mlflow.genai.delete_prompt_alias()`

Remove an alias. The underlying version is unaffected.

```python
mlflow.genai.delete_prompt_alias(
    name="catalog.schema.prompt_name",
    alias="staging",
)
```

### `mlflow.genai.delete_prompt()`

Delete a prompt entirely. **All versions must be deleted first** or the call will fail.

```python
mlflow.genai.delete_prompt(name="catalog.schema.prompt_name")
```

---

## MlflowClient Methods

For lower-level operations (version-level search, deletion), use `MlflowClient`:

```python
from mlflow import MlflowClient
client = MlflowClient()
```

### `client.search_prompts()`

```python
page = client.search_prompts(
    filter_string="catalog='my_catalog' schema='my_schema'",
    max_results=50,
    page_token=None,  # for pagination
)
# page is a list of Prompt objects with a .token attribute for next page
```

### `client.search_prompt_versions()`

```python
response = client.search_prompt_versions("catalog.schema.prompt_name")
# response.prompt_versions is a list of PromptVersion objects
# Each has: .name, .version, .template, .commit_message, .tags
```

### `client.delete_prompt_version()`

```python
client.delete_prompt_version(
    name="catalog.schema.prompt_name",
    version="2",  # string, not int
)
```

### `client.delete_prompt()`

```python
client.delete_prompt("catalog.schema.prompt_name")
# Fails if any versions still exist — delete all versions first
```

---

## URI Formats

Two URI patterns are supported when loading prompts:

| Pattern | Example |
|---|---|
| By version | `prompts:/catalog.schema.name/3` |
| By alias | `prompts:/catalog.schema.name@production` |

The `prompts:/` prefix is required when using the URI syntax. Without it, use keyword args:

```python
mlflow.genai.load_prompt("catalog.schema.name", version="3")
```

---

## Pagination

For schemas with many prompts, use token-based pagination:

```python
all_prompts = []
token = None
while True:
    page = client.search_prompts(
        filter_string="catalog='my_catalog' schema='my_schema'",
        max_results=50,
        page_token=token,
    )
    all_prompts.extend(page)
    token = page.token
    if not token:
        break
```

The same pattern works for `search_prompt_versions` when a prompt has many versions.

---

## Tags and Metadata

Tags are key-value string pairs attached to a prompt version at registration time:

```python
prompt = mlflow.genai.register_prompt(
    name="catalog.schema.name",
    template="...",
    commit_message="...",
    tags={
        "author": "alice@company.com",
        "tested_with": "gpt-4",
        "avg_latency_ms": "1200",
        "team": "content",
        "project": "summarization-v2",
    },
)
```

Tags are useful for:
- **Filtering** prompts programmatically after search
- **Auditing** who created a version and why
- **Tracking** which model a prompt was designed for
- **Organizing** prompts by team or project
