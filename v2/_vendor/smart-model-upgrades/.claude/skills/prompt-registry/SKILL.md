---
name: prompt-registry
description: >
  MLflow Prompt Registry on Databricks — manage, version, deploy, evaluate, and optimize prompt
  templates in Unity Catalog. Use this skill whenever the user mentions prompt registry, prompt
  management, prompt versioning, prompt templates, prompt aliases, registering prompts, loading
  prompts, searching prompts, deleting prompts, {{template_variables}}, prompt deployment,
  prompt optimization, GEPA, prompt evaluation, mlflow.genai.register_prompt,
  mlflow.genai.load_prompt, mlflow.genai.set_prompt_alias, mlflow.genai.optimize_prompts,
  or Unity Catalog prompts. Also trigger when the user wants to create, edit, compare, promote,
  or roll back prompts in any Databricks or MLflow context — even if they don't say "prompt registry"
  explicitly. If the user is building a GenAI app on Databricks and mentions prompts at all, use this skill.
---

# MLflow Prompt Registry

The Prompt Registry is a centralized repository for managing prompt templates across their full
lifecycle. It lives inside Unity Catalog, giving you versioning, governance, and lineage tracking
out of the box.

## Data Model

```
Prompt  (named entity in Unity Catalog)
 └─ Version  (immutable snapshot, auto-incrementing)
      ├─ Alias   (mutable pointer: "production", "staging", …)
      └─ Tags    (key-value metadata)
```

**Three principles to internalize:**

1. **Immutable versions** — every edit creates a new version; you never overwrite an existing one.
2. **Alias-based deployment** — mutable named references (`@production`, `@staging`) decouple
   your running app from specific version numbers. Reassign an alias to roll forward or back
   without redeploying.
3. **Commit messages** — every version carries a commit message explaining *why* the change
   was made, just like git.

## Prerequisites

```bash
pip install --upgrade "mlflow[databricks]>=3.1.0" openai
```

The user (or service principal) needs these Unity Catalog permissions on the target schema:

- `CREATE FUNCTION`
- `EXECUTE`
- `MANAGE`

```sql
GRANT CREATE FUNCTION, EXECUTE, MANAGE
ON SCHEMA <catalog>.<schema>
TO `<principal>`;
```

## Quick Start

### 1. Connect to Databricks

```python
import mlflow

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# Link experiment to a UC schema for prompts
mlflow.set_experiment("/Users/<you>/my_experiment")
mlflow.set_experiment_tags({
    "mlflow.promptRegistryLocation": "<catalog>.<schema>"
})
```

### 2. Register a prompt

```python
prompt = mlflow.genai.register_prompt(
    name="<catalog>.<schema>.my_prompt",
    template="Summarize this in {{num_sentences}} sentences:\n{{content}}",
    commit_message="Initial summarization prompt",
    tags={"team": "data-science", "task": "summarization"},
)
print(f"Created {prompt.name} v{prompt.version}")
```

Template variables use **double braces**: `{{variable_name}}`.

### 3. Set an alias and load it

```python
mlflow.genai.set_prompt_alias(
    name="<catalog>.<schema>.my_prompt",
    alias="production",
    version=prompt.version,
)

# In your app — load by alias so updates need no redeployment
prod_prompt = mlflow.genai.load_prompt("prompts:/<catalog>.<schema>.my_prompt@production")
rendered = prod_prompt.format(num_sentences=3, content="The quarterly report shows…")
```

### 4. Use in an LLM call

```python
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
client = w.serving_endpoints.get_open_ai_client()

response = client.chat.completions.create(
    model="databricks-claude-sonnet-4",
    messages=[{"role": "user", "content": rendered}],
)
print(response.choices[0].message.content)
```

## Core Operations

### Register (create or add version)

```python
prompt = mlflow.genai.register_prompt(
    name="catalog.schema.prompt_name",
    template="Your template with {{variables}}",
    commit_message="What changed and why",
    tags={"key": "value"},  # optional metadata
)
```

Calling `register_prompt` on an existing prompt creates a new version automatically.

### Load

```python
# By version number
p = mlflow.genai.load_prompt("prompts:/catalog.schema.name/3")

# By alias
p = mlflow.genai.load_prompt("prompts:/catalog.schema.name@production")

# Alternative keyword syntax
p = mlflow.genai.load_prompt("catalog.schema.name", version="3")

# Graceful fallback (returns None if missing)
p = mlflow.genai.load_prompt("catalog.schema.name", version="99", allow_missing=True)
```

### Search

```python
results = mlflow.genai.search_prompts(
    "catalog = 'my_catalog' AND schema = 'my_schema'"
)

# Filter programmatically
support_prompts = [p for p in results if "support" in p.name.lower()]
tagged = [p for p in results if p.tags.get("team") == "ml-eng"]
```

For large registries, use pagination — see `references/api-reference.md`.

### Alias management

```python
# Point alias to a version
mlflow.genai.set_prompt_alias(name="catalog.schema.name", alias="staging", version=4)

# Remove an alias (the version itself is untouched)
mlflow.genai.delete_prompt_alias(name="catalog.schema.name", alias="staging")
```

### Delete

```python
from mlflow import MlflowClient
client = MlflowClient()

# Delete a single version
client.delete_prompt_version("catalog.schema.name", "2")

# Delete the entire prompt (all versions must be removed first)
versions = client.search_prompt_versions("catalog.schema.name")
for v in versions.prompt_versions:
    client.delete_prompt_version("catalog.schema.name", str(v.version))
client.delete_prompt("catalog.schema.name")
```

## Template Syntax

### Simple template (string)

```python
template = "Answer {{question}} using context:\n{{context}}"
prompt.format(question="What is MLflow?", context="MLflow is…")
```

### Conversation template (list of dicts)

```python
template = [
    {"role": "system", "content": "You are a {{style}} assistant."},
    {"role": "user", "content": "{{question}}"},
]
prompt.format(style="concise", question="Explain RAG")
```

### Framework compatibility

LangChain and LlamaIndex use single-brace `{variable}` syntax. Convert with:

```python
# LangChain
from langchain_core.prompts import ChatPromptTemplate
lc_template = prompt.to_single_brace_format()
chain = ChatPromptTemplate.from_template(lc_template) | llm | parser

# LlamaIndex
from llama_index.core import PromptTemplate
li_template = PromptTemplate(prompt.to_single_brace_format())
```

## When to Read Reference Files

| If you need… | Read |
|---|---|
| Full API signatures, pagination, URI formats, tags | `references/api-reference.md` |
| Alias strategies, production patterns, rollback, promotion workflows | `references/deployment-patterns.md` |
| Evaluation with scorers, GEPA optimization, baseline vs optimized comparison | `references/evaluation-optimization.md` |

Read only the reference you need — they are self-contained.
