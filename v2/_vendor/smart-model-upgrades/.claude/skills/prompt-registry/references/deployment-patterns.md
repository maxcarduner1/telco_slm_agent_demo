# Prompt Deployment Patterns

## Table of Contents

1. [Alias Strategies](#alias-strategies)
2. [Production App Pattern](#production-app-pattern)
3. [Promotion Workflow](#promotion-workflow)
4. [Rollback](#rollback)
5. [Predict Function Pattern](#predict-function-pattern)
6. [Agent Framework Note](#agent-framework-note)

---

## Alias Strategies

Aliases are the deployment mechanism — they let you update what prompt your app uses without
touching the app code. Choose a strategy that fits your team:

### Environment-based (most common)

```
@dev        → version 5  (latest experiment)
@staging    → version 4  (under review)
@production → version 3  (serving traffic)
```

### Feature-branch

For A/B testing or parallel experiments:

```
@production     → version 3
@feature-tone   → version 6  (testing new tone)
@feature-short  → version 7  (testing shorter responses)
```

### Regional

When prompts differ by locale or regulatory requirements:

```
@production-us   → version 3
@production-eu   → version 4  (GDPR-specific language)
@production-asia → version 5  (localized style)
```

---

## Production App Pattern

The key idea: your app loads prompts by alias, and you control which version is live by
reassigning the alias — no redeployment needed.

```python
import mlflow
import os

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

PROMPT_NAME = os.getenv("PROMPT_URI", "catalog.schema.my_prompt")
PROMPT_ALIAS = os.getenv("PROMPT_ALIAS", "production")

def get_prompt():
    """Load the current production prompt."""
    return mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@{PROMPT_ALIAS}")

def generate_response(user_input: str) -> str:
    prompt = get_prompt()
    rendered = prompt.format(input=user_input)
    # Call your LLM here
    response = client.chat.completions.create(
        model="databricks-claude-sonnet-4",
        messages=[{"role": "user", "content": rendered}],
    )
    return response.choices[0].message.content
```

Using environment variables for the prompt name and alias makes the same app code
work across dev, staging, and production environments.

---

## Promotion Workflow

A typical flow for moving a prompt from development to production:

### Step 1: Iterate in dev

```python
# Register new versions as you iterate
prompt = mlflow.genai.register_prompt(
    name="catalog.schema.my_prompt",
    template=improved_template,
    commit_message="Added few-shot examples for better accuracy",
)
mlflow.genai.set_prompt_alias(name="catalog.schema.my_prompt", alias="dev", version=prompt.version)
```

### Step 2: Promote to staging

Once evaluation looks good:

```python
mlflow.genai.set_prompt_alias(name="catalog.schema.my_prompt", alias="staging", version=prompt.version)
```

### Step 3: Promote to production

After stakeholder review or automated checks pass:

```python
# Save current production for rollback
current_prod = mlflow.genai.load_prompt("prompts:/catalog.schema.my_prompt@production")
mlflow.genai.set_prompt_alias(
    name="catalog.schema.my_prompt",
    alias="production-previous",
    version=current_prod.version,
)

# Promote
mlflow.genai.set_prompt_alias(
    name="catalog.schema.my_prompt",
    alias="production",
    version=prompt.version,
)
print(f"Promoted v{prompt.version} to production")
```

---

## Rollback

If something goes wrong after promotion, roll back instantly:

```python
# Load the previous production version
prev = mlflow.genai.load_prompt("prompts:/catalog.schema.my_prompt@production-previous")

# Point production back to it
mlflow.genai.set_prompt_alias(
    name="catalog.schema.my_prompt",
    alias="production",
    version=prev.version,
)
print(f"Rolled back to v{prev.version}")
```

This takes effect immediately — any app loading `@production` will pick up the change on
its next prompt load. No redeployment or restart required.

---

## Predict Function Pattern

A common pattern for evaluation and optimization: a predict function that always loads the
latest prompt version dynamically. This way, when optimization creates a new version, the
predict function automatically picks it up.

```python
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
ai_client = w.serving_endpoints.get_open_ai_client()

FULL_PROMPT_NAME = "catalog.schema.my_prompt"
TARGET_MODEL = "databricks-gpt-oss-20b"

def predict_fn(input_text: str) -> str:
    """Generate a response using the latest registered prompt."""
    # Get latest version number
    from mlflow import MlflowClient
    client = MlflowClient()
    versions = client.search_prompt_versions(FULL_PROMPT_NAME)
    latest_version = versions.prompt_versions[0].version

    # Load and format
    prompt = mlflow.genai.load_prompt(f"prompts:/{FULL_PROMPT_NAME}/{latest_version}")
    rendered = prompt.format(input_text=input_text)

    # Call LLM
    completion = ai_client.chat.completions.create(
        model=TARGET_MODEL,
        messages=[{"role": "user", "content": rendered}],
    )
    content = completion.choices[0].message.content

    # Handle thinking models that return a list of content blocks
    if isinstance(content, list):
        text_parts = [block["text"] for block in content if block.get("type") == "text"]
        return "\n\n".join(text_parts)
    return content
```

---

## Agent Framework Note

When using **Mosaic AI Agent Framework**, automatic authentication passthrough is disabled
for prompt registry access. You need to set credentials explicitly:

```python
import os
os.environ["DATABRICKS_HOST"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
os.environ["DATABRICKS_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
```

This is only needed inside Agent Framework apps — standard notebooks and Databricks apps
handle authentication automatically.
