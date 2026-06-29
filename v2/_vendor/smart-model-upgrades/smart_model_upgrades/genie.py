"""Genie Space conversation management via Databricks SDK."""

from databricks.sdk import WorkspaceClient


def clear_conversations(space_id, include_all=False, client=None):
    """Delete all conversations in a Genie space (paginates until empty)."""
    client = client or WorkspaceClient()
    total = 0
    while True:
        resp = client.genie.list_conversations(
            space_id, page_size=100, include_all=include_all,
        )
        convos = resp.conversations or []
        if not convos:
            break
        for c in convos:
            client.genie.delete_conversation(space_id, c.conversation_id)
            total += 1
        print(f"  Deleted {total} so far...")
    print(f"Done. Deleted {total} conversations.")
