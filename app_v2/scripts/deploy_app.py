#!/usr/bin/env python3
"""Utility script to deploy source code to an existing Databricks App.

This is intentionally a script (not a notebook task) to avoid confusion with the
main data-setup DAG.
"""

import argparse
import time
from databricks.sdk import WorkspaceClient


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deploy Databricks App utility")
    p.add_argument("--profile", default="DEFAULT", help="Databricks CLI profile")
    p.add_argument("--app-name", default="otel-telco-agent", help="Databricks App name")
    p.add_argument(
        "--source-code-path",
        default=".",
        help="Workspace path containing app source code (for deployments API)",
    )
    p.add_argument("--catalog", default="", help="UC_CATALOG override")
    p.add_argument("--schema", default="", help="UC_SCHEMA override")
    p.add_argument("--warehouse-id", default="", help="DATABRICKS_WAREHOUSE_ID override")
    p.add_argument("--wait-timeout-seconds", type=int, default=600, help="Max wait time")
    p.add_argument("--poll-seconds", type=int, default=15, help="Polling interval")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    w = WorkspaceClient(profile=args.profile)

    def api(method: str, path: str, body=None):
        return w.api_client.do(method, f"/{path}", body=body)

    app_name = args.app_name
    source_code_path = args.source_code_path
    print(f"Profile  : {args.profile}")
    print(f"App name : {app_name}")
    print(f"Source   : {source_code_path}")

    app_info = api("GET", f"api/2.0/apps/{app_name}")
    compute_state = app_info.get("compute_status", {}).get("state", "")
    app_state = app_info.get("app_status", {}).get("state", "")
    print(f"App '{app_name}' found.")
    print(f"  Compute : {compute_state}")
    print(f"  App     : {app_state}")
    print(f"  URL     : {app_info.get('url', '')}")

    # Optionally patch workspace-specific env vars.
    env_overrides = {}
    if args.catalog:
        env_overrides["UC_CATALOG"] = args.catalog
    if args.schema:
        env_overrides["UC_SCHEMA"] = args.schema
    if args.warehouse_id:
        env_overrides["DATABRICKS_WAREHOUSE_ID"] = args.warehouse_id

    if env_overrides:
        current_env = app_info.get("config", {}).get("env", []) if isinstance(app_info.get("config"), dict) else []
        env_map = {e["name"]: e["value"] for e in current_env if isinstance(e, dict) and "name" in e}
        env_map.update(env_overrides)
        new_env = [{"name": k, "value": v} for k, v in env_map.items()]
        api("PATCH", f"api/2.0/apps/{app_name}", {"config": {"env": new_env}})
        print(f"Patched app env vars: {list(env_overrides.keys())}")
    else:
        print("No env overrides provided — skipping env patch.")

    if compute_state not in ("ACTIVE", "RUNNING"):
        print("App compute not running — starting...")
        api("POST", f"api/2.0/apps/{app_name}/start", {})
        for i in range(30):
            time.sleep(10)
            r = api("GET", f"api/2.0/apps/{app_name}")
            cs = r.get("compute_status", {}).get("state", "")
            print(f"  [{(i + 1) * 10}s] compute state: {cs}")
            if cs in ("ACTIVE", "RUNNING"):
                print("  Compute ready.")
                break
        else:
            raise RuntimeError("App compute did not become ACTIVE within 5 min.")
    else:
        print("App compute already running — skipping start.")

    deployment = api("POST", f"api/2.0/apps/{app_name}/deployments", {"source_code_path": source_code_path})
    deployment_id = deployment.get("deployment_id", "")
    print(f"Deployment started: {deployment_id}")

    elapsed = 0
    final_state = None
    while elapsed < args.wait_timeout_seconds:
        time.sleep(args.poll_seconds)
        elapsed += args.poll_seconds
        d = api("GET", f"api/2.0/apps/{app_name}/deployments/{deployment_id}")
        state = d.get("status", {}).get("state", "UNKNOWN")
        msg = d.get("status", {}).get("message", "")
        print(f"  [{elapsed}s] {state}" + (f" — {msg}" if msg else ""))
        if state == "SUCCEEDED":
            final_state = "SUCCEEDED"
            break
        if state in ("FAILED", "CANCELLED"):
            final_state = state
            break

    if final_state != "SUCCEEDED":
        raise RuntimeError(f"Deployment did not succeed (final state: {final_state}). Check app logs.")

    app_now = api("GET", f"api/2.0/apps/{app_name}")
    app_url = app_now.get("url", "")
    print("=" * 60)
    print("DEPLOYMENT COMPLETE")
    print("=" * 60)
    print(f"App   : {app_name}")
    print(f"URL   : {app_url}")
    print(f"Build : {deployment_id}")


if __name__ == "__main__":
    main()
