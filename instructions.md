given this doc, https://docs.google.com/document/d/1s5BGSWOvwGGNSfIuhWDX4WTUOkvCc6KZABUdncBcDUY/edit?tab=t.0#heading=h.lmc70jx8d5fs

help me create a demo that demonstrates the core capabilities outlined

don't worry about sharepoint, just create some docs and put in a uc volume

use CMEG workspace and network_analytics_enablement schema for this: cmegdemos_catalog.network_analytics_enablement

we also need some fake network data to analyze several different KPIs

see the network_kc_agentic_architecture diagram here, let's use all the core components but only a few VS indexes and one genie for structured KPI retrieval that points to a metric view that defines the KPIs consistently

create a new architecture diagram for me to import into lucid chart so that I can approve before we really start and lay out all of your steps

we absolutely want to leverage the recommended hugging face models for OTEL to power the SLMs that do all the RAG over various sub domain vs indexes but use a bigger frontier model for the supervisor, do this with langraph wrapped in our responseagent framework hosted on apps compute with long term and short term memory powered by lakebase. see https://github.com/databricks/app-templates/blob/main/agent-langgraph-advanced/README.md


