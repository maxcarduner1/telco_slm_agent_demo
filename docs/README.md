# Pre-built Telco Documents

This directory holds the 23 pre-generated telco documents used by the RAG pipeline.
Committing them here lets customers deploy the demo without spending Claude API tokens
on document generation.

## Structure

```
docs/
├── runbooks/    # 10 operational runbooks (VSWR, latency, coverage, VoLTE, etc.)
├── standards/   # 5 standards summaries (3GPP TS 38.331, 23.501, O-RAN WG2/WG4, TS 28.552)
└── incidents/   # 8 incident RCA reports (historical post-mortems)
```

All files are plain text (`.txt`). Notebook `03_parse_documents` chunks them for
Vector Search indexing.

## Populating this directory

If the `docs/` folders are empty, run the data setup job with the default settings
to generate documents with Claude Sonnet 4, then pull them back with:

```bash
./scripts/pull_docs_from_volume.sh <catalog> <schema> <profile>
# Example:
./scripts/pull_docs_from_volume.sh cmegdemos_catalog network_analytics_enablement fevm-cmegdemos
```

Then commit and push so future deployments use the pre-built files:

```bash
git add docs/
git commit -m "Add pre-generated telco docs"
git push
```
