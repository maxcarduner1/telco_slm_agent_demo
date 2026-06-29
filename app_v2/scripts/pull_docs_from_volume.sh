#!/usr/bin/env bash
# pull_docs_from_volume.sh
#
# Downloads the generated telco documents from a UC Volume to the local docs/
# directory so they can be committed to GitHub.  After committing, notebook 02
# will download pre-built files from GitHub instead of generating them with Claude.
#
# Usage:
#   ./scripts/pull_docs_from_volume.sh [catalog] [schema] [profile]
#
# Defaults:
#   catalog : cmegdemos_catalog
#   schema  : network_analytics_enablement
#   profile : fevm-cmegdemos
#
# Examples:
#   ./scripts/pull_docs_from_volume.sh
#   ./scripts/pull_docs_from_volume.sh my_catalog my_schema my_profile

set -euo pipefail

CATALOG="${1:-cmegdemos_catalog}"
SCHEMA="${2:-network_analytics_enablement}"
PROFILE="${3:-fevm-cmegdemos}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_DOCS="${REPO_ROOT}/docs"
VOLUME_PATH="dbfs:/Volumes/${CATALOG}/${SCHEMA}/telco_docs"

echo "Pulling telco documents from UC Volume"
echo "  Volume : ${VOLUME_PATH}"
echo "  Local  : ${LOCAL_DOCS}"
echo "  Profile: ${PROFILE}"
echo ""

# Verify databricks CLI is available
if ! command -v databricks &>/dev/null; then
    echo "ERROR: databricks CLI not found. Install with: pip install databricks-cli"
    exit 1
fi

for subdir in runbooks standards incidents; do
    src="${VOLUME_PATH}/${subdir}/"
    dst="${LOCAL_DOCS}/${subdir}/"
    mkdir -p "${dst}"

    echo "Downloading ${subdir}..."
    # databricks fs cp with --recursive copies all files in the directory
    databricks fs cp --recursive --overwrite --profile "${PROFILE}" "${src}" "${dst}" 2>/dev/null || {
        echo "  WARNING: failed to copy ${subdir} — check that the volume path exists"
        echo "           and notebook 02 has been run at least once."
        continue
    }

    count=$(find "${dst}" -name "*.txt" 2>/dev/null | wc -l | tr -d ' ')
    echo "  ${count} .txt files written to ${dst}"
done

echo ""
echo "Done. To enable pre-built downloads for future deployments:"
echo ""
echo "  git add docs/"
echo "  git commit -m 'Add pre-generated telco docs'"
echo "  git push"
echo ""
echo "Notebook 02 will now download these files from GitHub instead of"
echo "calling Claude — saving ~\$5–10 in token costs per fresh deployment."
