#!/usr/bin/env bash
# ==============================================================================
# Parallel Latent Reasoner (PRLR) — Single-Command E2E Reproducible Verifier
# Milestone 6 Requirement R9 / Feature 29
# ==============================================================================
set -euo pipefail

# Resolve repository and project root directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRLR_DIR="${SCRIPT_DIR}/../projects/parallel_latent_reasoner"
if [[ ! -d "${PRLR_DIR}" ]]; then
    PRLR_DIR="${SCRIPT_DIR}"
fi

echo "================================================================================"
echo "  PARALLEL LATENT REASONER (PRLR) — E2E REPRODUCIBLE VERIFICATION SUITE"
echo "  Target: Milestone 6 (R9: Features 28, 29)"
echo "  Directory: ${PRLR_DIR}"
echo "================================================================================"

export PYTHONPATH="${PRLR_DIR}/src:${PYTHONPATH:-}"

# Check Python environment
if ! command -v python3 &>/dev/null; then
    echo "[-] Error: python3 not found in PATH" >&2
    exit 1
fi

PYTHON_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "[*] Using Python: $(which python3) (version ${PYTHON_VERSION})"

# Execute Python verification runner
RUNNER_SCRIPT="${SCRIPT_DIR}/run_prlr_verification.py"
if [[ ! -f "${RUNNER_SCRIPT}" ]]; then
    RUNNER_SCRIPT="${PRLR_DIR}/scripts/run_prlr_verification.py"
fi
if [[ ! -f "${RUNNER_SCRIPT}" ]]; then
    RUNNER_SCRIPT="/Volumes/Storage/qan_transformers/.agents/teamwork_preview_explorer_m6_2/proposed_run_prlr_verification.py"
fi

python3 "${RUNNER_SCRIPT}" "$@"
EXIT_CODE=$?

if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo ""
    echo "================================================================================"
    echo "  [SUCCESS] All verification gates passed with returncode 0."
    echo "================================================================================"
else
    echo ""
    echo "================================================================================"
    echo "  [FAILURE] Verification suite failed with returncode ${EXIT_CODE}."
    echo "================================================================================"
fi

exit ${EXIT_CODE}
