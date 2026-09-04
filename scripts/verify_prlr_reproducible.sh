#!/usr/bin/env bash
# ==============================================================================
# Parallel Latent Reasoner (PRLR) — Single-Command E2E Reproducible Verifier
# Milestone 6 Requirement R9 / Feature 29
#
# Stages:
#   Stage 1: Clean Package Build (Wheel & Sdist via PEP 517)
#   Stage 2: Clean Checkout Test Resilience (Checkpoint Absence Mode)
#   Stage 3: Checkpoint Downloader & Cryptographic Integrity Verification
#   Stage 4: Comprehensive E2E Verification & Attestation (Checkpoint Presence Mode)
# ==============================================================================
set -euo pipefail

# ------------------------------------------------------------------------------
# Discovery & Environment Setup
# ------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/pyproject.toml" ]]; then
    PRLR_DIR="${SCRIPT_DIR}"
elif [[ -f "${SCRIPT_DIR}/../pyproject.toml" ]]; then
    PRLR_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    PRLR_DIR="${SCRIPT_DIR}"
fi

cd "${PRLR_DIR}"
export PYTHONPATH="${PRLR_DIR}/src:${PYTHONPATH:-}"

echo "================================================================================"
echo "  PARALLEL LATENT REASONER (PRLR) — E2E REPRODUCIBLE VERIFICATION SUITE"
echo "  Target: Milestone 6 (R9: Features 28, 29)"
echo "  Directory: ${PRLR_DIR}"
echo "================================================================================"

if ! command -v python3 &>/dev/null; then
    echo "[-] Error: python3 not found in PATH" >&2
    exit 1
fi

PYTHON_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
echo "[*] Using Python: $(which python3) (version ${PYTHON_VERSION})"
echo ""

# ------------------------------------------------------------------------------
# Stage 1: Packaging Verification (PEP 517 Build)
# ------------------------------------------------------------------------------
echo "================================================================================"
echo "  >>> STAGE 1: PACKAGING VERIFICATION (PEP 517 Build)"
echo "================================================================================"
echo "[*] Building source distribution and wheel into dist/..."

python3 - <<'EOF'
import sys
from pathlib import Path

dist_dir = Path("dist")
dist_dir.mkdir(parents=True, exist_ok=True)

try:
    import setuptools.build_meta as b
    wheel_name = b.build_wheel(str(dist_dir))
    sdist_name = b.build_sdist(str(dist_dir))
    print(f"[✓] Built wheel: {wheel_name}")
    print(f"[✓] Built sdist: {sdist_name}")
    assert (dist_dir / wheel_name).exists() and (dist_dir / wheel_name).stat().st_size > 0, "Wheel artifact invalid"
    assert (dist_dir / sdist_name).exists() and (dist_dir / sdist_name).stat().st_size > 0, "Sdist artifact invalid"
except Exception as e:
    print(f"[-] Packaging build failed: {e}", file=sys.stderr)
    sys.exit(1)
EOF
echo "[✓] Stage 1: Packaging verification passed."
echo ""

# ------------------------------------------------------------------------------
# Stage 2: Clean Checkout Test Resilience (Checkpoint Absence Mode)
# ------------------------------------------------------------------------------
echo "================================================================================"
echo "  >>> STAGE 2: CLEAN CHECKOUT TEST RESILIENCE (Checkpoint Absence Mode)"
echo "================================================================================"

CHECKPOINTS_DIR="${PRLR_DIR}/checkpoints"
TMP_STASH=""

cleanup_stash() {
    if [[ -n "${TMP_STASH}" && -d "${TMP_STASH}" ]]; then
        echo "[*] Restoring stashed checkpoints from ${TMP_STASH}..."
        if compgen -G "${TMP_STASH}/*.safetensors" > /dev/null; then
            mv "${TMP_STASH}"/*.safetensors "${CHECKPOINTS_DIR}/" 2>/dev/null || true
        fi
        rm -rf "${TMP_STASH}"
        echo "[✓] Checkpoints restored successfully."
    fi
}
trap cleanup_stash EXIT INT TERM

TMP_STASH="$(mktemp -d "${TMPDIR:-/tmp}/prlr_checkpoints_stash.XXXXXX")"

# Stash any existing .safetensors files to simulate clean clone
if compgen -G "${CHECKPOINTS_DIR}/*.safetensors" > /dev/null; then
    echo "[*] Stashing .safetensors files to temporary directory ${TMP_STASH}..."
    mv "${CHECKPOINTS_DIR}"/*.safetensors "${TMP_STASH}/"
else
    echo "[*] No local .safetensors files detected in ${CHECKPOINTS_DIR}."
fi

echo "[*] Executing checkpoint-dependent test suite in absence mode..."
ABSENCE_TESTS=(
    "tests/test_system_separation.py"
    "tests/test_challenger_m1_gemma4_adapter.py"
    "tests/test_challenger_m1_production_adapter.py"
    "tests/test_challenger_gemma2b_adapter.py"
    "tests/test_challenger_m3_empirical.py"
    "tests/test_production_pipeline.py"
    "tests/test_challenger_m1_empirical_adversarial.py"
)

python3 -m pytest "${ABSENCE_TESTS[@]}" -q --no-header
ABSENCE_EXIT_CODE=$?

if [[ ${ABSENCE_EXIT_CODE} -ne 0 ]]; then
    echo "[-] Error: Tests failed when checkpoints were absent (exit code ${ABSENCE_EXIT_CODE})!" >&2
    exit ${ABSENCE_EXIT_CODE}
fi

echo "[✓] Stage 2: Clean checkout resilience verified (0 failures, 0 errors in absence mode)."
echo ""

# Explicitly restore stashed checkpoints for stages 3 and 4
cleanup_stash
TMP_STASH=""
echo ""

# ------------------------------------------------------------------------------
# Stage 3: Checkpoint Downloader Verification
# ------------------------------------------------------------------------------
echo "================================================================================"
echo "  >>> STAGE 3: CHECKPOINT DOWNLOADER & CRYPTOGRAPHIC VERIFICATION"
echo "================================================================================"

DOWNLOADER_SCRIPT="${PRLR_DIR}/scripts/download_checkpoint.py"
if [[ -f "${DOWNLOADER_SCRIPT}" ]]; then
    echo "[*] Verifying checkpoint registry and cryptographic SHA-256 integrity..."
    python3 "${DOWNLOADER_SCRIPT}" --model all
    echo "[✓] Stage 3: Checkpoint integrity verification passed."
else
    echo "[!] Warning: ${DOWNLOADER_SCRIPT} not found, skipping Stage 3."
fi
echo ""

# ------------------------------------------------------------------------------
# Stage 4: Comprehensive E2E Verification & Attestation (Checkpoint Presence Mode)
# ------------------------------------------------------------------------------
echo "================================================================================"
echo "  >>> STAGE 4: COMPREHENSIVE E2E VERIFICATION & ATTESTATION"
echo "================================================================================"

RUNNER_SCRIPT="${PRLR_DIR}/scripts/run_prlr_verification.py"
if [[ ! -f "${RUNNER_SCRIPT}" ]]; then
    RUNNER_SCRIPT="${SCRIPT_DIR}/run_prlr_verification.py"
fi

python3 "${RUNNER_SCRIPT}" "$@"
VERIFY_EXIT_CODE=$?

if [[ ${VERIFY_EXIT_CODE} -eq 0 ]]; then
    echo ""
    echo "================================================================================"
    echo "  [SUCCESS] ALL REPRODUCIBILITY & VERIFICATION GATES PASSED (Returncode 0)"
    echo "================================================================================"
else
    echo ""
    echo "================================================================================"
    echo "  [FAILURE] Verification runner failed with returncode ${VERIFY_EXIT_CODE}."
    echo "================================================================================"
fi

exit ${VERIFY_EXIT_CODE}
