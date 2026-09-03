#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Verifying acceptance bundle at ${BUNDLE_DIR}..."
python3 "${BUNDLE_DIR}/verify_all.py"
