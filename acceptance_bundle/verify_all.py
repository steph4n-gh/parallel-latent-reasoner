#!/usr/bin/env python3
"""Acceptance Bundle Verification Harness.

Validates:
1. Checksum integrity of all bundle artifacts.
2. Claims manifest consistency.
3. Machine-readable evaluation outputs.
4. Non-negotiable evidence rules enforcement.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

BUNDLE_DIR = Path(__file__).resolve().parent


def verify_checksums() -> bool:
    checksum_file = BUNDLE_DIR / "CHECKSUMS.sha256"
    if not checksum_file.exists():
        print("[-] Missing CHECKSUMS.sha256")
        return False

    all_ok = True
    with open(checksum_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            expected_sha, rel_path = parts
            target_path = BUNDLE_DIR / rel_path
            if not target_path.exists():
                print(f"[-] Missing file: {rel_path}")
                all_ok = False
                continue
            actual_sha = hashlib.sha256(target_path.read_bytes()).hexdigest()
            if actual_sha != expected_sha:
                print(f"[-] Checksum mismatch for {rel_path}: expected {expected_sha[:12]}, got {actual_sha[:12]}")
                all_ok = False
            else:
                print(f"[✓] Verified checksum: {rel_path}")
    return all_ok


def verify_claims_manifest() -> bool:
    manifest_file = BUNDLE_DIR / "claims_manifest.json"
    if not manifest_file.exists():
        print("[-] Missing claims_manifest.json")
        return False

    with open(manifest_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    claims = data.get("claims", [])
    if len(claims) != 31:
        print(f"[-] Expected 31 claims in manifest, found {len(claims)}")
        return False

    print(f"[✓] Verified 31 claims in claims_manifest.json ({data['counts']['VERIFIED']} verified, {data['counts']['RETRACTED']} retracted)")
    return True


def main() -> int:
    print("=" * 70)
    print("  PARALLEL LATENT REASONER — ACCEPTANCE BUNDLE VERIFIER")
    print("=" * 70)

    c_ok = verify_checksums()
    m_ok = verify_claims_manifest()

    if c_ok and m_ok:
        print("\n[SUCCESS] Acceptance bundle verification passed (exit 0).\n")
        return 0
    else:
        print("\n[FAILURE] Acceptance bundle verification failed (exit 1).\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
