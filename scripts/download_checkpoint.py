#!/usr/bin/env python3
"""Automated downloader for PRLR Production Gemma 2B Adapter Checkpoint.

Downloads the production checkpoint artifact from GitHub Releases (v0.2.0-alpha)
and verifies its cryptographic SHA-256 hash.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import urllib.request

PROJECT_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"

RELEASE_TAG = "v0.2.0-alpha"
REPO_URL = "https://github.com/steph4n-gh/parallel-latent-reasoner/releases/download"

WEIGHTS_FILENAME = "gemma_2b_prlr_adapter.safetensors"
SIDECAR_FILENAME = "gemma_2b_prlr_adapter.json"

EXPECTED_WEIGHTS_SHA256 = "6048262d99e5d28851adfc87a379a2796802926605ab74e33553b4d9347028d7"


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest_path: Path) -> None:
    """Download a file with visual progress indication."""
    print(f"[*] Downloading: {url}")
    print(f"[*] Destination: {dest_path}")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "PRLR-Checkpoint-Downloader/1.0"},
    )

    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = downloaded / total_size * 100
                mb_down = downloaded / (1024 * 1024)
                mb_tot = total_size / (1024 * 1024)
                sys.stdout.write(f"\r    [{percent:5.1f}%] {mb_down:.1f} MB / {mb_tot:.1f} MB")
                sys.stdout.flush()

    print("\n[✓] Download complete.")


def ensure_checkpoint(
    target_dir: Path = CHECKPOINT_DIR,
    force: bool = False,
    quiet: bool = False,
) -> Path:
    """Ensure production adapter checkpoint exists and is verified.

    Downloads from GitHub Releases if missing or invalid.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    weights_path = target_dir / WEIGHTS_FILENAME
    sidecar_path = target_dir / SIDECAR_FILENAME

    # 1. Download sidecar JSON if missing
    sidecar_url = f"{REPO_URL}/{RELEASE_TAG}/{SIDECAR_FILENAME}"
    if not sidecar_path.exists() or force:
        try:
            download_file(sidecar_url, sidecar_path)
        except Exception as e:
            if not quiet:
                print(f"[!] Warning: Failed to download sidecar from {sidecar_url}: {e}")

    # 2. Check weights existence and checksum
    needs_download = force or not weights_path.exists()
    if not needs_download:
        if not quiet:
            print(f"[*] Verifying existing checkpoint: {weights_path}")
        actual_sha = compute_file_sha256(weights_path)
        if actual_sha == EXPECTED_WEIGHTS_SHA256:
            if not quiet:
                print(f"[✓] Checkpoint verified (SHA-256: {actual_sha[:16]}...)")
            return weights_path
        else:
            print(f"[!] Checksum mismatch! Expected {EXPECTED_WEIGHTS_SHA256}, got {actual_sha}")
            needs_download = True

    # 3. Download weights from GitHub release
    weights_url = f"{REPO_URL}/{RELEASE_TAG}/{WEIGHTS_FILENAME}"
    tmp_path = weights_path.with_suffix(".tmp")

    try:
        download_file(weights_url, tmp_path)
        actual_sha = compute_file_sha256(tmp_path)
        if actual_sha != EXPECTED_WEIGHTS_SHA256:
            tmp_path.unlink(missing_ok=True)
            raise ValueError(
                f"Downloaded checkpoint corrupted! Expected {EXPECTED_WEIGHTS_SHA256}, got {actual_sha}"
            )
        tmp_path.replace(weights_path)
        if not quiet:
            print(f"[✓] Successfully downloaded and verified: {weights_path}")
        return weights_path
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download checkpoint from {weights_url}: {e}\n"
            f"Please download manually from: https://github.com/steph4n-gh/parallel-latent-reasoner/releases/tag/{RELEASE_TAG}"
        ) from e


def main() -> int:
    parser = argparse.ArgumentParser(description="Download PRLR production adapter checkpoint")
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=CHECKPOINT_DIR,
        help="Target directory for checkpoint (default: checkpoints/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file exists",
    )
    args = parser.parse_args()

    try:
        ensure_checkpoint(target_dir=args.target_dir, force=args.force)
        return 0
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
