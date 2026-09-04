#!/usr/bin/env python3
"""Automated downloader for PRLR Production Gemma Adapter Checkpoints.

Supports:
- Google Gemma 4 12B Adapter (3840D, 16 slots, 1 layer, 4 steps) from GitHub Release v0.3.0-alpha
- Google Gemma 2B Adapter (2048D, 16 slots, 1 layer, 4 steps) with SHA-256 verification & backward compatibility
- Official release repository: steph4n-gh/qan-transformers with tag v0.3.0-alpha
- Cryptographic SHA-256 checksum verification before saving
- Optional --url override for mirrors and local artifact endpoints
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.error
import urllib.request

PROJECT_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"

REPO_RELEASE_URL = "https://github.com/steph4n-gh/qan-transformers/releases/download"
DEFAULT_RELEASE_TAG = "v0.3.0-alpha"

CHECKPOINT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "gemma_4_12b": {
        "release_tag": "v0.3.0-alpha",
        "fallback_release_tag": None,
        "weights_filename": "gemma_4_12b_prlr_adapter.safetensors",
        "sidecar_filename": "gemma_4_12b_prlr_adapter.json",
        "expected_sha256": "81412e358ad391753007f53e5148cb6a27097b4e97f06cff72a98701b4f18922",
        "fallback_sha256": None,
        "description": "Google Gemma 4 12B PRLR Adapter (3840D, 16 slots, 1 layer, 4 deliberation steps)",
        "dim": 3840,
    },
    "gemma_2b": {
        "release_tag": "v0.3.0-alpha",
        "fallback_release_tag": "v0.2.0-alpha",
        "weights_filename": "gemma_2b_prlr_adapter.safetensors",
        "sidecar_filename": "gemma_2b_prlr_adapter.json",
        "expected_sha256": "6fa029f60d0cb4d4d1e2e96a29d5b40cf6e2ef30f81d86d634db8be1dddbd69c",
        "fallback_sha256": "6048262d99e5d28851adfc87a379a2796802926605ab74e33553b4d9347028d7",
        "description": "Google Gemma 2B PRLR Adapter (2048D, 16 slots, 1 layer, 4 deliberation steps)",
        "dim": 2048,
    },
}


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 checksum of a file in 1 MB stream chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest_path: Path, quiet: bool = False) -> None:
    """Download a file with streaming progress indication."""
    if not quiet:
        print(f"[*] Downloading: {url}")
        print(f"[*] Destination: {dest_path}")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "PRLR-Checkpoint-Downloader/2.0"},
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
            if total_size > 0 and not quiet:
                percent = downloaded / total_size * 100
                mb_down = downloaded / (1024 * 1024)
                mb_tot = total_size / (1024 * 1024)
                sys.stdout.write(f"\r    [{percent:5.1f}%] {mb_down:.1f} MB / {mb_tot:.1f} MB")
                sys.stdout.flush()

    if not quiet:
        print("\n[✓] Download complete.")


def download_with_fallback(
    primary_url: str,
    fallback_url: Optional[str],
    dest_path: Path,
    quiet: bool = False,
) -> None:
    """Download from primary URL, falling back to secondary URL if primary returns 404."""
    try:
        download_file(primary_url, dest_path, quiet=quiet)
    except urllib.error.HTTPError as err:
        if err.code == 404 and fallback_url:
            if not quiet:
                print(f"[!] Asset not found at primary URL (HTTP 404). Trying fallback: {fallback_url}")
            download_file(fallback_url, dest_path, quiet=quiet)
        else:
            raise


def verify_checkpoint_files(
    weights_path: Path,
    sidecar_path: Path,
    expected_sha: Optional[Union[str, List[str], Tuple[str, ...]]] = None,
) -> Tuple[bool, str]:
    """Verify weights file against expected SHA and sidecar JSON."""
    if not weights_path.exists():
        return False, f"Weights file not found: {weights_path}"

    actual_sha = compute_file_sha256(weights_path)

    # Check against sidecar if present
    if sidecar_path.exists():
        try:
            with open(sidecar_path, "r", encoding="utf-8") as f:
                sidecar_data = json.load(f)
            sidecar_sha = sidecar_data.get("weights_sha256")
            if sidecar_sha and sidecar_sha != actual_sha:
                return False, f"Sidecar hash mismatch: expected {sidecar_sha}, got {actual_sha}"
        except Exception as e:
            return False, f"Failed to read sidecar: {e}"

    # Check against registry expected hash if present
    if expected_sha:
        allowed = [expected_sha] if isinstance(expected_sha, str) else list(expected_sha)
        if actual_sha not in allowed:
            return False, f"Registry hash mismatch: expected {allowed[0]}, got {actual_sha}"

    return True, actual_sha


def ensure_single_model_checkpoint(
    model_key: str,
    target_dir: Path = CHECKPOINT_DIR,
    release_tag: str = DEFAULT_RELEASE_TAG,
    force: bool = False,
    quiet: bool = False,
    custom_url: Optional[str] = None,
) -> Path:
    """Ensure a specific model adapter checkpoint exists and is cryptographically verified."""
    if model_key not in CHECKPOINT_REGISTRY:
        raise ValueError(f"Unknown model key '{model_key}'. Available: {list(CHECKPOINT_REGISTRY.keys())}")

    entry = CHECKPOINT_REGISTRY[model_key]
    weights_name = str(entry["weights_filename"])
    sidecar_name = str(entry["sidecar_filename"])
    tag = str(release_tag or entry["release_tag"])
    fallback_tag = entry.get("fallback_release_tag")
    expected_sha = entry.get("expected_sha256")
    fallback_sha = entry.get("fallback_sha256")

    allowed_hashes: List[str] = []
    if expected_sha:
        allowed_hashes.append(str(expected_sha))
    if fallback_sha:
        allowed_hashes.append(str(fallback_sha))

    target_dir.mkdir(parents=True, exist_ok=True)
    weights_path = target_dir / weights_name
    sidecar_path = target_dir / sidecar_name

    # Determine URLs
    if custom_url:
        if custom_url.endswith(".safetensors"):
            primary_weights_url = custom_url
            primary_sidecar_url = custom_url[:-12] + ".json"
            fallback_weights_url = None
            fallback_sidecar_url = None
        else:
            base_url = custom_url.rstrip("/")
            primary_weights_url = f"{base_url}/{tag}/{weights_name}"
            primary_sidecar_url = f"{base_url}/{tag}/{sidecar_name}"
            fallback_weights_url = None
            fallback_sidecar_url = None
    else:
        primary_weights_url = f"{REPO_RELEASE_URL}/{tag}/{weights_name}"
        fallback_weights_url = f"{REPO_RELEASE_URL}/{fallback_tag}/{weights_name}" if fallback_tag else None
        primary_sidecar_url = f"{REPO_RELEASE_URL}/{tag}/{sidecar_name}"
        fallback_sidecar_url = f"{REPO_RELEASE_URL}/{fallback_tag}/{sidecar_name}" if fallback_tag else None

    # 1. Download sidecar JSON if missing or forced
    if (not sidecar_path.exists() or force) and primary_sidecar_url:
        tmp_sidecar = sidecar_path.with_suffix(".tmp")
        try:
            download_with_fallback(primary_sidecar_url, fallback_sidecar_url, tmp_sidecar, quiet=quiet)
            with open(tmp_sidecar, "r", encoding="utf-8") as f:
                json.load(f)
            tmp_sidecar.replace(sidecar_path)
        except Exception as e:
            tmp_sidecar.unlink(missing_ok=True)
            if not quiet:
                print(f"[!] Warning: Failed to download sidecar from {primary_sidecar_url}: {e}")

    # Extract target SHA from sidecar if available
    target_sha = allowed_hashes[0] if allowed_hashes else None
    if sidecar_path.exists():
        try:
            with open(sidecar_path, "r", encoding="utf-8") as f:
                s_data = json.load(f)
            sidecar_sha = s_data.get("weights_sha256")
            if sidecar_sha:
                target_sha = sidecar_sha
                if sidecar_sha not in allowed_hashes:
                    allowed_hashes.append(sidecar_sha)
        except Exception:
            pass

    # 2. Check existing weights file on disk
    if weights_path.exists() and not force:
        valid, result = verify_checkpoint_files(weights_path, sidecar_path, allowed_hashes)
        if valid:
            if not quiet:
                print(f"[✓] Checkpoint verified for {model_key}: {weights_path} (SHA-256: {result[:16]}...)")
            return weights_path
        else:
            if not quiet:
                print(f"[!] Checksum verification failed for {weights_path}: {result}. Re-downloading...")

    # 3. Download weights and verify SHA-256 before atomic save
    tmp_path = weights_path.with_suffix(".tmp")
    try:
        download_with_fallback(primary_weights_url, fallback_weights_url, tmp_path, quiet=quiet)
        actual_sha = compute_file_sha256(tmp_path)

        if allowed_hashes and actual_sha not in allowed_hashes:
            tmp_path.unlink(missing_ok=True)
            raise ValueError(
                f"Downloaded checkpoint corrupted! Expected {allowed_hashes[0]}, got {actual_sha}"
            )

        tmp_path.replace(weights_path)
        if not quiet:
            print(f"[✓] Successfully downloaded and verified: {weights_path}")
        return weights_path
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        if model_key == "gemma_4_12b":
            raise RuntimeError(
                f"Failed to download Gemma 4 12B checkpoint from {primary_weights_url}: {e}\n"
                f"Note: If this release asset has not yet been uploaded to GitHub Release {tag}, "
                f"you can train the adapter locally using:\n"
                f"    python3 train_gemma4_adapter.py\n"
            ) from e
        else:
            raise RuntimeError(
                f"Failed to download checkpoint from {primary_weights_url}: {e}\n"
                f"Please download manually from: https://github.com/steph4n-gh/qan-transformers/releases/tag/{tag}"
            ) from e


def ensure_checkpoint(
    model: str = "all",
    target_dir: Path = CHECKPOINT_DIR,
    release_tag: str = DEFAULT_RELEASE_TAG,
    force: bool = False,
    quiet: bool = False,
    filename: Optional[str] = None,
    custom_url: Optional[str] = None,
) -> Path:
    """Ensure production adapter checkpoint(s) exist and are verified.

    Backward-compatible entry point for tests and pipeline execution.
    """
    # Infer model from filename if specified
    if filename:
        if "gemma_2b" in filename:
            model = "gemma_2b"
        elif "gemma_4_12b" in filename:
            model = "gemma_4_12b"

    if model == "all":
        # Ensure both 2B and 4_12B
        weights_4 = None
        weights_2 = None
        try:
            weights_2 = ensure_single_model_checkpoint(
                "gemma_2b", target_dir, release_tag, force, quiet, custom_url
            )
        except Exception as e:
            if not quiet:
                print(f"[!] Warning: Gemma 2B check failed: {e}")

        try:
            weights_4 = ensure_single_model_checkpoint(
                "gemma_4_12b", target_dir, release_tag, force, quiet, custom_url
            )
        except Exception as e:
            if weights_2 is not None:
                return weights_2
            raise

        return weights_4 or weights_2 or (target_dir / str(CHECKPOINT_REGISTRY["gemma_4_12b"]["weights_filename"]))
    else:
        return ensure_single_model_checkpoint(
            model, target_dir, release_tag, force, quiet, custom_url
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PRLR Production Adapter Checkpoint Downloader & Integrity Verifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["gemma_4_12b", "gemma_2b", "all"],
        default="gemma_4_12b",
        help="Target model adapter to download/verify.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=CHECKPOINT_DIR,
        help="Target directory for checkpoint storage.",
    )
    parser.add_argument(
        "--release-tag",
        type=str,
        default=DEFAULT_RELEASE_TAG,
        help="GitHub release tag to target.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Custom URL or base download URL override (e.g. for mirrors or internal releases).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist on disk.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress download progress output.",
    )
    args = parser.parse_args()

    try:
        res = ensure_checkpoint(
            model=args.model,
            target_dir=args.target_dir,
            release_tag=args.release_tag,
            force=args.force,
            quiet=args.quiet,
            custom_url=args.url,
        )
        print(f"[✓] Checkpoint ready: {res}")
        return 0
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
