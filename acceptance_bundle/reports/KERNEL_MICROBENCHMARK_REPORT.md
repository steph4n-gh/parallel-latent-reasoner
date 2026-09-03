# Recurrent Latent Memory Kernel Microbenchmark Report

> ⚠️ **DISCLAIMER (Non-Negotiable Evidence Rules 3 & 4)**:  
> *RECURRENT LATENT MEMORY KERNEL MICROBENCHMARK: Measures pure tensor recurrent execution in unified memory on Apple Silicon Metal GPU. Contains ZERO Chain-of-Thought (CoT), language generation, or cognitive reasoning claims per Non-Negotiable Evidence Rules 3 and 4.*

---

## 1. Hardware & Execution Metadata (Rule 10)

- **Timestamp (UTC)**: `2026-09-03T13:10:52.172238+00:00`
- **Command**: `run_kernel_microbenchmark.py --quick`
- **Git Commit**: `a90ad7ecebdd7a2f7c9d7d5a84227bd5bc729732` (Dirty: `True`)
- **Device Name**: `Apple M4 Pro`
- **Platform**: `Darwin 25.6.0` (arm64)
- **Total RAM**: `24.0 GB`
- **Runtime Versions**: Python `3.14.4`, MLX `0.31.2`, NumPy `2.4.6`
- **Random Seed**: `42`

---

## 2. Kernel Microbenchmark Results

| Condition | M (Slots) | T (Steps) | Mode | Median Latency (ms) | Achieved GFLOP/s | Bandwidth (GB/s) | Slot Steps/s | Peak VRAM (MB) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `gemma_2b_m16_t8_b1_bfloat16_compiled` | 16 | 8 | Compiled JIT | 52.71 ms | 352.0 | 24.4 | 2,428 | 1824.02 |

---

## 3. Memory Stability & Zero-Leak Soak Verification

- **`gemma_2b_m16_t8_b1_bfloat16_compiled`**: Peak VRAM `1824.02 MB`, Active VRAM `0.31 MB`, 200-Run Memory Growth: **`0.00 MB`** (✅ ZERO LEAK)

---

## 4. Compliance Attestation
- **Rule 4 (No CoT Claims)**: Verified. All metrics strictly profile recurrent tensor operations.
- **Rule 6 (Hardware Timers)**: Verified. Real hardware timers with `mx.eval()` GPU synchronization.
- **Rule 7 (Measured Memory)**: Verified. Allocator-backed Metal memory queries.
- **Rule 10 (Reproducibility)**: Verified. Full cryptographic hashes, hardware, and runtime logged.

