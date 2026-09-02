# Guide: Apple Silicon Hardware & Model Sizing Reference

PRLR is optimized specifically for the unified memory architecture of Apple Silicon (M-series processors), where the CPU and Metal GPU share high-bandwidth unified RAM.

## 1. Memory Bandwidth vs. Compute Mechanics

| Factor | Autoregressive Generation | Parallel Latent Deliberation (PRLR) |
|---|---|---|
| **Execution Phase** | Memory-Bandwidth Bound | Compute / Cache Bound |
| **Arithmetic Intensity** | $\sim 1\text{ FLOP / Byte}$ | $> 100\text{ FLOP / Byte}$ |
| **Weight Access** | Read full weights for *every* token | Weights resident in L2/SRAM during sweeps |
| **KV-Cache Footprint** | $O(N)$ growth per token | Strictly constant $M=16$ slots |

## 2. Apple Silicon Sizing & Compatibility Table

| Mac Hardware Tier | Unified RAM | Max Metal Process Cap | Optimal Model Preset | Max Tested Context |
|---|:---:|:---:|:---:|:---:|
| **M2 / M3 / M4 Base** | 16 GB | ~11.5 GB | `gemma_2b`, `gemma_e4b` | 8k tokens |
| **M3 Pro / M4 Pro** | 24 GB | ~17.5 GB | `gemma_12b_q4` (5.02 GB) | 16k tokens |
| **M2 Max / M3 Max / M4 Max** | 36–64 GB | ~28–48 GB | `gemma_26b_a4b` (6.38 GB) | 32k tokens |
| **M2 Ultra / M3 Ultra** | 128–192 GB | ~96–144 GB | `gemma_26b_a4b` (Full FP16) | 64k tokens |

## 3. Running Multi-Scale Benchmarks

To benchmark all resident scales on your local Mac:

```bash
python3 run_benchmark.py --presets compact_test,gemma_2b,gemma_9b,gemma_12b
```

Summary metrics will be exported to `results/scale_benchmark_summary.json` and `.csv`.
