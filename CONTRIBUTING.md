# Contributing to Parallel Latent Reasoner (PRLR)

We welcome contributions that push the frontier of high-performance local inference and continuous latent reasoning on Apple Silicon. 

PRLR is an opinionated, craft-focused project modeled after the open-source philosophy of Omarchy: we value technical excellence, clean architecture, empirical evidence, and code that is fast and joyful to work on.

---

## 1. Guiding Principles

- **Focus on Craft and Technical Merit**: Code should be readable, well-structured, and fast. Keep abstractions lean and avoid unnecessary layers.
- **Empirical Evidence**: If you propose an optimization or architectural tweak, bring the numbers. Back claims with measured wall-clock latencies or benchmark logs.
- **Native MLX & Apple Silicon**: PRLR is built specifically for Apple Silicon Metal GPUs. Prefer pure MLX tensor operations that keep memory allocations in unified SRAM/L2 cache and avoid unnecessary host NumPy transfers.
- **Atomic Commits**: Keep git commits clean and focused. Each commit should contain one coherent fix or feature with a succinct, descriptive commit message.

---

## 2. Setting Up Your Development Environment

You will need a Mac with Apple Silicon (M1/M2/M3/M4) running macOS with Metal support and Python >= 3.11:

```bash
# 1. Clone the repository
git clone https://github.com/steph4n-gh/parallel-latent-reasoner.git
cd parallel-latent-reasoner

# 2. Set up a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install in editable development mode with dev dependencies
pip install -e ".[dev]"
```

---

## 3. Testing Standard (100% Pass Required)

PRLR enforces an invariant test suite covering numerical stability, parameter weight-tying, Lipschitz contractive bounds ($\alpha \le 0.05$), and end-to-end decoding:

```bash
# Run the complete test suite
pytest tests/ -v
```

**All 248 tests must pass**. We do not merge pull requests with failing or flaky tests.

---

## 4. Benchmark Validation

Any change touching the core engine (`engine.py`), models (`models.py`), or E-Gate (`egate.py`) must be validated against the multi-domain cognitive suite:

```bash
# Quick visual check via the interactive terminal visualizer
python3 demo.py --interactive

# Automated multi-scale and cognitive verification benchmark
python3 run_benchmark.py
```

---

## 5. Submitting Pull Requests

1. **Fork the repo** and create a feature branch (e.g. `feat/metal-kernel-fusion` or `fix/egate-patience`).
2. **Write or update tests** for your new functionality.
3. **Ensure formatting & tests pass**: Run `pytest tests/ -v`.
4. **Open a clean Pull Request** describing:
   - What problem this solves
   - How it was verified
   - Any measured latency or throughput impact on your hardware (e.g. M3 Max, M4 Pro)

Keep it focused, direct, and straightforward. Thank you for helping build the future of local latent reasoning!
