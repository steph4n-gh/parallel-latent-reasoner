# Contributing to Parallel Latent Reasoner (PRLR)

Thank you for your interest in contributing to PRLR! We welcome community contributions in the areas of recurrent-depth architectures, MLX kernel optimizations, continuous-latent reasoning probes, and distillation datasets.

## Development Workflow

1. **Fork and Clone**:
   ```bash
   git clone https://github.com/steph4n-gh/parallel-latent-reasoner.git
   cd parallel-latent-reasoner
   ```

2. **Environment Setup**:
   Requires Python >= 3.10 and macOS with Apple Silicon (Metal GPU).
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .
   pip install pytest pytest-xdist
   ```

3. **Running the Test Suite**:
   PRLR maintains a strict 100% pass-rate requirement:
   ```bash
   pytest tests/ -v
   ```

4. **Code Quality & Guidelines**:
   - Write pure MLX operations where possible; avoid unnecessary NumPy host-device round-trips.
   - Maintain strict Lipschitz contractive residual dynamics (ReZero $\alpha \le 0.05$).
   - Ensure all public APIs include full type hints and docstrings.
   - Any architectural changes must be validated against the 25-case cognitive benchmark suite.

## Reporting Issues

If you discover bugs, numerical instability, or unexpected behavior:
1. Open an Issue on GitHub detailing your hardware (e.g. M3 Max, M4 Pro), macOS version, and MLX version.
2. Provide a minimal reproducible example script.
