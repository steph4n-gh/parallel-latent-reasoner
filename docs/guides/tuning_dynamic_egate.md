# Guide: Tuning the 3-Signal Dynamic Consensus E-Gate

The 3-Signal Dynamic Consensus E-Gate monitors continuous latent trajectories in real-time and halts computation as soon as the representation converges, saving up to $83\%$ of compute on simpler inputs.

## 1. The 3 Consensus Signals

$$\text{Halt}(t) = (t \ge T_{\min}) \land \left[ \left( \frac{v(t)}{v(1)} < \tau_v \right) \land \left( \hat{y}^{(t)} == \hat{y}^{(t-1)} \right) \land \left( |\Delta \text{erank}| < \tau_r \right) \right] \lor (t \ge T_{\max})$$

1. **Velocity Decay Signal**:
   $$\frac{v(t)}{v(1)} < \tau_v \quad (\text{Default: } \tau_v = 0.10)$$
   Measures kinetic energy dissipation. Ensures the continuous state has settled near a local minimum.
2. **Coda Discrete Consensus**:
   $$\hat{y}^{(t)} == \hat{y}^{(t-1)}$$
   The discrete solution predicted by the linear readout head remains unchanged between consecutive recurrent passes.
3. **Effective Rank Plateau**:
   $$|\text{erank}(S^{(t)}) - \text{erank}(S^{(t-1)})| < \tau_r \quad (\text{Default: } \tau_r = 0.005)$$
   The spectral Shannon entropy across the singular values of the $M=16$ slots has saturated, indicating that no new orthogonal features are being recruited.

## 2. Tuning for Speed vs. Precision

```python
from parallel_latent_reasoner import GemmaDeliberationPipeline

pipeline = GemmaDeliberationPipeline.from_preset("compact_test")

# AGGRESSIVE EARLY-EXIT (Max speed, for simple triage / classification):
output = pipeline.generate(
    prompt="Classify intent: I want to cancel my account",
    min_steps=1,
    max_steps=4,
    tol_rel_vel=0.20,      # Halts earlier at 80% velocity drop
    tol_erank_delta=0.010,  # Looser rank plateau
)

# CONSERVATIVE DELIBERATION (Max depth, for multi-constraint planning):
output = pipeline.generate(
    prompt="Plan orbital payload under 5 constraints...",
    min_steps=4,
    max_steps=12,
    tol_rel_vel=0.05,      # Requires 95% velocity drop
    tol_erank_delta=0.002,  # Tight rank plateau
    patience=2,             # Requires 2 consecutive consensus steps
)
```
