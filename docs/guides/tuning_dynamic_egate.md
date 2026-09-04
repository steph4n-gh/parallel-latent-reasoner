# Guide: Tuning the 4-Signal Calibrated Dynamic Consensus E-Gate

The **4-Signal Calibrated Dynamic Consensus E-Gate** monitors continuous latent trajectories in real-time and halts computation as soon as representations stabilize across both geometric and semantic dimensions, saving recurrent compute while retaining full accuracy.

## 1. The 4 Calibrated Consensus Signals

$$\text{Halt}(t) = (t \ge T_{\min}) \land \left[ \left( v(t) < \tau_v \right) \land \left( H(t) < \tau_e \right) \land \left( m(t) > \tau_m \right) \land \left( |\Delta \text{erank}(t)| < \tau_r \right) \right] \lor (t \ge T_{\max})$$

1. **Kinetic State Velocity Decay**:
   $$v(t) = \frac{\|S^{(t)} - S^{(t-1)}\|_F}{\max(\|S^{(1)} - S^{(0)}\|_F, 10^{-6})} < \tau_v \quad (\text{Calibrated: } \tau_v = 0.98)$$
   Measures kinetic energy dissipation in working memory slot representations.
2. **First-Token Prediction Entropy**:
   $$H(t) = -\sum_{i} p_i \ln p_i < \tau_e \quad (\text{Calibrated: } \tau_e = 0.65\text{ nats})$$
   Monitors the information-theoretic uncertainty of top logits directly from latent probe activations.
3. **Top-1 vs Top-2 Decision Margin**:
   $$m(t) = z_{(1)} - z_{(2)} > \tau_m \quad (\text{Calibrated: } \tau_m = 2.80)$$
   Ensures unambiguous hypothesis separation before exiting to causal decode.
4. **Gram Effective Rank Plateau**:
   $$\Delta r(t) = |\text{erank}(S^{(t)}) - \text{erank}(S^{(t-1)})| < \tau_r \quad (\text{Calibrated: } \tau_r = 0.006)$$
   Verifies that the effective rank across memory slots has stabilized, indicating feature recruitment saturation.

## 2. Python API Usage

```python
from prlr.pipeline import PRLRPipeline

# Initialize production pipeline with verified pretrained weights
pipeline = PRLRPipeline(
    deliberation_steps=4,
    num_slots=16,
)

# Execute inference with calibrated 4-signal consensus E-gate
result = pipeline.deliberate_and_verify(
    prompt="Route request: customer wants refund for order 4201",
    max_steps=12,
    max_new_tokens=64,
    enable_dynamic_gate=True,
)

print(f"Decoded Action    : {result.decoded_text}")
print(f"Executed Steps    : {result.deliberation_steps}")
print(f"Exit Verdict      : {result.egate_verdict}")
print(f"Shannon Entropy   : {result.shannon_entropy:.2f} bits")
```
