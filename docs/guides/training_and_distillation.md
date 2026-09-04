# Guide: Fine-Tuning & Distilling Recurrent Adapters (BPTT)

PRLR provides production MLX Backpropagation Through Time (BPTT) training runners (`train_gemma_adapter.py` and `train_gemma4_adapter.py`) that run natively on Apple Silicon Metal GPUs. They train the weight-tied recurrent adapter (prelude projections, slot reasoning blocks, dedicated cross-attention, and causal prefix decoder) on solver-backed procedural datasets using masked answer cross-entropy.

## 1. How BPTT Training Operates

Instead of generating lengthy intermediate tokens, the adapter learns to iteratively update $M=16$ continuous memory slots across $T=4$ deliberation sweeps:

$$\mathcal{L} = -\frac{1}{|M_{\text{target}}|} \sum_{i \in M_{\text{target}}} \log P(y_i \mid y_{<i}, S^{(T)}, H_{\text{prompt}})$$

1. **Backbone**: Official frozen Google Gemma backbone (`google/gemma-2b-it` or `google-gemma-4-12B-it-4bit`) extracts contextual representations $H_{\text{prompt}}$.
2. **Adapter**: $M=16$ working memory slots update via parallel Jacobi sweeps with bounded sigmoidal residual scaling: $\alpha = \alpha_{\max} \cdot \sigma(\text{raw}\_\alpha)$.
3. **Loss Masking**: Masked cross-entropy applies strictly to target solution tokens ($M_{\text{target}}$), with zero loss computed over prompts or padding.
4. **BPTT Autodiff**: Gradients flow backward through all $T$ unroll steps via MLX `nn.value_and_grad`.

## 2. Launching Training

### Option A: Gemma 2B Recurrent Adapter
```bash
python3 train_gemma_adapter.py \
    --data-path data/prlr_domain_v1/train.jsonl \
    --epochs 8 \
    --lr 3e-4 \
    --target-loss 0.15 \
    --save-name gemma_2b_prlr_adapter.safetensors
```

### Option B: Gemma 4 12B Recurrent Adapter
```bash
python3 train_gemma4_adapter.py \
    --data-path data/prlr_domain_v1/train.jsonl \
    --epochs 4 \
    --batch-size 1 \
    --lr 4e-4 \
    --target-loss 0.08 \
    --save-name gemma_4_12b_prlr_adapter.safetensors
```

Checkpoints are serialized to `.safetensors` alongside a cryptographic SHA-256 JSON sidecar registering full provenance, hyperparameter geometry, and training metrics per Rule 10.

## 3. Key Hyperparameters

- `deliberation_steps`: Set between $2$ and $8$ ($T=4$ default for stable convergence).
- `residual_bounding`: Bounded sigmoidal scaling with $\alpha_{\max} = 0.5$ and ReZero initialization $\alpha \le 0.05$.
- `learning_rate`: $1\times 10^{-4}$ to $4\times 10^{-4}$ with AdamW and linear warmup + cosine decay.
