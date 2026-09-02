# Guide: Fine-Tuning & Distilling Your Own Model

PRLR includes a native MLX Backpropagation Through Time (BPTT) distillation engine (`trainer.py`) that runs directly on Apple Silicon Metal GPUs. It trains the lightweight adapter (Prelude, AdaRMSNorm, and Coda head) to compress intermediate chain-of-thought tokens into continuous vector updates.

## 1. How Distillation Works

Instead of generating 300 sequential thought tokens ($O(N)$ memory reads), the student model learns to update $M=16$ memory slots across $T=8$ sweeps:

$$\mathcal{L} = \mathcal{L}_{\text{CE}}(\text{Answer} \mid S^{(T)}) + \lambda \cdot \mathcal{L}_{\text{align}}(S^{(T)}, h_{\text{teacher}})$$

1. **Teacher**: A frozen autoregressive model (such as Gemma 4 12B) generates the target thought trace and solution.
2. **Student**: The weight-tied PRLR block unrolls in continuous space.
3. **BPTT**: Gradients flow backward through all $T$ unroll steps via MLX autodiff (`nn.value_and_grad`).

## 2. Launching Training

To train on the curated cognitive dataset:

```bash
python3 -c "
import mlx.core as mx
from parallel_latent_reasoner.config import GemmaLatentConfig
from parallel_latent_reasoner.models import MLXCompactGemmaModel
from parallel_latent_reasoner.trainer import DistillationTrainer, DistillationTrainingConfig
from parallel_latent_reasoner.dataset import build_curated_cognitive_dataset

# 1. Config and Model
config = GemmaLatentConfig.compact_test(deliberation_steps=8, num_memory_slots=16)
model = MLXCompactGemmaModel(config)

# 2. Dataset
dataset = build_curated_cognitive_dataset()

# 3. Trainer
train_cfg = DistillationTrainingConfig(
    learning_rate=3e-4,
    weight_decay=0.01,
    batch_size=4,
    epochs=10,
    deliberation_steps=8,
)
trainer = DistillationTrainer(model, train_cfg)
metrics = trainer.train(dataset)

# 4. Save Trained Weights
model.save_adapter_weights('checkpoints/custom_prlr_adapter.npz')
print('Training completed and weights saved!')
"
```

## 3. Key Hyperparameters

- `deliberation_steps`: Set between $4$ and $12$. $T=8$ is optimal for balanced speed and depth.
- `rezero_alpha`: Initialized to $\le 0.05$ to guarantee contractive Lipschitz dynamics.
- `learning_rate`: $1\times 10^{-4}$ to $5\times 10^{-4}$ with AdamW.
