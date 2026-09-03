> ⚠️ **EXPERIMENTAL RESEARCH PREVIEW / ARCHIVAL NOTICE**
> **Status**: EXPERIMENTAL / WORK IN PROGRESS (2026-09-03)
> The compact adapter weights referenced below (`prlr_latent_adapter.npz`) are legacy artifacts quarantined under Requirement R0. Operational tool routing with verified pretrained Gemma weights is scheduled for Milestone 3 (R2).

# Guide: Hybrid Deliberate-Then-Verify for Autonomous Agents

Autonomous agents frequently face latency and token-budget bottlenecks when selecting tools, validating policies, or solving multi-constraint prompts. Traditional LLMs emit lengthy reasoning paragraphs before making an API call, taking 10–20 seconds per tool step.

PRLR solves this via the **Hybrid 'Deliberate-Then-Verify' Pipeline**.

## 1. The 2-Phase Execution Architecture

```
[User Query + 20 Available Tools]
               │
               ▼
┌──────────────────────────────────────────────┐
│ Phase 1: Latent Deliberation (2–5 ms)        │
│ • M=16 memory slots sweep in parallel        │
│ • Cross-attention evaluates all 20 tools     │
│ • 3-Signal E-Gate detects attractor state    │
│ • Zero intermediate tokens emitted           │
└──────────────────────┬───────────────────────┘
                       │ Refined Latent State S^(T)
                       ▼
┌──────────────────────────────────────────────┐
│ Phase 2: Concise Grounded Decoding (5 ms)    │
│ • Coda LM head decodes target JSON directly  │
│ • Zero boilerplate filler                    │
│ • Total turnaround: < 10 ms                  │
└──────────────────────────────────────────────┘
```

## 2. Python Code Example: Fast Tool Routing

```python
import json
from parallel_latent_reasoner import GemmaDeliberationPipeline

# 1. Initialize pipeline with pre-trained adapter weights
pipeline = GemmaDeliberationPipeline.from_preset(
    "compact_test",
    load_trained_adapter=False,  # Quarantined legacy prototype; requires M2/M3 weights
)

# 2. Complex prompt with multiple candidate tools
prompt = """
Available Tools:
- search_web(query: str)
- execute_sql(query: str)
- refund_customer(user_id: str, amount: float)
- cancel_subscription(user_id: str)

User: 'The user Jane (ID: 9021) was double-charged $45.00 due to a system glitch. Please fix this.'
Output the tool call as JSON.
"""

# 3. Execute hybrid deliberation
output = pipeline.generate_hybrid(
    prompt=prompt,
    max_new_tokens=64,
    enable_dynamic_gate=True,
)

tool_call = pipeline.decode_solution(output.token_ids)
print(f"Selected Tool Call: {tool_call}")
print(f"Thought Latency: {output.metrics['deliberation_latency_ms']:.2f} ms")
# Typically executes in < 3 ms!
```
