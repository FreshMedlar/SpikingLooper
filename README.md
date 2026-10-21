# Project Nord v5.0 (Genesis Autogenic)

Nord is a 1.088B parameter autoregressive spiking neural network (SNN) language model. It operates on discrete binary spikes (`{0, 1}`) across discrete timesteps with ~93% activation sparsity, trained from scratch using surrogate-gradient backpropagation.

---

## Architecture

The model processes inputs across $T_{\text{total}} = 10$ timesteps ($T=8$ fast basis steps + $T_{\text{slow}}=2$ slow integration steps) with $d_{\text{model}} = 2048$, 32 attention heads, and a vocabulary of 128,256 tokens.

```
Input Tokens (Batch, Seq_Len)
  │
  ▼
[Temporal Spike Encoder]  ──>  Token embedding + fast/slow basis projections (T=10, d=2048)
  │
  ▼
[Sensory Zone]            ──>  4 blocks: Spiking Synaptic Resonance (LIF Q/K + RoPE) + LIF FFN
  │
  ▼
[Association Zone]        ──>  4 blocks: Spike-Driven MoE (8 experts, Top-2 routing)
  │
  ▼
[Genesis Memory Core]     ──>  Triple persistent LIF banks (Structural 96, Personal 96, Auxiliary 64)
  │                            + Identity GRU + 32-slot Archive Grid
  ▼
[Executive Zone]          ──>  4 blocks: Decision formulation + STDP online plasticity
  │
  ▼
[Readout & LM Head]       ──>  Readout LIF + EMA temporal smoothing (α ≈ 0.80) ──> 128,256 logits
```

### Core Components

1. **Associative LIF Neuron**
   - **Synaptic current**: $I[t] = \beta_{\text{syn}} I[t-1] + x[t]$
   - **Membrane potential**: $V[t] = \beta_{\text{mem}} V[t-1] (1 - S[t-1]) + I[t] + V_{\text{cascade}}[t]$
   - **Spike generation**: $S[t] = \Theta(V[t] - V_{\text{th}})$
   - **Backward pass**: Non-differentiable threshold uses an arc-tangent surrogate gradient:
     $$\frac{\partial S}{\partial V} = \frac{\alpha}{2\pi (1 + (\alpha (V - V_{\text{th}}))^2)}$$
   - **Minicolumn Lateral Excitation**: 128 clusters diffuse lateral excitation ($r=3$), preventing dead neuron regions.
   - **LeakyClamp**: Retains sub-threshold potential floors ($V \ge -0.1$) to stabilize gradient flow.

2. **Spiking Synaptic Resonance**
   - Query ($W_q$) and Key ($W_k$) projections pass through independent LIF neurons.
   - Binary spike activations are aggregated across timesteps with learned softmax temporal mixing weights.
   - Scaled dot-product attention applies causal masking with Rotary Position Embeddings (RoPE).

3. **Spike-Driven Mixture of Experts (MoE)**
   - Located in the Association Zone (8 experts, top-2 active per token).
   - Routing scores are derived directly from the spatial spike rates across minicolumn clusters.

4. **Genesis Memory Core (v5.0)**
   - **Triple Bank**: 256 persistent memory units split into Structural (96), Personal (96), and Auxiliary (64) banks with $\tau_{\text{mem}} = 0.99$.
   - **Purpose Router**: Linear projection with softmax assigns dynamic allocation weights across the three banks.
   - **Identity GRU**: Maintains hidden state continuity across context boundaries.
   - **Archive Grid**: 32 learnable key-value slots for differentiable episodic retrieval.

5. **EMA Temporal Readout**
   - Converts discrete temporal spikes back to token logits:
     $$EMA[t] = \alpha \cdot EMA[t-1] + (1 - \alpha) \cdot V_{\text{mem}}[t]$$
   - The final step fuses smoothed membrane potential with sequence mean spike activity, normalized before projection to vocabulary logits.

---

## Training Process

### 1. Training Setup
- **Data**: FineWeb-Edu and OpenHermes 2.5 tokenized with LLaMA-3 tokenizer (vocab size 128,256).
- **Sequence Length**: 512 tokens.
- **Batching**: Micro-batch size 1 with gradient accumulation 64 (effective batch size 64).
- **Optimizer**: AdamW ($\beta_1 = 0.9, \beta_2 = 0.95$, weight decay $0.01$, gradient clip $1.0$).
- **Learning Rate**: Peak $\text{LR} = 1.0 \times 10^{-4}$ with 2,000-step linear warmup and cosine decay to $1.0 \times 10^{-5}$.
- **Precision**: PyTorch AMP (FP16 on CUDA / BF16 on TPU).

### 2. Loss Formulation
The model is trained end-to-end on a composite loss:

$$\mathcal{L}_{\text{total}} = (1 - \alpha_{\text{KD}}) \mathcal{L}_{\text{CE}} + \alpha_{\text{KD}} \mathcal{L}_{\text{KD}} + \mathcal{L}_{\text{spike}} + 0.01 \mathcal{L}_{\text{MoE}} + \mathcal{L}_{\text{Genesis}}$$

- **$\mathcal{L}_{\text{CE}}$**: Causal cross-entropy for next-token prediction.
- **$\mathcal{L}_{\text{spike}}$**: Penalizes divergence from the biological target firing rate ($\rho_0 = 0.03$):
  $$\mathcal{L}_{\text{spike}} = 0.5 \cdot (\bar{\rho}_{\text{model}} - \rho_0)^2$$
- **$\mathcal{L}_{\text{MoE}}$**: Coefficient of variation balancing loss across experts.
- **$\mathcal{L}_{\text{Genesis}}$**: Regularizes memory bank balance and maximizes entropy over purpose selection and archive retrieval.

### 3. Training Phases
1. **Steps 0 – 22,000 (Pure Spike Pretraining)**:
   - Trained from random initialization using only cross-entropy and spike loss.
   - Loss decreased from 13.40 to 4.57 while maintaining >86% sparsity.
2. **Steps 22,000 – 27,000 (Stabilization & Distillation)**:
   - Phase 1: $\alpha_{\text{KD}} = 0.5$ with teacher model distribution matching via KL divergence ($T=2.0$).
   - Phase 2: $\alpha_{\text{KD}}$ linearly annealed to 0.
   - Phase 3: Final training strictly with cross-entropy and online STDP modulation, converging to loss **4.40**.

---

## Usage

### Installation
```bash
git clone git@github.com:FreshMedlar/SpikingLooper.git
cd SpikingLooper
pip install torch transformers datasets lmdb numpy
```

### Training
```bash
# Download data
python download_data.py

# Train 1.088B model with Genesis v5
python train_nord_kd.py \
    --dataset train_data.jsonl \
    --model_dir checkpoints/nord \
    --preset 1.1b \
    --genesis-v5

# Optional: with knowledge distillation
python train_nord_kd.py \
    --dataset train_data.jsonl \
    --model_dir checkpoints/nord \
    --preset 1.1b \
    --genesis-v5 \
    --kd-teacher gpt2-medium \
    --kd-weight 0.5
```

### Inference
```bash
# Interactive chat
python chat_v4.py

# Low VRAM / CPU offloading
python chat_v4_cpu_offload.py
```

---

## File Structure

```
.
├── nord_core_700m.py        # Model architecture (LIF, MoE, Genesis v5, SSR attention)
├── train_nord_kd.py         # Trainer with optional distillation schedule
├── train_nord_tpu_700m.py   # TPU/CUDA training script with EWC
├── chat_v4.py               # Interactive CLI chat and spike visualization
├── chat_v4_cpu_offload.py   # Inference script with layer-by-layer offloading
├── download_data.py         # Data download and preprocessing script
└── README.md
```

## License
Apache 2.0
