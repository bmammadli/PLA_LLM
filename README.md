# PLA_LLM: Predictive Latent Adapter for LLMs

**Predictive Latent Adapter (PLA)** is a lightweight plug-and-play framework for parameter-efficient adaptation of **frozen Large Language Models (LLMs)** through latent representation refinement.

Instead of updating the full transformer backbone during fine-tuning, PLA learns a compact trainable adapter that refines the model's final hidden representation before prediction. The goal is to achieve effective downstream adaptation while substantially reducing the number of trainable parameters.

---

## Model Overview

<p align="center">
  <img src="architecture.png" alt="PLA architecture" width="100%">
</p>

PLA introduces a lightweight trainable adapter between the frozen transformer backbone and the frozen language-model head.

The pretrained LLM processes the input normally through its sequence of transformer blocks. The final hidden representation produced by the backbone is then intercepted by PLA, refined, and passed to the original frozen prediction head.

The framework consists of five principal stages:

1. **Input processing** — Input tokens are converted into embeddings and processed sequentially through the frozen transformer backbone.
2. **Final hidden representation** — The final transformer output is normalized to obtain the hidden representation $H$ that would normally be passed directly to the language-model head.
3. **Predictive Latent Adapter** — PLA learns a correction to $H$ together with an input-dependent scaling factor and produces a refined representation $H'$.
4. **Prediction** — The refined representation $H'$ is passed to the original frozen language-model head to produce output logits and next-token predictions.
5. **Optimization** — The prediction loss is backpropagated through the frozen prediction head to PLA, while optimizer updates are applied only to the PLA parameters.

---

## Motivation

Modern LLMs may contain billions of parameters. Updating the complete model during fine-tuning can require substantial GPU memory, computation, and training resources.

PLA investigates whether downstream adaptation can instead be achieved by operating directly on the final hidden representation of a pretrained model while leaving the transformer backbone unchanged.

The central hypothesis is:

> Once a pretrained transformer has learned informative internal representations, downstream adaptation may be achieved by learning how its final hidden representation should be corrected before prediction, without updating the transformer backbone itself.

PLA therefore separates the large frozen language model from a compact trainable latent-refinement module.

---

## Mathematical Formulation

Given an input sequence $x$, the frozen transformer backbone $F_{\theta}$ produces its final hidden representation:

```math
H = F_{\theta}(x),
```

where $\theta$ denotes the frozen parameters of the pretrained transformer and

```math
H \in \mathbb{R}^{N \times d}.
```

Here, $N$ is the sequence length and $d$ is the hidden dimension of the model.

The trainable PLA adapter predicts a latent correction:

```math
\Delta H = A_{\phi}(H),
```

where $A_{\phi}$ denotes the adapter network and $\phi$ represents its trainable parameters.

A lightweight gating function produces an input-dependent scaling coefficient:

```math
\alpha(H) \in [0,1].
```

PLA computes the refined hidden representation as:

```math
H' = H + \alpha(H)\odot\Delta H.
```

Substituting the predicted correction gives:

```math
H' = H + \alpha(H)\odot A_{\phi}(H).
```

The dimensionality is preserved:

```math
H' \in \mathbb{R}^{N \times d}.
```

The refined representation is then passed to the original frozen language-model head:

```math
Z = D(H'),
```

where $D(\cdot)$ denotes the frozen language-model head and $Z$ contains the vocabulary logits.

For a vocabulary of size $V$:

```math
Z \in \mathbb{R}^{N \times V}.
```

The corresponding token probabilities are obtained through:

```math
P(y \mid x)
=
\operatorname{softmax}(Z).
```

During PLA training, the transformer backbone and language-model head remain frozen. Only the parameters of the PLA adapter and its gating mechanism are optimized.

---

## Training Objective

For the initial implementation, PLA is trained directly through the downstream prediction objective.

For next-token prediction, the training loss is the standard cross-entropy loss:

```math
\mathcal{L}_{\mathrm{PLA}}
=
\mathcal{L}_{\mathrm{CE}}(\hat{y},y),
```

where $y$ is the target token and $\hat{y}$ represents the model prediction produced from the refined hidden representation $H'$.

The forward computation is therefore:

```math
x
\rightarrow
F_{\theta}(x)
\rightarrow
H
\rightarrow
\mathrm{PLA}_{\phi}(H)
\rightarrow
H'
\rightarrow
D(H')
\rightarrow
Z
\rightarrow
\mathcal{L}_{\mathrm{CE}}.
```

The loss gradient passes through the frozen language-model head to PLA, but optimizer updates are applied only to the PLA parameters:

```math
\theta \; \text{frozen},
\qquad
D \; \text{frozen},
\qquad
\phi \; \text{trainable}.
```

This allows PLA to learn how the final hidden representation should be modified to improve the downstream prediction without changing the pretrained transformer weights.

---

## Parameter-Efficient Adaptation

The hidden representation $H$ is an activation produced during the forward pass. It is **not** itself a collection of trainable parameters.

PLA introduces a comparatively small set of trainable parameters through:

- the adapter network $A_{\phi}$;
- the gating or scaling mechanism $\alpha(H)$.

A compact bottleneck adapter can, for example, be defined schematically as:

```math
A_{\phi}(H)
=
W_{\mathrm{up}}
\,
\sigma
\left(
H W_{\mathrm{down}}
\right),
```

with

```math
W_{\mathrm{down}}
\in
\mathbb{R}^{d \times r},
\qquad
W_{\mathrm{up}}
\in
\mathbb{R}^{r \times d},
\qquad
r \ll d.
```

Here, $d$ is the original LLM hidden dimension and $r$ is a substantially smaller bottleneck dimension.

Ignoring bias terms, the adapter therefore contains approximately:

```math
N_{\mathrm{adapter}}
\approx
2dr
```

trainable parameters.

Including the gating mechanism:

```math
N_{\mathrm{PLA}}
\approx
2dr
+
N_{\mathrm{gate}}.
```

The exact parameter count depends on the final implementation of the adapter and gating network.

In full fine-tuning, the trainable parameter count is approximately the complete model parameter count:

```math
N_{\mathrm{trainable}}^{\mathrm{full}}
\approx
N_{\mathrm{model}}.
```

PLA instead optimizes only:

```math
N_{\mathrm{trainable}}^{\mathrm{PLA}}
=
N_{\mathrm{adapter}}
+
N_{\mathrm{gate}}.
```

The fraction of model parameters optimized during PLA adaptation can therefore be reported as:

```math
\rho_{\mathrm{PLA}}
=
\frac{
N_{\mathrm{trainable}}^{\mathrm{PLA}}
}{
N_{\mathrm{model}}
}
\times 100\%.
```

Because $r \ll d$ and the pretrained transformer remains frozen, PLA is designed to require substantially fewer trainable parameters than full fine-tuning.

The exact trainable-parameter ratio will be reported from the implementation rather than assumed theoretically.

---

## Expected Efficiency Benefits

Freezing the pretrained transformer and optimizing only the PLA module is expected to reduce several costs associated with full fine-tuning, particularly those related to trainable model states.

The framework is designed to provide:

- substantially fewer trainable parameters than full fine-tuning;
- reduced optimizer-state memory;
- reduced parameter-gradient storage;
- preservation of the original pretrained model weights;
- a compact task-specific adapter that can be attached to a frozen backbone.

PLA will also be evaluated for potential reductions in:

- peak GPU-memory consumption;
- training time;
- computational cost;
- energy consumption.

These latter benefits are **experimental questions rather than assumed consequences of parameter reduction**. Quantitative claims will therefore be reported only after controlled benchmarking.

---

## Why PLA?

Many parameter-efficient fine-tuning methods adapt a pretrained model by introducing or optimizing trainable components within its transformer layers.

PLA explores a different intervention point:

> **Instead of modifying the transformer itself, PLA learns how the model's final hidden representation should be refined immediately before prediction.**

The core architecture is:

```text
Input
  ↓
Frozen LLM
  ↓
Final Hidden Representation H
  ↓
PLA Adapter
  ↓
Refined Representation H'
  ↓
Frozen LM Head
  ↓
Prediction
```

The main characteristics of PLA are:

- lightweight plug-and-play design;
- frozen transformer backbone;
- frozen language-model head;
- refinement of the final hidden representation;
- residual latent correction;
- input-dependent correction scaling;
- optimization through the downstream task loss;
- preservation of the pretrained model weights;
- substantially fewer trainable parameters than full fine-tuning.

---

## Initial Experimental Plan

The first public implementation of PLA will use **TinyLlama** as a lightweight proof of concept.

The initial study will compare three adaptation strategies using the same backbone and downstream task:

| Method | Backbone | Trainable component |
|---|---|---|
| Full fine-tuning | TinyLlama | Complete model |
| LoRA | TinyLlama | Low-rank adaptation parameters |
| **PLA** | TinyLlama | Latent adapter and gating mechanism |

The evaluation will focus on:

- downstream task performance;
- number of trainable parameters;
- percentage of trainable parameters;
- peak GPU-memory consumption;
- training time;
- convergence behavior;
- computational cost;
- estimated energy consumption.

The primary question of the initial experiment is:

> **Can refinement of the final hidden representation provide competitive downstream adaptation while training substantially fewer parameters than full fine-tuning?**

The initial objective is not to claim state-of-the-art performance, but to experimentally evaluate the feasibility and efficiency of the proposed adaptation mechanism.

---

## Planned Benchmark Table

Results will be reported in a format similar to the following:

| Method | Trainable Parameters | Trainable Ratio | Peak GPU Memory | Training Time | Task Performance |
|---|---:|---:|---:|---:|---:|
| Full fine-tuning | — | 100% | — | — | — |
| LoRA | — | — | — | — | — |
| **PLA** | — | — | — | — | — |

Exact values will be added after the initial TinyLlama experiments.

---

## Project Status

🚧 **PLA_LLM is currently under active development.**

- ✅ Concept and motivation
- ✅ Revised latent-refinement architecture
- ✅ Mathematical formulation
- 🔄 PyTorch implementation
- 🔄 TinyLlama baseline
- 🔄 LoRA baseline
- 🔄 PLA training experiments
- 🔄 Initial efficiency benchmark

The repository will be updated with source code and experimental results as development progresses.

---

## Citation

A research preprint describing PLA is planned following the initial implementation and benchmark evaluation.

Until then, this repository serves as the public technical description of the proposed method.

---

## License

This project is released under the [MIT License](LICENSE).
