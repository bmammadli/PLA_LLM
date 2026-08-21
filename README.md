# PLA_LLM: Predictive Latent Adapter for LLMs

**Predictive Latent Adapter (PLA)** is a lightweight plug-and-play framework for adapting **frozen Large Language Models (LLMs)** by refining the model's final hidden representation before prediction.

Instead of updating the billions of parameters in a pretrained transformer during fine-tuning, PLA keeps the transformer backbone and language-model head frozen and trains only a compact latent-refinement module. By substantially reducing the number of trainable parameters, PLA aims to lower computational cost, GPU-memory requirements, training time, and energy consumption while maintaining competitive downstream performance.

---

## Model Overview

<p align="center">
  <img src="architecture.png" alt="PLA architecture" width="100%">
</p>

PLA introduces a lightweight trainable adapter between the **final hidden representation of the frozen transformer** and the **frozen language-model head**.

Rather than modifying the pretrained transformer weights, PLA operates directly on the representation that would normally be passed from the final transformer block to the language-model head.

The framework consists of five principal stages:

1. **Input processing** — Input tokens are converted into embeddings and processed through the frozen transformer backbone.
2. **Final hidden representation** — The final transformer block produces the hidden representation used by the language-model head.
3. **Predictive Latent Adapter** — PLA learns a residual correction to this representation together with an input-dependent scaling factor.
4. **Prediction** — The refined representation is passed to the original frozen language-model head to produce vocabulary logits and next-token probabilities.
5. **Optimization** — Gradients optimize only the PLA parameters, while the transformer backbone and language-model head remain frozen.

---

## Motivation

Modern LLMs may contain billions of parameters. Updating the complete model during fine-tuning can therefore require substantial GPU memory, computational time, and energy.

PLA investigates whether effective downstream adaptation can instead be achieved by optimizing a much smaller module operating directly on the model's final hidden representation.

The central hypothesis is:

> Once a pretrained transformer has learned informative internal representations, downstream adaptation may be achieved by learning how its final representation should be corrected before prediction, without updating the pretrained transformer itself.

PLA therefore separates the large frozen language model from a compact trainable latent-refinement module.

---

## Mathematical Formulation

Given an input sequence $x$ containing $N$ tokens, the frozen transformer backbone produces the final hidden representation:

```math
H = F_{\theta}(x),
```

where $F_{\theta}$ denotes the pretrained transformer backbone with frozen parameters $\theta$.

For hidden dimension $d$:

```math
H \in \mathbb{R}^{N \times d}.
```

In a standard causal language model, this final hidden representation would normally be passed directly to the language-model head.

PLA instead introduces a trainable transformation before that prediction step.

The adapter predicts a latent correction:

```math
\Delta H = A_{\phi}(H),
```

where $A_{\phi}$ denotes the trainable adapter network and $\phi$ represents its parameters.

A lightweight gating mechanism produces an input-dependent scaling coefficient:

```math
\alpha(H) \in [0,1].
```

PLA computes the refined hidden representation as:

```math
H'
=
H
+
\alpha(H)\odot\Delta H.
```

Substituting the predicted correction gives:

```math
H'
=
H
+
\alpha(H)\odot A_{\phi}(H).
```

The residual connection preserves the original representation while allowing PLA to learn a controlled correction before prediction.

The refined representation is then passed to the frozen language-model head:

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
\mathrm{softmax}(Z).
```

During PLA training, the transformer backbone and language-model head remain frozen. Only the parameters of the PLA adapter and its gating mechanism are optimized.

---

## Training Objective

The primary training signal is the downstream language-modeling objective.

For next-token prediction, the output loss can be written as:

```math
\mathcal{L}_{\mathrm{out}}
=
-\sum_{t}
\log
P(y_t \mid x_{\leq t}).
```

Because the prediction depends on the refined representation $H'$, gradients from the output loss propagate through the frozen language-model head to the PLA module.

The optimization problem is therefore:

```math
\phi^{*}
=
\underset{\phi}{\mathrm{argmin}}
\;
\mathcal{L}_{\mathrm{out}}.
```

The pretrained transformer parameters remain fixed:

```math
\theta
=
\mathrm{constant}.
```

The parameters of the frozen language-model head also remain unchanged.

An optional regularization term may be introduced to control the magnitude or structure of the latent correction:

```math
\mathcal{L}
=
\mathcal{L}_{\mathrm{out}}
+
\lambda_{\mathrm{reg}}
\mathcal{L}_{\mathrm{reg}}.
```

For example, a simple correction regularizer may penalize unnecessarily large changes to the original hidden representation:

```math
\mathcal{L}_{\mathrm{reg}}
=
\lVert H' - H \rVert_2^2.
```

This formulation encourages PLA to improve task predictions while keeping the learned correction compact when appropriate.

The exact training objective and regularization strategy will be evaluated experimentally.

---

## Parameter-Efficient Adaptation

The hidden representation $H$ is an activation generated during the forward pass. It is **not itself a collection of trainable parameters**.

The number of trainable PLA parameters is determined by the architecture of:

- the adapter network $A_{\phi}$;
- the gating or scaling mechanism $\alpha(H)$.

A compact bottleneck adapter can be defined schematically as:

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

where:

```math
W_{\mathrm{down}}
\in
\mathbb{R}^{d \times r},
```

and

```math
W_{\mathrm{up}}
\in
\mathbb{R}^{r \times d},
```

with:

```math
r \ll d.
```

Here, $d$ is the hidden dimension of the language model and $r$ is a substantially smaller bottleneck dimension.

Ignoring bias terms, the approximate number of adapter parameters is:

```math
N_{\mathrm{adapter}}
\approx
2dr.
```

Including the gating mechanism, the approximate PLA parameter count becomes:

```math
N_{\mathrm{PLA}}
\approx
2dr
+
N_{\mathrm{gate}}.
```

Because $r$ can be chosen substantially smaller than $d$, the PLA module can contain far fewer trainable parameters than the complete transformer.

For full fine-tuning:

```math
N_{\mathrm{trainable}}^{\mathrm{full}}
=
N_{\mathrm{total}}.
```

For PLA:

```math
N_{\mathrm{trainable}}^{\mathrm{PLA}}
=
N_{\mathrm{adapter}}
+
N_{\mathrm{gate}}.
```

The fraction of trainable parameters can therefore be reported as:

```math
\rho_{\mathrm{PLA}}
=
\frac{
N_{\mathrm{trainable}}^{\mathrm{PLA}}
}{
N_{\mathrm{total}}
}
\times 100\%.
```

The exact PLA parameter count depends on the backbone hidden dimension, bottleneck size, gating architecture, and implementation choices.

These values will be reported experimentally rather than assumed in advance.

---

## Expected Efficiency Benefits

By freezing the pretrained transformer and language-model head and optimizing only the PLA module, the framework is designed to provide:

- substantially fewer trainable parameters than full fine-tuning;
- reduced optimizer-state memory;
- reduced parameter-gradient storage;
- lower training-memory requirements;
- shorter adaptation time;
- lower computational cost;
- lower energy consumption;
- preservation of the original pretrained model weights;
- a compact task-specific module that can be attached to a frozen backbone.

It is important to distinguish **parameter efficiency** from total computational cost.

The frozen transformer must still execute its forward pass to produce $H$, and gradients must propagate through the downstream computation required to optimize PLA. Therefore, freezing the backbone does not eliminate the computational cost of running the LLM.

The expected efficiency gains arise primarily from avoiding optimization of the backbone's large parameter set and reducing trainable-parameter, optimizer-state, and gradient-storage requirements.

Quantitative claims will be added after controlled benchmarking.

---

## Why PLA?

Many parameter-efficient fine-tuning methods adapt pretrained models by introducing or optimizing trainable components within selected transformer layers.

PLA explores a different intervention point:

> **Instead of modifying the transformer itself, PLA learns how the model's final hidden representation should be refined immediately before prediction.**

The main characteristics of PLA are:

- lightweight plug-and-play design;
- frozen transformer backbone;
- frozen language-model head;
- refinement of the final hidden representation;
- residual latent correction;
- input-dependent correction scaling;
- optimization through the downstream task loss;
- preservation of the original pretrained model weights;
- substantially fewer trainable parameters than full fine-tuning.

This provides a simple research question:

> **How much downstream adaptation can be achieved by modifying only the representation immediately before the language-model head?**

---

## Initial Experimental Plan

The first public implementation of PLA will use **TinyLlama** as a lightweight proof of concept.

The initial study will compare:

| Method | Backbone | Trainable Component |
|---|---|---|
| Full fine-tuning | TinyLlama | Complete model |
| LoRA | TinyLlama | Low-rank adaptation parameters |
| **PLA** | TinyLlama | Final-representation adapter and gating mechanism |

The evaluation will focus on:

- downstream task performance;
- number of trainable parameters;
- percentage of trainable parameters;
- peak GPU-memory consumption;
- training time;
- convergence behavior;
- computational cost;
- estimated energy consumption.

Particular attention will be given to the trade-off between:

```math
\mathrm{Task\ Performance}
\quad\mathrm{vs.}\quad
\mathrm{Adaptation\ Cost}.
```

The initial objective is not to claim state-of-the-art performance.

Instead, the experiments will determine whether refining only the final hidden representation can provide competitive downstream adaptation while requiring substantially fewer trainable parameters than full fine-tuning.

---

## Planned Benchmark

The initial benchmark will compare PLA against full fine-tuning and LoRA under the same backbone, dataset, and evaluation conditions.

| Method | Trainable Parameters | Trainable Ratio | Peak GPU Memory | Training Time | Task Performance |
|---|---:|---:|---:|---:|---:|
| Full fine-tuning | — | 100% | — | — | — |
| LoRA | — | — | — | — | — |
| **PLA** | — | — | — | — | — |

Additional efficiency metrics may include:

- GPU-hours;
- number of optimizer parameters;
- training throughput;
- memory usage;
- estimated energy consumption;
- performance per trainable parameter.

Exact values will be added after the initial TinyLlama experiments.

---

## Project Status

🚧 **PLA_LLM is currently under active development.**

- ✅ Concept and motivation
- ✅ Initial PLA architecture
- ✅ Mathematical formulation
- 🔄 PyTorch implementation
- 🔄 TinyLlama baseline
- 🔄 LoRA baseline
- 🔄 PLA training experiments
- 🔄 Efficiency benchmark

The repository will be updated with source code, implementation details, and experimental results as development progresses.

---

## Research Direction

The initial implementation focuses deliberately on the simplest PLA configuration: **refining the final hidden representation immediately before the language-model head**.

The first experiments will investigate:

1. whether this representation contains sufficient information for effective downstream adaptation;
2. how much performance can be recovered without modifying the transformer backbone;
3. how the PLA bottleneck dimension affects performance and parameter count;
4. whether learned gating improves over an ungated residual adapter;
5. how PLA compares with LoRA and full fine-tuning in performance, memory, training time, and computational cost.

These experiments will determine whether more advanced PLA variants are warranted.

---

## Citation

A research preprint describing PLA is planned following the initial implementation and benchmark evaluation.

Until then, this repository serves as the public technical description of the proposed method.

---

## License

This project is released under the [MIT License](LICENSE).
