# PLA_LLM: Predictive Latent Adapter for LLMs

**Predictive Latent Adapter (PLA)** is a lightweight plug-and-play framework for adapting **frozen Large Language Models (LLMs)** through latent representation refinement.

Instead of updating the billions of parameters in a pretrained transformer during fine-tuning, PLA keeps the transformer backbone and prediction head frozen and trains only a compact latent-refinement module. By substantially reducing the number of trainable parameters, PLA aims to lower computational cost, GPU-memory requirements, training time, and energy consumption while maintaining competitive downstream performance.

---

## Model Overview

<p align="center">
  <img src="architecture.png" alt="PLA architecture" width="100%">
</p>

PLA introduces a lightweight trainable adapter between the frozen transformer backbone and the frozen prediction head. Rather than modifying the pretrained transformer weights, PLA operates directly on the hidden representations produced by the backbone.

The framework consists of five principal stages:

1. **Input processing** — Input tokens are converted into embeddings and processed by the frozen transformer backbone.
2. **Hidden-state construction** — Hidden representations from selected transformer layers are concatenated to construct a unified latent representation.
3. **Predictive Latent Adapter** — The trainable PLA module predicts a latent correction together with an input-dependent scaling factor.
4. **Prediction** — The refined latent representation is passed to the frozen language-model head.
5. **Optimization** — Gradients update only the PLA parameters, while the transformer backbone and prediction head remain frozen.

---

## Motivation

Modern LLMs may contain billions of parameters. Updating the complete model during fine-tuning can therefore require substantial GPU memory, computational time, and energy.

PLA investigates whether effective adaptation can instead be achieved by optimizing a much smaller module operating on the model's hidden representations.

The central hypothesis is:

> Once a pretrained transformer has learned informative internal representations, downstream adaptation may be achieved by learning how those representations should be corrected without repeatedly updating the entire backbone.

PLA therefore separates the large frozen language model from a compact trainable latent-refinement module.

---

## Mathematical Formulation

Given an input sequence $x$, the frozen transformer backbone produces hidden representations from $L$ selected layers:

```math
H_1, H_2, \ldots, H_L.
```

These representations are concatenated to construct the latent representation supplied to PLA:

```math
H = [H_1; H_2; \cdots; H_L].
```

The trainable adapter predicts a latent correction:

```math
\Delta H = A_{\phi}(H),
```

where $A_{\phi}$ denotes the adapter network and $\phi$ represents its trainable parameters.

A lightweight gating function produces an input-dependent scaling coefficient:

```math
\alpha(H) \in [0,1].
```

PLA then computes the refined latent representation:

```math
H' = H + \alpha(H)\odot\Delta H.
```

Substituting the predicted correction gives:

```math
H' = H + \alpha(H)\odot A_{\phi}(H).
```

The refined representation is passed to the frozen prediction head:

```math
\hat{y} = D(H'),
```

where $D(\cdot)$ denotes the frozen language-model head.

During training, only the parameters of the adapter network $A_{\phi}$ and the gating function $\alpha(H)$ are updated.

---

## Training Objective

PLA is trained by minimizing a combination of the downstream prediction loss and a predictive latent-refinement loss:

```math
\mathcal{L}
=
\lambda_{\mathrm{out}}\mathcal{L}_{\mathrm{out}}
+
\lambda_{\mathrm{pred}}\mathcal{L}_{\mathrm{pred}}.
```

Here:

- $\mathcal{L}_{\mathrm{out}}$ is the downstream task loss, such as cross-entropy for next-token prediction.
- $\mathcal{L}_{\mathrm{pred}}$ encourages the adapter to learn useful latent corrections.
- $\lambda_{\mathrm{out}}$ and $\lambda_{\mathrm{pred}}$ control the relative contribution of the two objectives.

The transformer backbone and prediction head participate in the forward and backward computational graph, but their parameters remain frozen. Optimizer updates are applied only to the PLA module.

---

## Parameter-Efficient Adaptation

The hidden representation $H$ is an activation produced during the forward pass. It is not itself a collection of trainable parameters.

The trainable parameter count of PLA is determined by the architecture of:

- the adapter network $A_{\phi}$;
- the gating or scaling function $\alpha(H)$.

Therefore, the number of PLA parameters does not need to equal the number of values contained in the concatenated hidden representation.

A compact bottleneck adapter can be defined schematically as:

```math
A_{\phi}(H)
=
W_{\mathrm{up}}
\,
\sigma
\left(
W_{\mathrm{down}}H
\right),
```

where:

```math
W_{\mathrm{down}} \in \mathbb{R}^{d \times r},
\qquad
W_{\mathrm{up}} \in \mathbb{R}^{r \times d},
\qquad
r \ll d.
```

Here, $d$ is the latent representation dimension and $r$ is a smaller bottleneck dimension.

For this bottleneck structure, the approximate number of adapter parameters is:

```math
N_{\mathrm{adapter}}
\approx
2dr,
```

excluding bias terms and the parameters of the gating network.

Including the gating network, the approximate PLA parameter count becomes:

```math
N_{\mathrm{PLA}}
\approx
2dr
+
N_{\mathrm{gate}}.
```

Because $r$ can be chosen substantially smaller than $d$, PLA can contain far fewer trainable parameters than the complete transformer backbone.

Full fine-tuning updates all model parameters:

```math
N_{\mathrm{trainable}}^{\mathrm{full}}
=
N_{\mathrm{backbone}}
+
N_{\mathrm{head}}.
```

PLA instead updates only the adapter and gating components:

```math
N_{\mathrm{trainable}}^{\mathrm{PLA}}
=
N_{\mathrm{adapter}}
+
N_{\mathrm{gate}}.
```

The percentage of trainable parameters can be reported as:

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

The exact PLA parameter count and reduction percentage will be reported after the first implementation is finalized.

---

## Expected Efficiency Benefits

By freezing the pretrained backbone and updating only the PLA module, the framework is designed to provide:

- substantially fewer trainable parameters than full fine-tuning;
- reduced optimizer-state memory;
- reduced gradient-storage requirements;
- lower peak GPU-memory consumption;
- shorter adaptation time;
- lower computational and energy cost;
- preservation of the original pretrained model weights.

These are currently design objectives. Quantitative claims will be added after controlled benchmarking.

---

## Why PLA?

Most parameter-efficient fine-tuning methods introduce trainable components within selected transformer layers.

PLA explores a different adaptation mechanism:

> PLA learns how the model's hidden representation should change before prediction while leaving the original transformer parameters unchanged.

The main characteristics of PLA are:

- lightweight plug-and-play design;
- frozen transformer backbone;
- frozen language-model head;
- direct latent-representation refinement;
- input-dependent correction scaling;
- end-to-end optimization through the final task loss;
- compatibility with pretrained transformer models;
- substantially fewer trainable parameters than full fine-tuning.

---

## Initial Experimental Plan

The first public implementation of PLA will use **TinyLlama** as a lightweight proof of concept.

The initial study will compare:

| Method | Backbone | Trainable component |
|---|---|---|
| Full fine-tuning | TinyLlama | Complete model |
| LoRA | TinyLlama | Low-rank adaptation parameters |
| **PLA** | TinyLlama | Latent adapter and gating network |

The evaluation will focus on:

- downstream task performance;
- number of trainable parameters;
- percentage of trainable parameters;
- peak GPU-memory consumption;
- training time;
- convergence behavior;
- computational cost;
- estimated energy efficiency.

The initial objective is not to claim state-of-the-art performance. It is to determine whether PLA can maintain competitive task performance while updating substantially fewer parameters than full fine-tuning.

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
- ✅ Initial architecture
- ✅ Mathematical formulation
- 🔄 PyTorch implementation
- 🔄 TinyLlama baseline
- 🔄 LoRA baseline
- 🔄 PLA training experiments
- 🔄 Initial efficiency benchmark

The repository will be updated with source code and initial experimental results as development progresses.

---

## Citation

A research preprint describing PLA is planned following the initial implementation and benchmark evaluation.

Until then, this repository serves as the public technical description of the proposed method.

---

## License

This project is released under the [MIT License](LICENSE).
