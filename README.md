# PLA_LLM: Predictive Latent Adapter for LLMs

**Predictive Latent Adapter (PLA)** is a lightweight plug-and-play framework for adapting **frozen Large Language Models (LLMs)** through **latent representation refinement**. Rather than updating billions of backbone parameters, PLA learns a compact trainable adapter that predicts how latent representations should be refined while keeping both the pretrained transformer backbone and prediction head unchanged.

---

## Model Overview

<p align="center">
  <img src="architecture.png" width="100%">
</p>

PLA introduces a lightweight trainable adapter between a frozen transformer backbone and the prediction head. Instead of modifying the transformer weights, PLA operates directly on the latent representations produced by the backbone.

The workflow consists of five stages:

1. **Input Processing** – Input tokens are embedded and processed by a frozen transformer backbone.
2. **Hidden Representation Construction** – Hidden representations from all transformer layers are concatenated to construct a unified latent representation.
3. **Predictive Latent Adapter (PLA)** – A lightweight adapter predicts a latent correction together with an adaptive scaling factor to refine the latent representation.
4. **Prediction** – The refined latent representation is passed through the frozen prediction head to produce the final prediction.
5. **Training** – During optimization, gradients update only the PLA parameters while both the transformer backbone and prediction head remain frozen.

---

# Mathematical Formulation

Given an input sequence $x$, the frozen transformer backbone produces hidden representations from all transformer layers

$$
H = [H_1; H_2; \cdots; H_L],
$$

where $H_i$ denotes the hidden representation extracted from the $i$-th transformer layer.

The PLA module predicts a latent correction

$$
\Delta H = A_{\phi}(H),
$$

where $A_{\phi}$ is the trainable adapter network.

An adaptive gating function predicts an input-dependent scaling coefficient

$$
\alpha(H)\in[0,1].
$$

The refined latent representation is computed as

$$
H' = H + \alpha(H)\odot\Delta H,
$$

which is subsequently passed to the frozen prediction head

$$
\hat{y}=D(H').
$$

Only the parameters of the PLA module are optimized during training.

---

# Training Objective

PLA is optimized using a combination of the downstream prediction loss and a predictive latent refinement loss

$$
\mathcal{L}
=
\lambda_{\mathrm{out}}\mathcal{L}_{\mathrm{out}}
+
\lambda_{\mathrm{pred}}\mathcal{L}_{\mathrm{pred}}.
$$

where

- **Output Loss** ($\mathcal{L}_{\mathrm{out}}$) supervises the downstream prediction.
- **Predictive Latent Loss** ($\mathcal{L}_{\mathrm{pred}}$) encourages meaningful latent refinements while preserving useful information.

---

# Why PLA?

Conventional parameter-efficient fine-tuning methods (e.g., LoRA) adapt large language models by modifying internal transformer parameters through additional trainable weights.

PLA explores an alternative adaptation strategy by **learning how latent representations should be refined**, while leaving the pretrained transformer entirely unchanged.

Key characteristics include:

- Lightweight plug-and-play architecture
- Frozen transformer backbone
- Frozen prediction head
- Latent representation refinement
- Parameter-efficient adaptation
- End-to-end optimization through task supervision
- Compatible with existing pretrained LLMs

---

# Initial Experimental Plan

The first implementation of PLA will be evaluated using **TinyLlama** as a lightweight proof-of-concept.

The initial benchmark will compare:

| Method | Model |
|---------|-------|
| Full Fine-Tuning | TinyLlama |
| LoRA | TinyLlama |
| **PLA (Proposed)** | TinyLlama |

The primary objective of the initial study is to investigate whether latent representation refinement can reduce training cost while maintaining competitive downstream performance.

The evaluation will include:

- Downstream task performance
- Number of trainable parameters
- GPU memory consumption
- Training time
- Overall training efficiency

Following the initial proof-of-concept, PLA will be extended to larger open-source language models.

---

# Project Status

🚧 **PLA_LLM is currently under active development.**

Current progress:

- ✅ Method formulation
- ✅ Architecture design
- ✅ Mathematical formulation
- 🔄 PyTorch implementation
- 🔄 Initial TinyLlama experiments
- 🔄 LoRA comparison
- 🔄 Initial benchmark results

This repository will be continuously updated as implementation progresses, including source code, benchmark results, experimental analysis, and documentation.

---

# Citation

If you find this project useful, please consider starring the repository.

A research preprint describing PLA is planned for future release.

---

# License

This project is released under the MIT License.
