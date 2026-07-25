# PLA_LLM: Predictive Latent Adapter for LLMs

**Predictive Latent Adapter (PLA)** is a lightweight plug-and-play framework for adapting **frozen Large Language Models (LLMs)** through **latent representation refinement**. Instead of updating billions of backbone parameters, PLA learns a compact trainable adapter that predicts how the model's latent representations should be refined to improve downstream predictions while keeping the original backbone and prediction head unchanged.

---

## Model Overview

<p align="center">
  <img src="architecture.png" width="900">
</p>

The proposed PLA framework introduces a lightweight trainable adapter between the frozen transformer backbone and the prediction head.

The workflow consists of five stages:

1. **Input Processing** – Input tokens are embedded and processed by a frozen transformer backbone.
2. **Hidden Representation Construction** – Hidden representations from all transformer layers are concatenated to form a unified latent representation.
3. **Predictive Latent Adapter (PLA)** – A lightweight adapter predicts a latent correction together with an adaptive scaling factor to refine the latent representation.
4. **Prediction** – The refined latent representation is passed to the frozen prediction head to produce the final output.
5. **Training** – Gradients update only the PLA parameters, while both the transformer backbone and prediction head remain frozen.

---

## Mathematical Formulation

Given an input sequence \(x\), the frozen transformer backbone produces hidden representations

\[
H = [H_1; H_2; \cdots; H_L],
\]

where \(H_i\) denotes the hidden representation extracted from the \(i\)-th transformer layer.

PLA predicts a latent correction

\[
\Delta H = A_{\phi}(H),
\]

where \(A_{\phi}\) is the trainable adapter.

An adaptive gating function predicts an input-dependent scaling coefficient

\[
\alpha(H)\in[0,1].
\]

The refined latent representation is computed as

\[
H' = H + \alpha(H)\odot\Delta H,
\]

which is then passed to the frozen prediction head

\[
\hat{y}=D(H').
\]

Only the parameters of the PLA module are optimized during training.

---

## Training Objective

PLA is optimized using a combination of task prediction loss and predictive latent refinement loss

\[
\mathcal{L}
=
\lambda_{\text{out}}\mathcal{L}_{\text{out}}
+
\lambda_{\text{pred}}\mathcal{L}_{\text{pred}},
\]

where

- **Output Loss** (\(\mathcal{L}_{out}\)) supervises the downstream prediction.
- **Predictive Latent Loss** (\(\mathcal{L}_{pred}\)) encourages meaningful latent refinements.

---

## Why PLA?

Unlike conventional parameter-efficient fine-tuning methods that modify internal transformer weights, PLA directly learns **how latent representations should evolve** while leaving the pretrained model unchanged.

Key characteristics include:

- Plug-and-play architecture
- Frozen transformer backbone
- Frozen prediction head
- Lightweight trainable adapter
- Latent representation refinement
- Parameter-efficient adaptation
- End-to-end optimization through task loss

---

## Initial Experimental Plan

The first public implementation of PLA will be evaluated on **TinyLlama** as a lightweight proof-of-concept.

The initial benchmark will compare:

| Method | Model |
|---------|-------|
| Full Fine-Tuning | TinyLlama |
| LoRA | TinyLlama |
| **PLA (Proposed)** | TinyLlama |

The first study will focus on evaluating whether latent representation refinement can reduce training cost while maintaining competitive downstream performance.

The evaluation will include:

- Task performance
- Number of trainable parameters
- GPU memory consumption
- Training time
- Overall training efficiency

---

## Project Status

🚧 **This repository is currently under active development.**

Current progress:

- ✅ Method formulation
- ✅ Architecture design
- ✅ Mathematical formulation
- 🔄 PyTorch implementation
- 🔄 Initial TinyLlama experiments
- 🔄 PLA vs LoRA benchmark

The repository will be continuously updated with implementation details, experimental results, and benchmark comparisons.
