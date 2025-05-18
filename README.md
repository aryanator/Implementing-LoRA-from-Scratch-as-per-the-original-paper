# LoRA Fine-Tuning on BERT (SST-2)

This repository demonstrates how to apply **Low-Rank Adaptation (LoRA)** to the self-attention layers of a pre-trained BERT model for downstream classification. We follow the methodology outlined in:

> **LoRA: Low-Rank Adaptation of Large Language Models**  
> Edward Hu, Yelong Shen, et al. (Microsoft Research)  
> [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)

### 📌 Summary

We inject trainable low-rank matrices (`A`, `B`) into the query and value projection layers of `bert-base-uncased`, freeze the original model weights, and fine-tune only the LoRA adapters and a classification head on the **GLUE SST-2** sentiment classification task.

---

### 🧪 Training Setup

- Dataset: SST-2 (binary sentiment classification)
- Model: `bert-base-uncased` with LoRA adapters (r=4, α=10)
- Optimizer: AdamW (`lr=2e-5`)
- Loss: CrossEntropy
- Epochs: 3
- Hardware: CPU/GPU (CUDA supported)

---

### 📦 Requirements

```bash
pip install torch transformers datasets
