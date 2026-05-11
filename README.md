# ML Upskilling 🚀

This repository is a **living workspace** documenting my journey toward becoming a **strong Machine Learning / ML Systems Engineer**. It combines **theory, implementation, and experimentation** with a focus on *how modern deep learning models are actually built and trained*.

The repo currently includes:

* A clean **code architecture for Vision Transformers (ViT)**
* A **training script** structured like production research code
* A **PyTorch implementation of the forward pass of Flash Attention**

Beyond code, this repository also serves as a **learning log** where I track progress, experiments, and conceptual understanding over time.

---

## 🎯 Goals of This Repository

* Deepen understanding of **modern deep learning architectures**
* Learn **ML engineering best practices** (modularity, reproducibility, scalability)
* Bridge the gap between **research papers and real implementations**
* Build intuition around **performance-critical components** (e.g., attention, memory, compute)
* Create a public record of **consistent ML upskilling**

--- 

## 🧠 What I'm Studying

This repo reflects hands-on learning in areas such as:

* Transformer architectures (ViT, attention mechanisms)
* Training pipelines in PyTorch
* Efficient attention (Flash Attention, memory-aware computation)
* Model initialization, forward passes, and optimization
* Reading and implementing research papers

---

## 📂 Repository Structure

```text
ML-upskilling/
│
├── vit/
│   ├── model.py          # Vision Transformer architecture
│   ├── layers.py         # Transformer blocks, attention, MLPs
│   └── utils.py          # Helper functions
│
├── flash_attention/
│   └── forward.py        # PyTorch implementation of Flash Attention (forward pass)
     ── train.py
|
|__ Paligemma-Pytorch
│
├              # Training script for ViT
├── requirements.txt
└── README.md
```

> **Note:** Folder names may evolve as the repo grows.

---

## 🧩 Vision Transformer (ViT)

The ViT implementation focuses on:

* Patch embedding
* Positional embeddings
* Multi-head self-attention
* Transformer encoder blocks
* Classification head

The architecture is written to be:

* Modular
* Readable
* Easy to extend (e.g., different attention variants, depth, heads)


---

## ⚡ Flash Attention (Forward Pass)

The Flash Attention code explores:

* Memory-efficient attention computation
* Avoiding explicit attention matrix materialization
* Understanding why Flash Attention is faster and more scalable

The goal here is **conceptual clarity**, not just speed.

This implementation helps answer questions like:

* Where does standard attention waste memory?
* How does blocking improve cache efficiency?
* Why does Flash Attention matter for large models?

---

## 🏋️ Training Script

The training script is structured to resemble real-world ML code:

* Clear separation of model, data, and training logic
* Explicit forward/backward passes
* Configurable hyperparameters

---

## PaliGemma 

A compact PyTorch implementation of a PaliGemma-style multimodal inference pipeline. This project combines a SigLIP-inspired vision encoder with a Gemma decoder-only language model to answer text prompts conditioned on an input image.


## 📈 Study Plan & Progress Tracking

I use this repository to:

* Implement ideas from papers I read
* Refactor code as my understanding improves
* Track milestones in my ML learning journey

Future additions may include:

* Experiment logs
* Notes from papers
* Benchmarks and profiling results
* Ablation studies

---

## 🔮 Planned Additions

* Backward pass for Flash Attention
* More transformer variants
* Training optimizations (mixed precision, gradient checkpointing)
* ML systems topics (profiling, memory analysis)
* Notes linking implementations to research papers
* Triton inference server and MLOPs (Mandetory)


---

## 🛠️ Tech Stack

* Python
* PyTorch
* NumPy
* CUDA (conceptually, where relevant)

---

---

## 🤝 Contributions

This is primarily a personal learning repository, but:

* Suggestions
* Discussions
* Improvements

are always welcome via issues or pull requests.

---

## 📬 Contact

If you're also learning ML systems, transformers, or efficient deep learning, feel free to connect.

GitHub: [https://github.com/Tommyhillphyga](https://github.com/Tommyhillphyga)

---

