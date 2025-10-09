# RL-based Defusion Distillation

This repository presents an framework for accelerating diffusion model through RL-based distillation. This is a customized implementation of the **[ddpo-pytorch](https://github.com/%3Coriginal-author%3E/ddpo-pytorch)** framework.

The modifications focus on advancing research in **diffusion-based policy optimization**, **text–image aesthetic alignment**, and **reinforcement learning with generative models**.

---

## 🧩 Overview

This work builds upon the foundational implementation of *Diffusion-Driven Policy Optimization (DDPO)*, introducing modifications to the training pipeline, configuration structure, and evaluation scripts.
The primary goals of this repository are to:

* Facilitate controlled experimentation with DDPO and PPO variants.
* Enable multimodal and aesthetic alignment tasks (e.g., CLIP-guided objectives).
* Provide a reproducible framework for research in reinforcement learning from synthetic feedback.

---

## 📁 Repository Structure

```
config/           # Experiment and model configuration files
ddpo_pytorch/     # Core modified implementation of DDPO
scripts/          # Custom training, evaluation, and analysis scripts
setup.py          # Installation configuration
LICENSE           # License information
```

---

## ⚙️ Installation

To install dependencies and prepare the environment:

```bash
pip install -e .
```

Ensure that compatible versions of `torch`, `transformers`, and related dependencies are installed according to your experimental requirements.

---

## 🚀 Usage Example

Run training with a specified configuration:

```bash
python scripts/train.py --config config/experiment.yaml
```

All hyperparameters, logging options, and model configurations are defined in the corresponding YAML files under the `config/` directory.

---

## 🧠 Attribution

This repository is **derived from and extends**
the open-source project **[ddpo-pytorch](https://github.com/%3Coriginal-author%3E/ddpo-pytorch)**.

The original authors are gratefully acknowledged for their valuable contribution to the open research community.
Substantial modifications, enhancements, and new experimental components have been introduced by the current author to support additional research objectives.

Please cite the original `ddpo-pytorch` work when using or referencing this repository in academic publications.

---

## 📜 License

This project retains the original license terms from `ddpo-pytorch` (see `LICENSE` file).
Users are encouraged to review that file for details regarding usage, modification, and redistribution rights.

---
