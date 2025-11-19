# ReDiF: RL-based Distillation of Diffusion Models

This repository contains the official implementation of **ReDiF**, a reinforcement-learning–based framework for few-step distillation of diffusion models. ReDiF treats the student model as a policy in an MDP and optimizes it using RL with perceptual or semantic reward signals—enabling high-quality image generation with far fewer inference steps. this framework implementation is based on the **[ddpo-pytorch](https://github.com/%3Coriginal-author%3E/ddpo-pytorch)** framework.

---

## 🚀 Overview

Diffusion models achieve remarkable image quality but require many sampling steps, making inference slow. Standard distillation approaches often rely on reconstruction losses or consistency constraints, which limit flexibility and require differentiability.

**ReDiF overcomes these limitations** by framing distillation as a reinforcement learning problem, where the student model is trained to maximize semantic alignment with a high-step teacher using flexible reward signals such as CLIP and DINO.

Key advantages:

- Works with **nondifferentiable, sparse, delayed, or perceptual rewards**
- Operates as a **model-agnostic, framework-agnostic optimization layer**
- Compatible with **any UNet-based student** (here: SD v1.5 architecture)
- Achieves strong performance under **extreme step reduction** (e.g., 4–8 steps)

---

## 🧠 Key Features

- RL-based student optimization using PPO/GRPO variants  
- Perceptual reward design using CLIP, DINO, aesthetic scores, and text–image alignment  
- Support for nondifferentiable or delayed rewards without backprop through teacher  
- Easily integrates with existing distillation workflows  

---

## 📁 Repository Structure

```
config/           # Experiment and training configurations
ddpo_pytorch/     # Core RL optimization (modified DDPO implementation)
scripts/          # Training, evaluation, and utility scripts
setup.py          # Installation metadata
LICENSE           # MIT license
README.md         # Project overview
```

---

## 🧩 Installation

```bash
git clone https://github.com/AM-Tighkhorshid/ReDiF.git
cd ReDiF
pip install -e .
```

Ensure that you have PyTorch, HuggingFace Transformers, and other dependencies listed in `setup.py`.

---

## 🔧 Training

To start training with a specific configuration:

```bash
python scripts/train.py --config config/experiment.yaml
```

All model, optimizer, reward, and RL settings are controlled through the YAML config files.

---

## 🏋️‍♂️ Training Details

- Student architecture: **Stable Diffusion v1.5 UNet**
- Training dataset: **10,000 randomly selected COCO images**
- Hardware: **single NVIDIA A100**
- Because of resource limits:
  - Batch size = **8**
  - Gradient accumulation = **2** (effective batch size = 16)
- **All RL-related hyperparameters follow the original DDPO paper**
  - Advantage estimation  
  - Reward normalization  
  - PPO/GRPO clipping  
  - Optimization settings  
  - Rollout generation strategy  
- Divergence-based regularization coefficient = **1.0** when used  
- Student initialization: **behavior cloning** from the fine-tuned teacher  

---

## 📊 Benchmarks & Results

ReDiF achieves strong performance under aggressive step reduction, outperforming or matching baselines such as:

- DDIM / Consistency Distillation  
- Progressive and Adversarial Distillation  
- DMD / DMD2  

ReDiF delivers improved recall and coverage, competitive fidelity, and more robust generation with flexible reward design.  
For full results and analysis, see the accompanying paper.

---

## 📚 Citation

If you use ReDiF in your research, please cite:

```
@article{ReDiF2025,
  title   = {ReDiF: Reinforced Distillation for Few step diffusion},
  author  = {...},
  year    = {2025},
  journal = {...}
}
```

---

## 🤝 Why ReDiF?

ReDiF is ideal if you need:

- A **few-step diffusion student** with better fidelity  
- **Semantic/perceptual reward optimization** instead of L2/consistency losses  
- An optimization layer that is **plug-and-play** with any distillation method  
- Freedom to use **non-differentiable reward signals**  

---

## 📝 Notes & Future Work

- Current version supports image generation; multimodal extensions are straightforward by redefining rewards.  
- RL-based distillation is more compute-intensive than simple consistency distillation but offers greater flexibility.  
- Future improvements may target 2–3 step sampling and integration with quantization or NAS.

---

## 💬 Contact

For questions or issues, please open a GitHub Issue or submit a pull request.

---
