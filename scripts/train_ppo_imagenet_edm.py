"""
ReDiF -- Reinforced Distillation for Few-step Diffusion.
EDM (Karras et al., 2022) teacher on class-conditional ImageNet 64x64.

This is the EDM/ImageNet counterpart of `train_ppo_coco.py` (Stable Diffusion / COCO).
Key differences w.r.t. the SD version:

  * Conditioning is a *class label* (one-hot, 1000 ImageNet classes) instead of a text prompt,
    so the "prompt pool" becomes a fixed pool of (initial noise, class label) pairs.
  * The sampler is EDM's sigma-parameterised sampler, not DDIM. The stochastic policy is an
    *ancestral* Euler step (k-diffusion style), which is the exact analogue of DDIM with eta > 0:
    the Gaussian mean depends on the student network, hence log-probabilities are differentiable.
  * The teacher is deterministic (Heun, 2nd order). Because the teacher image only depends on
    (z_T, class), all teacher targets are precomputed ONCE and cached on disk. During RL training
    the teacher is never called again -> huge speed-up compared to the SD/COCO setting.
  * Reward = cosine similarity between semantic embeddings (DINOv2 / CLIP) of the student image
    and the cached teacher image for the *same* (z_T, class) pair. Optional KL regularisation
    towards a frozen reference policy (the behaviour-cloned student init = teacher weights).

Requirements
------------
  * NVIDIA EDM repo on PYTHONPATH (for `dnnlib` and `torch_utils`):
        git clone https://github.com/NVlabs/edm && export PYTHONPATH=$PYTHONPATH:$(pwd)/edm
  * Teacher checkpoint (class-conditional, 296M params, ADM architecture):
        https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl
  * torch, torchvision, transformers (CLIP), optional: wandb.

Typical usage (single A100 80GB)
--------------------------------
  # 1) Precompute the teacher targets once (~1000 pairs, 256-step Heun = 511 NFE, the setting
  #    recommended by the EDM paper for ImageNet-64):
  python train_ppo_imagenet_edm.py --precompute_only --num_pairs 1000 --teacher_steps 256

  # 2) RL distillation, 4-step and 8-step students back to back (same pool, same cached targets):
  python train_ppo_imagenet_edm.py --student_steps 4,8 --reward_fn clip+dino --kl_coeff 1.0

Outputs land in <output_dir>/steps4/ and <output_dir>/steps8/ (checkpoints, sample grids, log.jsonl).
"""

import argparse
import contextlib
import copy
import json
import math
import os
import pickle
import random
import time
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn.functional as F

# -----------------------------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()

    # --- models -------------------------------------------------------------------------------
    p.add_argument("--teacher_pkl", type=str,
                   default="https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl",
                   help="EDM pickle (local path or URL). ImageNet-64 checkpoint is class-conditional only.")
    p.add_argument("--ckpt_dir", type=str, default="./checkpoints",
                   help="Where the teacher pickle is downloaded to if --teacher_pkl is a URL.")
    p.add_argument("--edm_repo", type=str, default=None,
                   help="Path to a local clone of https://github.com/NVlabs/edm. Required unless the repo "
                        "is already on PYTHONPATH: the checkpoint is a torch_utils.persistence pickle and "
                        "cannot be unpickled without it.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--teacher_fp16", type=int, default=1,
                   help="Run the teacher trunk in fp16 during target precomputation (EDM's own mixed "
                        "precision, the same setting NVlabs uses for ImageNet-64 sampling). ~2x faster "
                        "over 511 NFE x num_pairs. Set 0 for a strictly fp32 reference.")
    p.add_argument("--student_fp16", type=int, default=0,
                   help="Run the student trunk in fp16 during rollout and PPO updates. Parameters stay "
                        "fp32, but there is no GradScaler here, so keep 0 unless memory forces otherwise.")

    # --- sampling schedule --------------------------------------------------------------------
    p.add_argument("--student_steps", type=str, default="4,8",
                   help="Comma-separated list of student step counts. Each entry is trained as a separate "
                        "run (fresh behaviour-cloning init) but shares the same (z_T, class) pool and the "
                        "same cached teacher targets, so the 4-step vs 8-step ablation is exactly matched.")
    p.add_argument("--teacher_steps", type=int, default=256, help="Teacher Heun steps (NFE = 2*steps-1).")
    p.add_argument("--sigma_min", type=float, default=0.002)
    p.add_argument("--sigma_max", type=float, default=80.0)
    p.add_argument("--rho", type=float, default=7.0)
    p.add_argument("--eta", type=float, default=1.0,
                   help="Exploration level of the ancestral step (analogue of DDIM eta). eta=0 is fully "
                        "deterministic => zero-variance policy => no policy gradient at all. eta=1 is the "
                        "fully ancestral (DDPM-like) step, which with only 4 steps throws away almost all of "
                        "the deterministic ODE direction (sigma_down ~ 0) and is far too noisy. 0.2-0.4 keeps "
                        "the trajectory close to the ODE path while leaving enough exploration for PPO.")

    # --- (noise, class) pair pool -------------------------------------------------------------
    p.add_argument("--num_pairs", type=int, default=1000, help="Size of the fixed (z_T, class) pool.")
    p.add_argument("--num_classes_used", type=int, default=1000,
                   help="Restrict the pool to the first K ImageNet classes (K=1000 => all).")
    p.add_argument("--pool_seed", type=int, default=1234)
    p.add_argument("--active_pairs", type=int, default=0,
                   help="Sample rollouts only from the first N pairs of the pool (0 = all). The pool "
                        "and its cached teacher images are unchanged, so this costs nothing. With a "
                        "10k pool and 256 samples/epoch each pair is revisited once every 40 epochs, "
                        "which is why per-pair statistics never accumulate; DDPO works with under a "
                        "thousand prompts it returns to constantly.")
    p.add_argument("--teacher_cache", type=str, default="./teacher_cache_in64.pt",
                   help="Resolved relative to the CURRENT WORKING DIRECTORY, not to this script.")
    p.add_argument("--overwrite_cache", type=int, default=0,
                   help="0 = if an existing --teacher_cache belongs to a different pool, write the "
                        "newly generated one to a suffixed filename instead of replacing it. The old "
                        "cache is usually still needed for evaluation.")
    p.add_argument("--reuse_cache_pool", type=int, default=1,
                   help="If an existing cache has the same --pool_seed but a different --num_pairs, "
                        "adopt the cache's size instead of regenerating everything.")
    p.add_argument("--online_teacher", type=int, default=0,
                   help="1 = generate the teacher target inside every rollout batch from fresh noise "
                        "(like the SD/COCO script). Since the teacher sampler is deterministic, its output "
                        "depends only on (z_T, class), so this recomputes identical targets whenever a pair "
                        "repeats and costs ~25x more teacher NFE than caching. Prefer raising --num_pairs. "
                        "Only useful as an ablation, and then lower --teacher_steps (e.g. 32).")
    p.add_argument("--precompute_only", action="store_true")
    p.add_argument("--precompute_batch", type=int, default=64)

    # --- RL / PPO -----------------------------------------------------------------------------
    p.add_argument("--num_epochs", type=int, default=20)
    p.add_argument("--sample_batch_size", type=int, default=16,
                   help="Rollout batch size (no grad). Must be a multiple of --group_size.")
    p.add_argument("--num_batches_per_epoch", type=int, default=4,
                   help="Rollout batches gathered per epoch; samples_per_epoch = this * sample_batch_size.")
    p.add_argument("--train_batch_size", type=int, default=8, help="PPO minibatch, in (sample, timestep) pairs.")
    p.add_argument("--grad_accum", type=int, default=1, help="Effective batch = train_batch_size * grad_accum.")
    p.add_argument("--inner_epochs", type=int, default=1,
                   help="PPO passes over the same rollout buffer. >1 is what makes the clipped surrogate "
                        "meaningful, since on the first pass the ratio is 1 by construction.")
    p.add_argument("--timestep_fraction", type=float, default=1.0,
                   help="Fraction of the student's timesteps trained on per inner epoch.")
    p.add_argument("--clip_range", type=float, default=1e-4,
                   help="PPO clip range. Small because log-probs are averaged over pixels (DDPO convention).")
    p.add_argument("--adv_clip_max", type=float, default=5.0)
    p.add_argument("--ppo_style", type=str, default="coco", choices=["coco", "pairs"],
                   help="coco = the exact update structure of train_ppo_coco.py: iterate over "
                        "rollout minibatches, and within each, over timestep POSITIONS with a "
                        "per-sample shuffled order, accumulating gradients across every "
                        "(minibatch, timestep) and taking ONE optimizer step per inner epoch. "
                        "pairs = the earlier variant here, which flattened (sample, timestep) into "
                        "independent minibatches and stepped after each one.")
    p.add_argument("--kl_style", type=str, default="coco_latent", choices=["coco_latent", "per_step_ref"],
                   help="coco_latent = the SD script's regularizer: 0.5*||x_student_final - "
                        "x_teacher||^2, one scalar per rollout batch. per_step_ref = the per-step "
                        "drift penalty against a frozen reference policy used earlier here.")
    p.add_argument("--group_size", type=int, default=1,
                   help="How many rollouts share the same (z_T, class) pair within a batch, differing "
                        "only in the policy's exploration noise. With group_size > 1 the advantage is "
                        "whitened INSIDE each group, so it reflects the policy's own choices instead of "
                        "how intrinsically easy that pair is. Must divide --sample_batch_size.")
    p.add_argument("--stat_key", type=str, default="pair", choices=["pair", "class", "batch"],
                   help="What the reward baseline is computed over. 'pair' is the correct analogue of "
                        "DDPO's per-prompt tracker here: the teacher target is fixed per (z_T, class) "
                        "pair, so pair identity -- not class -- is what determines the achievable reward. "
                        "'class' pools 1000 classes and almost never fills its buffer.")
    p.add_argument("--stat_buffer", type=int, default=16)
    p.add_argument("--stat_min_count", type=int, default=16)
    p.add_argument("--adv_normalize", type=str, default="global_std",
                   choices=["group_std", "global_std", "mean_only"],
                   help="How the group-relative advantage is scaled. group_std divides by each "
                        "group's own std, which blows a group whose rewards are nearly identical up "
                        "to unit magnitude -- amplifying pure noise into a full-strength gradient. "
                        "global_std subtracts the group mean but divides by the epoch-wide std, so "
                        "uninformative groups stay small. mean_only applies no scaling at all.")
    p.add_argument("--kl_coeff", type=float, default=1.0,
                   help="Coefficient of the per-step regularizer towards the frozen reference policy.")
    p.add_argument("--kl_mode", type=str, default="mse", choices=["mse", "analytic"],
                   help="analytic = the true Gaussian KL ||mu-mu_ref||^2 / (2*sigma_up^2). Correct, but "
                        "sigma_up shrinks to ~0.002 on the last step, so that one step's penalty is ~1e6x "
                        "the others and eats the entire gradient-norm budget. mse = the unnormalised "
                        "||mu-mu_ref||^2, which weights every step equally.")

    # --- optimisation -------------------------------------------------------------------------
    p.add_argument("--lr", type=float, default=1e-5,
                   help="NOTE: the SD config uses 3e-4, but that is a LoRA learning rate on a frozen "
                        "backbone. Here every one of the 296M EDM parameters is trainable, so 3e-4 "
                        "destroys the model within a few steps. 1e-5 is the full-finetune equivalent.")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--adam_eps", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--lora_rank", type=int, default=0,
                   help="0 = full fine-tune. >0 attaches rank-r LoRA to the modules matching "
                        "--lora_targets and freezes everything else, mirroring the SD branch. Use "
                        "with a LoRA-scale learning rate (3e-4), not the full-finetune 1e-5.")
    p.add_argument("--lora_alpha", type=float, default=0,
                   help="LoRA scaling; 0 means alpha = rank (scale 1.0).")
    p.add_argument("--lora_targets", type=str, default="qkv|proj|affine",
                   help="Regex over module names. EDM's attention is qkv/proj 1x1 convs; affine is "
                        "the per-block conditioning projection.")
    p.add_argument("--trainable_regex", type=str, default="",
                   help="Only parameters whose name matches this regex are trained; everything else "
                        "is frozen. The REINFORCE gradient's alignment with the true gradient scales "
                        "like sqrt(n_samples / n_params), so shrinking the parameter space is the "
                        "cheapest way to raise the signal-to-noise ratio -- the same reason the SD "
                        "branch uses LoRA. Useful values for the EDM UNet: 'dec\\.' (decoder only), "
                        "'qkv|proj' (attention), 'bias$|norm' (BitFit-style, ~0.1% of the weights).")
    p.add_argument("--step_weighting", type=str, default="none", choices=["none", "sigma"],
                   help="The log-prob gradient at step t scales like 1/sigma_up[t], and sigma_up spans "
                        "~12 down to ~0.0006 over a 16-step Karras schedule. Unweighted, the last two "
                        "steps produce gradients ~10^4x larger than the first ones and, after grad-norm "
                        "clipping, consume the entire update -- the early steps never train. 'sigma' "
                        "multiplies each step's loss by sigma_up[t] / mean(sigma_up), equalising the "
                        "gradient magnitude across the trajectory.")

    # --- reward -------------------------------------------------------------------------------
    p.add_argument("--reward_fn", type=str, default="clip+dino",
                   help="Comma/plus separated terms: clip, dino, perception, lpips, mse, kl, "
                        "text_image, aesthetic. E.g. 'clip+dino', 'dino,lpips,mse'.")
    p.add_argument("--reward_weights", type=str, default=None,
                   help="Optional per-term weights matching --reward_fn, e.g. '1.0,0.5'. The terms "
                        "live on very different scales (cosine ~0.8, -MSE ~-0.01, -LPIPS ~-0.5), so "
                        "an unweighted sum lets one term dominate -- check the per-term logs first.")
    p.add_argument("--perception_model", type=str, default="vit_pe_core_base_patch16_224.fb",
                   help="timm Perception Encoder checkpoint (also: vit_pe_core_large_patch14_336.fb).")
    p.add_argument("--lpips_resolution", type=int, default=64,
                   help="LPIPS input size. Defaults to the native 64px: upsampling first only adds "
                        "interpolation artefacts to a perceptual distance.")
    p.add_argument("--kl_bins", type=int, default=32,
                   help="Histogram bins for the pixel-intensity KL reward.")
    p.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--dino_model", type=str, default="vit_base_patch16_dinov3.lvd1689m",
                   help="timm name (e.g. vit_small|base|large_patch16_dinov3.lvd1689m -- the weight tag "
                        "suffix is mandatory), a HuggingFace id (facebook/dinov2-base), or a torch.hub "
                        "name (dinov2_vitb14). The backend is inferred from the name.")
    p.add_argument("--dino_tokens", type=str, default="patch", choices=["patch", "cls"],
                   help="patch = mean cosine similarity over patch tokens (dense/structural agreement, "
                        "the stricter signal for distillation). cls = single global descriptor.")
    p.add_argument("--reward_resolution", type=int, default=224,
                   help="Encoders expect 224px; ImageNet-64 samples are bicubically upsampled.")

    # --- logging / io -------------------------------------------------------------------------
    p.add_argument("--output_dir", type=str, default="./edm_logs")
    p.add_argument("--run_name", type=str, default="",
                   help="Sub-directory name. Empty => auto-generated from the reward terms, kl, "
                        "batch size and lr (same convention as the SD branch).")
    p.add_argument("--save_every", type=int, default=5, help="Epochs between periodic checkpoints.")
    p.add_argument("--num_checkpoint_limit", type=int, default=5,
                   help="How many periodic checkpoints to keep (best.pt and last.pt are never rotated).")
    p.add_argument("--resume_from", type=str, default="",
                   help="Checkpoint .pt to resume the student (and optimizer) from.")
    p.add_argument("--sample_every", type=int, default=1, help="Epochs between eval-image dumps.")
    p.add_argument("--eval_reward_n", type=int, default=64,
                   help="Pairs used for the deterministic held-out reward logged each epoch as "
                        "eval_reward. Fixed pairs + eta=0 removes almost all the variance that makes "
                        "the training reward_mean unreadable. 0 disables it.")
    p.add_argument("--eval_eta", type=float, default=0.0,
                   help="eta for the eval_reward pass. 0 = deterministic Euler = deployment setting.")
    p.add_argument("--eval_seed", type=int, default=12345,
                   help="Fixed seed for the eval grid, so before/after comparisons are like-for-like.")
    p.add_argument("--allow_tf32", type=int, default=1)
    p.add_argument("--seed", type=int, default=24)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="redif-edm-imagenet")

    return p.parse_args()


# -----------------------------------------------------------------------------------------------
# EDM helpers
# -----------------------------------------------------------------------------------------------


def _resolve_checkpoint(path_or_url, cache_dir):
    """Return a local path. Downloads the pickle once if an http(s) URL is given."""
    if not str(path_or_url).startswith(("http://", "https://")):
        return path_or_url

    import urllib.request
    from urllib.parse import urlparse

    os.makedirs(cache_dir, exist_ok=True)
    local = os.path.join(cache_dir, os.path.basename(urlparse(path_or_url).path))
    if os.path.exists(local) and os.path.getsize(local) > 0:
        return local

    print(f"[init] downloading {path_or_url}\n       -> {local} (~1.2 GB, one time)")

    def _hook(blocks, block_size, total):
        done = blocks * block_size
        pct = 100.0 * done / total if total > 0 else 0.0
        print(f"\r       {done / 1e6:.0f} MB ({pct:.1f}%)", end="", flush=True)

    tmp = local + ".part"
    urllib.request.urlretrieve(path_or_url, tmp, _hook)
    os.replace(tmp, local)
    print()
    return local


def load_edm(pkl_path, device, cache_dir="./checkpoints", edm_repo=None):
    """Load an EDM network from an NVlabs pickle. Returns the EMA precond model.

    The pickle is produced by `torch_utils.persistence`, so the NVlabs/edm repository must be
    importable even for a purely local file -- otherwise `pickle.load` cannot rebuild the object.
    """
    if edm_repo:
        import sys
        edm_repo = os.path.abspath(os.path.expanduser(edm_repo))
        if not os.path.isdir(edm_repo):
            raise RuntimeError(f"--edm_repo does not exist: {edm_repo}\n"
                               "Clone it first: git clone https://github.com/NVlabs/edm")
        if not os.path.isdir(os.path.join(edm_repo, "torch_utils")):
            raise RuntimeError(f"{edm_repo} is not an NVlabs/edm checkout "
                               "(no torch_utils/ inside). Point --edm_repo at the repo root.")
        sys.path.insert(0, edm_repo)
    try:
        import torch_utils  # noqa: F401  (needed by the persistence machinery inside the pickle)
        import dnnlib  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "The NVlabs/edm repository must be importable to unpickle an EDM checkpoint "
            f"(missing: {e.name}).\n"
            "    git clone https://github.com/NVlabs/edm\n"
            "    export PYTHONPATH=$PYTHONPATH:$(pwd)/edm\n"
            "or pass --edm_repo /path/to/edm"
        ) from e

    local_path = _resolve_checkpoint(pkl_path, cache_dir)
    with open(local_path, "rb") as f:
        data = pickle.load(f)
    net = data["ema"] if isinstance(data, dict) and "ema" in data else data
    net = net.to(device)
    # Precision is handled by autocast; disable the network's internal fp16 cast for stable training.
    # Precision is controlled explicitly by the caller via `use_fp16` (EDM's own mixed precision).
    if hasattr(net, "use_fp16"):
        net.use_fp16 = False
    return net


def karras_sigmas(num_steps, sigma_min, sigma_max, rho, device, end_at_sigma_min=True):
    """EDM/Karras time discretisation.

    If `end_at_sigma_min` is True the trajectory ends at sigma_min (0.002) instead of exactly 0.
    This matters for RL: an ancestral step into sigma=0 has zero variance, i.e. a deterministic
    final transition with no log-probability, so the last (and most important) student step would
    receive no policy gradient. Stopping at sigma_min keeps every transition stochastic; the
    residual noise at 0.002 is visually negligible for data in [-1, 1].
    """
    i = torch.arange(num_steps, dtype=torch.float64, device=device)
    t = (sigma_max ** (1 / rho) + i / max(num_steps - 1, 1) *
         (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    if end_at_sigma_min:
        sigmas = t  # length = num_steps, sigmas[-1] = sigma_min
    else:
        sigmas = torch.cat([t, torch.zeros(1, dtype=torch.float64, device=device)])
    return sigmas.to(torch.float32)


def ancestral_coeffs(sigma_from, sigma_to, eta):
    """k-diffusion style ancestral split: sigma_to^2 = sigma_down^2 + sigma_up^2."""
    if sigma_to <= 0:
        return sigma_to, 0.0
    var_up = (sigma_to ** 2) * (sigma_from ** 2 - sigma_to ** 2) / (sigma_from ** 2)
    sigma_up = min(sigma_to, eta * math.sqrt(max(var_up, 0.0)))
    sigma_down = math.sqrt(max(sigma_to ** 2 - sigma_up ** 2, 0.0))
    return sigma_down, sigma_up


def one_hot_labels(class_ids, label_dim, device):
    return F.one_hot(class_ids.to(device), num_classes=label_dim).to(torch.float32)


def _as_batch(v, x):
    """Accept a python float or a per-sample tensor; return a [B] float tensor on x's device."""
    if not torch.is_tensor(v):
        return torch.full((x.shape[0],), float(v), device=x.device, dtype=torch.float32)
    return v.to(device=x.device, dtype=torch.float32).reshape(-1)


def edm_step_mean(net, x, sigma_from, sigma_down, labels):
    """Euler step towards sigma_down. Returns (mean, denoised).

    mean = x + (sigma_down - sigma_from) * (x - D_theta(x, sigma_from)) / sigma_from

    sigma_from/sigma_down may be scalars or per-sample tensors. Per-sample support is what lets the
    COCO-style update train a whole rollout batch at a *different* timestep per sample in one
    forward, exactly as the SD script's `t_perm` shuffling does.
    """
    sf = _as_batch(sigma_from, x)
    sd = _as_batch(sigma_down, x)
    denoised = net(x, sf, labels)
    sf4, sd4 = sf.view(-1, 1, 1, 1), sd.view(-1, 1, 1, 1)
    d = (x - denoised) / sf4
    mean = x + (sd4 - sf4) * d
    return mean, denoised


def gaussian_logprob(x_next, mean, sigma_up):
    """Isotropic Gaussian log-density, averaged over pixel dimensions (DDPO convention)."""
    s = _as_batch(sigma_up, x_next).view(-1, 1, 1, 1)
    lp = -((x_next - mean) ** 2) / (2 * s ** 2) - s.log() - 0.5 * math.log(2 * math.pi)
    return lp.mean(dim=tuple(range(1, lp.ndim)))


# -----------------------------------------------------------------------------------------------
# Samplers
# -----------------------------------------------------------------------------------------------


@torch.no_grad()
def student_rollout(net, ref_net, latents, labels, sigmas, eta, autocast_ctx):
    """Run the N-step stochastic student policy and record the trajectory.

    Returns a dict with:
        latents_cur  [B, N, C, H, W]   state x_i    (input of each transition)
        latents_next [B, N, C, H, W]   state x_{i+1}
        log_probs    [B, N]            log pi_old(x_{i+1} | x_i)
        ref_means    [B, N, C, H, W]   mean of the frozen reference policy at the same x_i
        images       [B, C, H, W]      final sample (at sigma_min)
    """
    x = latents * sigmas[0]
    xs_cur, xs_next, logps, ref_means = [], [], [], []

    for i in range(len(sigmas) - 1):
        s_from = float(sigmas[i])
        s_to = float(sigmas[i + 1])
        s_down, s_up = ancestral_coeffs(s_from, s_to, eta)
        assert s_up > 0, "sigma_up == 0 => deterministic transition, no policy gradient. Increase eta."

        with autocast_ctx():
            mean, _ = edm_step_mean(net, x, s_from, s_down, labels)
            mean = mean.float()
            if ref_net is not None:
                ref_mean, _ = edm_step_mean(ref_net, x, s_from, s_down, labels)
                ref_mean = ref_mean.float()
            else:
                ref_mean = mean

        noise = torch.randn_like(mean)
        x_next = mean + s_up * noise
        logp = gaussian_logprob(x_next, mean, s_up)

        xs_cur.append(x.clone())
        xs_next.append(x_next.clone())
        logps.append(logp)
        ref_means.append(ref_mean.clone())
        x = x_next

    return {
        "latents_cur": torch.stack(xs_cur, dim=1),
        "latents_next": torch.stack(xs_next, dim=1),
        "log_probs": torch.stack(logps, dim=1),
        "ref_means": torch.stack(ref_means, dim=1),
        "images": x,
    }


@torch.no_grad()
def student_sample_det(net, latents, labels, sigmas, eta=0.0):
    """Run the student's schedule with eta (0 = deterministic Euler). Used for evaluation.

    Training rewards are measured under the exploration policy (eta > 0), so they understate what
    the student can actually do: the injected noise costs reward by itself. Deployment uses eta=0,
    and so should the progress metric -- it removes the sampling noise entirely, which is most of
    the epoch-to-epoch variance in `reward_mean`.
    """
    x = latents * sigmas[0]
    for i in range(len(sigmas) - 1):
        s_from, s_to = float(sigmas[i]), float(sigmas[i + 1])
        s_down, s_up = ancestral_coeffs(s_from, s_to, eta)
        mean, _ = edm_step_mean(net, x, s_from, s_down, labels)
        x = mean.float()
        if s_up > 0:
            x = x + s_up * torch.randn_like(x)
    return x


class LoRAWrapper(torch.nn.Module):
    """Low-rank adapter around an EDM 1x1 Conv2d / Linear, the analogue of LoRAAttnProcessor.

    The SD branch attaches LoRA to the UNet's attention processors and trains at 3e-4. EDM's
    DhariwalUNet has no diffusers attention processors, but its attention blocks are ordinary 1x1
    convolutions (`qkv`, `proj`), so wrapping those gives the same thing: the base weight stays
    frozen and only a rank-r correction is learned. B starts at zero, so the student is EXACTLY the
    teacher at init -- and a few optimizer steps at a LoRA-scale learning rate change the output
    far more than the same steps spread over 296M full weights.
    """

    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        w = base.weight
        if w.dim() == 4:
            out_c, in_c = w.shape[0], w.shape[1]
            self.A = torch.nn.Parameter(torch.randn(rank, in_c, 1, 1) / math.sqrt(in_c))
            self.B = torch.nn.Parameter(torch.zeros(out_c, rank, 1, 1))
            self.kind = "conv"
        else:
            out_f, in_f = w.shape
            self.A = torch.nn.Parameter(torch.randn(rank, in_f) / math.sqrt(in_f))
            self.B = torch.nn.Parameter(torch.zeros(out_f, rank))
            self.kind = "linear"
        self.scale = alpha / rank

    def forward(self, x, *a, **kw):
        out = self.base(x, *a, **kw)
        h = x.to(self.A.dtype)
        if self.kind == "conv":
            delta = F.conv2d(F.conv2d(h, self.A), self.B)
        else:
            delta = F.linear(F.linear(h, self.A), self.B)
        return out + self.scale * delta.to(out.dtype)


def apply_lora(net, rank, alpha, target_regex):
    """Wrap every matching 1x1 conv / linear in `net`. Returns the list of wrapped module names."""
    import re
    pat = re.compile(target_regex)
    wrapped = []
    for name, module in list(net.named_modules()):
        if not pat.search(name) or isinstance(module, LoRAWrapper):
            continue
        w = getattr(module, "weight", None)
        if w is None or w.dim() not in (2, 4):
            continue
        if w.dim() == 4 and (w.shape[2] != 1 or w.shape[3] != 1):
            continue  # resampling / k>1 convs change the spatial size; the adapter path would not match
        if getattr(module, "up", False) or getattr(module, "down", False):
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = net.get_submodule(parent_name) if parent_name else net
        setattr(parent, attr, LoRAWrapper(module, rank, alpha))
        wrapped.append(name)
    return wrapped


@torch.no_grad()
def lora_drift(student):
    """Frobenius norm of the learned low-rank correction, relative to the frozen base weights.

    param_drift is useless under LoRA -- the base weights never move, so it stays at exactly 0.
    This measures what actually changed: ||scale * B@A|| / ||W_base||, aggregated.
    """
    num, den = 0.0, 0.0
    for m in student.modules():
        if isinstance(m, LoRAWrapper):
            a = m.A.detach().float().reshape(m.A.shape[0], -1)
            b = m.B.detach().float().reshape(m.B.shape[0], -1)
            num += ((m.scale * (b @ a)) ** 2).sum().item()
            den += (m.base.weight.detach().float() ** 2).sum().item()
    return math.sqrt(num / max(den, 1e-12))


@torch.no_grad()
def param_drift(student, reference):
    """RMS parameter change since init, relative to the init's own RMS.

    The single number that says how far the policy has actually moved. Adam steps are scale-free
    (~lr per parameter per step), so this grows roughly like lr * sqrt(num_optimizer_steps) and is
    far more informative than the loss: a run that degrades the model and a run that does nothing
    look identical in the reward curve but differ by orders of magnitude here.
    """
    ref = dict(reference.named_parameters())
    num, den = 0.0, 0.0
    for name, p_s in student.named_parameters():
        p_r = ref.get(name.replace(".base.", "."))
        if p_r is None:
            continue
        d = p_s.detach().float() - p_r.detach().float().to(p_s.device)
        num += (d ** 2).sum().item()
        den += (p_r.detach().float() ** 2).sum().item()
    return math.sqrt(num / max(den, 1e-12))


@torch.no_grad()
def teacher_sample(net, latents, labels, num_steps, sigma_min, sigma_max, rho, autocast_ctx):
    """Deterministic 2nd-order Heun sampler (EDM Algorithm 1, S_churn = 0).

    Deterministic on purpose: the reward is a *paired* teacher-student comparison, so the target
    must be a single well-defined image per (z_T, class) pair -- which also makes it cacheable.
    """
    sigmas = karras_sigmas(num_steps, sigma_min, sigma_max, rho, latents.device, end_at_sigma_min=False)
    x_next = latents * sigmas[0]
    for i in range(len(sigmas) - 1):
        t_cur, t_next = float(sigmas[i]), float(sigmas[i + 1])
        x_cur = x_next
        with autocast_ctx():
            sb = torch.full((x_cur.shape[0],), t_cur, device=x_cur.device, dtype=torch.float32)
            denoised = net(x_cur, sb, labels).float()
        d_cur = (x_cur - denoised) / t_cur
        x_next = x_cur + (t_next - t_cur) * d_cur
        if t_next > 0:  # 2nd-order correction
            with autocast_ctx():
                sb = torch.full((x_cur.shape[0],), t_next, device=x_cur.device, dtype=torch.float32)
                denoised = net(x_next, sb, labels).float()
            d_prime = (x_next - denoised) / t_next
            x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next


# -----------------------------------------------------------------------------------------------
# (noise, class) pool + teacher cache
# -----------------------------------------------------------------------------------------------


class PairPool:
    """A fixed, data-free pool of (initial noise, class label) pairs -- the ImageNet analogue of
    the fixed prompt set used in the COCO experiment."""

    def __init__(self, num_pairs, num_classes_used, label_dim, img_channels, img_resolution, seed):
        g = torch.Generator().manual_seed(seed)
        self.latents = torch.randn(num_pairs, img_channels, img_resolution, img_resolution, generator=g)
        k = min(num_classes_used, label_dim)
        self.classes = torch.randint(0, k, (num_pairs,), generator=g)
        self.n = num_pairs

    def batch(self, idx):
        return self.latents[idx], self.classes[idx]


def peek_teacher_cache(path):
    """Load an existing teacher cache, or return None. Prints exactly what it found and where.

    The pool the cache belongs to is identified by (pool_seed, num_pairs) and BOTH matter: PairPool
    draws its latents and then its class labels from one generator, so a pool of a different size
    consumes a different number of randoms before the labels are drawn. The latents for the first N
    indices survive, the labels do not -- reusing images across pool sizes would silently pair every
    teacher image with the wrong class.
    """
    abspath = os.path.abspath(path)
    if not os.path.exists(abspath):
        print(f"[cache] no file at {abspath} -> will generate")
        return None
    blob = torch.load(abspath, map_location="cpu")
    print(f"[cache] found {abspath} ({os.path.getsize(abspath) / 1e6:.0f} MB): "
          f"pool_seed={blob.get('pool_seed')}, num_pairs={blob.get('num_pairs')}, "
          f"teacher_steps={blob.get('teacher_steps')}")
    return blob


def build_teacher_cache(args, teacher, pool, device, autocast_ctx, blob=None):
    """Generate (and cache) one teacher image per pool entry. Runs once."""
    if blob is not None:
        if blob.get("num_pairs") == pool.n and blob.get("pool_seed") == args.pool_seed:
            print(f"[cache] reusing {blob['num_pairs']} cached teacher images -- no teacher sampling")
            return blob["images"]
        print(f"[cache] MISMATCH: cache is (seed={blob.get('pool_seed')}, "
              f"pairs={blob.get('num_pairs')}) but this run wants (seed={args.pool_seed}, "
              f"pairs={pool.n}) -> regenerating.")

    print(f"[cache] generating {pool.n} teacher images with {args.teacher_steps}-step Heun "
          f"(NFE = {2 * args.teacher_steps - 1}) ...")
    out = torch.empty(pool.n, teacher.img_channels, teacher.img_resolution, teacher.img_resolution,
                      dtype=torch.uint8)
    t0 = time.time()
    for start in range(0, pool.n, args.precompute_batch):
        idx = torch.arange(start, min(start + args.precompute_batch, pool.n))
        lat, cls = pool.batch(idx)
        lat = lat.to(device)
        labels = one_hot_labels(cls, teacher.label_dim, device)
        img = teacher_sample(teacher, lat, labels, args.teacher_steps,
                             args.sigma_min, args.sigma_max, args.rho, autocast_ctx)
        out[idx] = to_uint8(img).cpu()
        done = idx[-1].item() + 1
        print(f"  {done}/{pool.n}  ({time.time() - t0:.0f}s)", flush=True)

    # Never clobber an existing cache that belongs to a different pool: regenerating one is hours
    # of teacher sampling, and the old file is usually still needed for evaluation. Write the new
    # one under a name that encodes its pool instead, and say so loudly.
    out_path = args.teacher_cache
    if blob is not None and not args.overwrite_cache:
        stem, ext = os.path.splitext(args.teacher_cache)
        out_path = f"{stem}_seed{args.pool_seed}_n{pool.n}{ext}"
        print(f"[cache] {os.path.abspath(args.teacher_cache)} belongs to a different pool and will "
              f"NOT be overwritten (pass --overwrite_cache 1 to force).")

    torch.save({"images": out, "num_pairs": pool.n, "pool_seed": args.pool_seed,
                "teacher_steps": args.teacher_steps}, out_path)
    print(f"[cache] wrote {os.path.abspath(out_path)}")
    return out


def to_uint8(x):
    """EDM images live in [-1, 1]."""
    return (x.clamp(-1, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)


def to_unit(x_uint8):
    return x_uint8.to(torch.float32) / 255.0


# -----------------------------------------------------------------------------------------------
# Reward
# -----------------------------------------------------------------------------------------------


class RewardBank:
    """Terminal reward for the paired teacher/student comparison.

    Mirrors ddpo_pytorch/rewards.py from the SD/COCO branch so both branches of the paper share
    the same reward definitions. Terms are chosen with --reward_fn ("clip+dino", "dino,lpips",
    ...) and summed with optional per-term --reward_weights.

        clip        CLIP image-image cosine similarity
        dino        DINOv3 patch-token (or CLS) cosine similarity
        perception  Meta Perception Encoder cosine similarity (a CLIP-style semantic encoder --
                    NOT a perceptual-distance metric like LPIPS)
        lpips       negative LPIPS perceptual distance
        mse         negative pixel-space MSE
        kl          negative forward KL between per-channel intensity histograms
        text_image  CLIP alignment between the student image and "a photo of a <class name>"
                    (the ImageNet analogue of the COCO caption reward -- teacher-free)
        aesthetic   LAION aesthetic score of the student image only

    Every encoder is frozen and runs under no_grad: in DDPO-style PPO the reward is a black-box
    scalar and the student is updated only through the importance-ratio x advantage term.

    Resolution caveat: ImageNet-64 samples are bicubically upsampled to --reward_resolution before
    encoding, so absolute values are not comparable with the 512px COCO numbers.
    """

    SUPPORTED = ("clip", "dino", "perception", "lpips", "mse", "kl", "text_image", "aesthetic")
    IMNET_MEAN = (0.485, 0.456, 0.406)
    IMNET_STD = (0.229, 0.224, 0.225)
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, args, device):
        self.device = device
        self.res = args.reward_resolution
        self.dino_tokens = args.dino_tokens
        self.lpips_res = args.lpips_resolution
        self.kl_bins = args.kl_bins

        raw = args.reward_fn.replace("+", ",")
        self.terms = [t.strip() for t in raw.split(",") if t.strip()]
        unknown = [t for t in self.terms if t not in self.SUPPORTED]
        if unknown:
            raise ValueError(f"Unknown reward term(s) {unknown}. Choose from {list(self.SUPPORTED)}.")
        if args.reward_weights:
            w = [float(x) for x in args.reward_weights.replace("+", ",").split(",")]
            if len(w) != len(self.terms):
                raise ValueError(f"--reward_weights has {len(w)} entries but --reward_fn has "
                                 f"{len(self.terms)} terms.")
        else:
            w = [1.0] * len(self.terms)
        self.weights = dict(zip(self.terms, w))
        print(f"[reward] terms={self.terms} weights={[self.weights[t] for t in self.terms]}")

        self.clip = self.clip_full = self.dino = self.pe = self.lpips = self.aesthetic = None
        self.text_embeds = None

        if "clip" in self.terms or "text_image" in self.terms:
            self._build_clip(args, need_text="text_image" in self.terms)
        if "dino" in self.terms:
            self._build_dino(args)
        if "perception" in self.terms:
            self._build_perception(args)
        if "lpips" in self.terms:
            import lpips as lpips_lib
            self.lpips = lpips_lib.LPIPS(net="vgg").to(device).eval().requires_grad_(False)
        if "aesthetic" in self.terms:
            from ddpo_pytorch.aesthetic_scorer import AestheticScorer
            self.aesthetic = AestheticScorer(dtype=torch.float32).to(device).eval().requires_grad_(False)

    # -- encoder construction ------------------------------------------------------------------

    def _build_clip(self, args, need_text):
        if need_text:
            from transformers import CLIPModel, CLIPTokenizer
            self.clip_full = CLIPModel.from_pretrained(args.clip_model).to(self.device).eval()
            self.clip_full.requires_grad_(False)
            tok = CLIPTokenizer.from_pretrained(args.clip_model)
            names = self._imagenet_class_names(args)
            prompts = [f"a photo of a {n}" for n in names]
            embs = []
            with torch.no_grad():
                for i in range(0, len(prompts), 256):
                    batch = tok(prompts[i:i + 256], return_tensors="pt", padding=True,
                                truncation=True).to(self.device)
                    embs.append(F.normalize(self.clip_full.get_text_features(**batch).float(), dim=-1))
            self.text_embeds = torch.cat(embs)  # [num_classes, D], precomputed once
        else:
            from transformers import CLIPVisionModelWithProjection
            self.clip = CLIPVisionModelWithProjection.from_pretrained(
                args.clip_model).to(self.device).eval()
            self.clip.requires_grad_(False)

    @staticmethod
    def _imagenet_class_names(args):
        """Class-index -> human readable name, the ImageNet stand-in for COCO captions."""
        try:
            from timm.data import ImageNetInfo
            info = ImageNetInfo()
            return [info.index_to_description(i) for i in range(1000)]
        except Exception as e:
            raise RuntimeError(
                "The text_image reward needs ImageNet class names; timm.data.ImageNetInfo failed "
                f"({e}). Upgrade timm or drop text_image from --reward_fn.") from e

    def _build_dino(self, args):
        name = args.dino_model
        if name.startswith("vit_") or "dinov3" in name:
            import timm
            try:
                # dynamic_img_size interpolates the position embeddings, so the 64px samples can be
                # encoded at whatever --reward_resolution we pick instead of the pretraining size.
                self.dino = timm.create_model(name, pretrained=True, num_classes=0,
                                              dynamic_img_size=True).to(self.device).eval()
            except RuntimeError as e:
                raise RuntimeError(
                    f"Could not load '{name}' ({e}). The weight tag suffix is mandatory "
                    "(e.g. vit_base_patch16_dinov3.lvd1689m). Try `pip install -U timm`.") from e
            self._dino_prefix = getattr(self.dino, "num_prefix_tokens", 1)
            self._dino_backend = "timm"
        elif "/" in name:
            from transformers import AutoModel
            self.dino = AutoModel.from_pretrained(name).to(self.device).eval()
            self._dino_backend = "hf"
        else:
            self.dino = torch.hub.load("facebookresearch/dinov2", name).to(self.device).eval()
            self._dino_backend = "hub"
        self.dino.requires_grad_(False)

    def _build_perception(self, args):
        import timm
        try:
            self.pe = timm.create_model(args.perception_model, pretrained=True,
                                        num_classes=0).to(self.device).eval()
        except RuntimeError as e:
            raise RuntimeError(
                f"Could not load '{args.perception_model}' ({e}). Perception Encoder entrypoints "
                "are recent -- try `pip install -U timm`.") from e
        self.pe.requires_grad_(False)
        try:
            cfg = timm.data.resolve_data_config({}, model=self.pe)
            self.pe_mean = list(cfg.get("mean", (0.5, 0.5, 0.5)))
            self.pe_std = list(cfg.get("std", (0.5, 0.5, 0.5)))
            self.pe_size = cfg.get("input_size", (3, 224, 224))[-1]
        except Exception:  # PE-Core's documented preprocessing
            self.pe_mean, self.pe_std, self.pe_size = [0.5] * 3, [0.5] * 3, 224

    # -- preprocessing -------------------------------------------------------------------------

    def _prep(self, x01, mean, std, size=None):
        size = size or self.res
        x = F.interpolate(x01, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)
        m = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
        s = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
        return (x - m) / s

    # -- individual terms (each returns a [B] tensor, higher = better) --------------------------

    @staticmethod
    def _cosine(a, b):
        """a, b are L2-normalised along the last axis; token-level inputs are averaged to a scalar."""
        sim = (a * b).sum(dim=-1)
        if sim.dim() > 1:
            sim = sim.mean(dim=tuple(range(1, sim.dim())))
        return sim

    def _embed_clip(self, x01):
        px = self._prep(x01, self.CLIP_MEAN, self.CLIP_STD, 224)
        if self.clip_full is not None:
            e = self.clip_full.get_image_features(pixel_values=px)
        else:
            e = self.clip(pixel_values=px).image_embeds
        return F.normalize(e.float(), dim=-1)

    def _embed_dino(self, x01):
        px = self._prep(x01, self.IMNET_MEAN, self.IMNET_STD)
        if self._dino_backend == "timm":
            f = self.dino.forward_features(px)                  # [B, prefix + patches, D]
            e = f[:, self._dino_prefix:, :] if self.dino_tokens == "patch" else f[:, 0]
        elif self._dino_backend == "hf":
            h = self.dino(pixel_values=px).last_hidden_state
            e = h[:, 1:, :] if self.dino_tokens == "patch" else h[:, 0]
        else:
            e = self.dino(px)
        return F.normalize(e.float(), dim=-1)

    def _embed_pe(self, x01):
        px = self._prep(x01, self.pe_mean, self.pe_std, self.pe_size)
        return F.normalize(self.pe(px).float(), dim=-1)         # already globally pooled

    def _term_lpips(self, s01, t01):
        s = self._prep(s01, (0.5,) * 3, (0.5,) * 3, self.lpips_res)  # -> [-1, 1]
        t = self._prep(t01, (0.5,) * 3, (0.5,) * 3, self.lpips_res)
        return -self.lpips(s, t).view(-1)

    @staticmethod
    def _term_mse(s01, t01):
        return -((s01 - t01) ** 2).mean(dim=(1, 2, 3))

    def _term_kl(self, s01, t01, eps=1e-6):
        """Negative forward KL(teacher || student) over per-channel intensity histograms.

        Forward KL keeps the teacher as the reference distribution (mode-covering): the student is
        punished hardest for putting no mass in an intensity bin the teacher actually uses. Blind
        to spatial structure by construction -- pair it with clip/dino/perception, never alone.
        """
        def hist(x):
            B, C, H, W = x.shape
            idx = ((x.clamp(0, 1 - 1e-6)) * self.kl_bins).long().clamp(0, self.kl_bins - 1)
            idx = idx.reshape(B, C, H * W)
            counts = torch.zeros(B, C, self.kl_bins, device=x.device, dtype=x.dtype)
            counts.scatter_add_(2, idx, torch.ones_like(idx, dtype=x.dtype))
            return (counts + eps) / (counts.sum(-1, keepdim=True) + eps * self.kl_bins)

        s_p, t_p = hist(s01), hist(t01)
        return -(t_p * (t_p.log() - s_p.log())).sum(-1).sum(-1)

    # -- public API ----------------------------------------------------------------------------

    @torch.no_grad()
    def __call__(self, student_imgs_pm1, teacher_imgs_uint8, classes=None):
        """student_imgs_pm1: [B,3,H,W] in [-1,1]; teacher_imgs_uint8: [B,3,H,W] uint8.

        Returns (weighted total [B] on cpu, dict of raw per-term scores).
        """
        s01 = ((student_imgs_pm1.clamp(-1, 1) + 1) / 2).to(self.device)
        t01 = to_unit(teacher_imgs_uint8).to(self.device)

        parts = {}
        for term in self.terms:
            if term == "clip":
                parts[term] = self._cosine(self._embed_clip(s01), self._embed_clip(t01))
            elif term == "dino":
                parts[term] = self._cosine(self._embed_dino(s01), self._embed_dino(t01))
            elif term == "perception":
                parts[term] = self._cosine(self._embed_pe(s01), self._embed_pe(t01))
            elif term == "lpips":
                parts[term] = self._term_lpips(s01, t01)
            elif term == "mse":
                parts[term] = self._term_mse(s01, t01)
            elif term == "kl":
                parts[term] = self._term_kl(s01, t01)
            elif term == "text_image":
                if classes is None:
                    raise ValueError("text_image reward needs the class ids of the batch.")
                img_e = self._embed_clip(s01)
                parts[term] = (img_e * self.text_embeds[classes.to(self.device)]).sum(-1)
            elif term == "aesthetic":
                u8 = (s01 * 255).round().clamp(0, 255).to(torch.uint8)
                sc = self.aesthetic(F.interpolate(u8.float() / 255, (224, 224), mode="bicubic",
                                                  align_corners=False).mul(255).round()
                                    .clamp(0, 255).to(torch.uint8))
                parts[term] = sc if isinstance(sc, torch.Tensor) else torch.tensor(sc, device=self.device)
            parts[term] = parts[term].float().cpu()

        total = sum(self.weights[t] * parts[t] for t in self.terms)
        return total, parts


# Backwards-compatible alias for the earlier single-purpose class name.
SemanticReward = RewardBank


class PerKeyStatTracker:
    """Reward whitening per conditioning key (pair id here, prompt in the COCO version).

    Why the key matters more than it does in the COCO setting: there, the same prompt is re-rolled
    with fresh latents, so per-prompt statistics capture prompt difficulty. Here the teacher target
    is fixed per (z_T, class) pair, so the achievable reward varies enormously ACROSS pairs and only
    slightly WITHIN a pair. Whitening against a mixed-pair baseline turns most of the advantage into
    "this pair happened to be easy", which is uncorrelated with anything the policy did -- pure
    gradient noise with a bias attached.
    """

    def __init__(self, buffer_size, min_count):
        self.buf = defaultdict(lambda: deque(maxlen=buffer_size))
        self.min_count = min_count

    def update(self, keys, rewards):
        keys = np.asarray(keys)
        rewards = np.asarray(rewards)
        adv = np.empty_like(rewards)
        for k in np.unique(keys):
            m = keys == k
            self.buf[int(k)].extend(rewards[m].tolist())
            if len(self.buf[int(k)]) < self.min_count:
                mu, sd = rewards.mean(), rewards.std()
            else:
                arr = np.asarray(self.buf[int(k)])
                mu, sd = arr.mean(), arr.std()
            adv[m] = (rewards[m] - mu) / (sd + 1e-6)
        return adv


def group_advantages(keys, rewards, mode="global_std"):
    """Centre rewards within each group of identical (z_T, class) rollouts (GRPO-style).

    Every sample in a group faces the exact same target and the same starting noise, so the spread
    inside the group is caused only by the policy's exploration. That makes the group mean the
    lowest-variance unbiased baseline available, with no warm-up and no cross-pair bias.

    The scaling is the part worth being careful about. Dividing by the group's own std (the plain
    GRPO recipe) rescales every group to unit variance, including groups where all four rollouts
    scored within 1e-4 of each other -- i.e. where the ranking is meaningless. Those degenerate
    groups then contribute gradients just as strong as informative ones. Scaling by the epoch-wide
    std instead keeps informative groups large and uninformative ones small.

    Returns (advantages, mean within-group std) -- the second value is the diagnostic that tells you
    whether the exploration noise produces any reward signal at all.
    """
    keys = np.asarray(keys)
    rewards = np.asarray(rewards)
    adv = np.empty_like(rewards)
    within = []
    for k in np.unique(keys):
        m = keys == k
        r = rewards[m]
        if m.sum() > 1:
            within.append(r.std())
            adv[m] = r - r.mean()
            if mode == "group_std":
                adv[m] = adv[m] / (r.std() + 1e-6)
        else:
            adv[m] = 0.0
    if mode == "global_std":
        adv = adv / (rewards.std() + 1e-6)
    return adv, float(np.mean(within)) if within else 0.0


# Backwards-compatible alias.
PerClassStatTracker = PerKeyStatTracker


# -----------------------------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------------------------


def make_autocast(device, dtype):
    """EDM is NOT compatible with torch.autocast.

    `EDMPrecond.forward` ends with `assert F_x.dtype == dtype`, where dtype is decided by the
    network's own `use_fp16` flag. Under autocast the inner model returns bf16/fp16 while the
    precond expects fp32, and the assertion fires. EDM ships its own mixed precision instead:
    every layer does `weight.to(x.dtype)`, so setting `net.use_fp16 = True` runs the trunk in fp16
    while parameters stay fp32. We therefore always use a null context and control precision
    through `use_fp16` (see --teacher_fp16 / --student_fp16).
    """
    return contextlib.nullcontext


PLOT_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.6,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,   # editable text in Illustrator, required by many venues
    "ps.fonttype": 42,
}


def save_training_curves(run_dir, hist, reward_label, best_epoch):
    """Publication-style loss / reward curves, written next to the checkpoints.

    Same styling as the SD branch so both sets of figures can go in one paper. Rewritten every
    epoch (cheap) rather than only at the end, so a run that is killed early still leaves usable
    figures behind. Uses the Agg backend explicitly: this script normally runs headless over SSH.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] plotting unavailable: {e}")
        return

    epochs = hist["epoch"]
    if len(epochs) < 2:
        return

    with plt.rc_context(PLOT_RC):
        # -- Loss curve ------------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(4.0, 3.0))
        ax.plot(epochs, hist["pg_loss"], color="#1f77b4", label="Policy loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, linewidth=0.4, alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(run_dir, f"loss_curve.{ext}"))
        plt.close(fig)

        # -- Reward curve ----------------------------------------------------------------------
        r = np.asarray(hist["reward_mean"], dtype=np.float64)
        sd = np.asarray(hist["reward_std"], dtype=np.float64)

        fig, ax = plt.subplots(figsize=(4.0, 3.0))
        ax.plot(epochs, r, color="#1f3fbf", label=f"Train reward ({reward_label})")
        ax.fill_between(epochs, r - sd, r + sd, color="#a9bdf2", alpha=0.4, linewidth=0,
                        label=r"$\pm 1$ std")
        # The deterministic held-out curve belongs on the same axes: identical reward terms, so the
        # same units. The offset between the two lines is purely the cost of the exploration noise.
        if any(v is not None for v in hist["eval_reward"]):
            ev = np.asarray([np.nan if v is None else v for v in hist["eval_reward"]], dtype=np.float64)
            ax.plot(epochs, ev, color="#2e8b57", label="Held-out reward (eta=0)")
        if best_epoch is not None:
            ax.axvline(best_epoch, color="#B94F4F", linestyle="--", linewidth=1.0,
                       label=f"Best epoch ({best_epoch})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Reward")
        ax.grid(True, linewidth=0.4, alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(run_dir, f"reward_curve.{ext}"))
        plt.close(fig)


def save_grid(images_pm1, path, nrow=8):
    try:
        import torchvision
        torchvision.utils.save_image((images_pm1.clamp(-1, 1) + 1) / 2, path, nrow=nrow)
    except Exception as e:
        print(f"[warn] could not save grid: {e}")


def train_student(args, num_steps, base_net, pool, teacher_images, reward_fn, device, autocast_ctx,
                  teacher_net=None):
    """One complete PPO distillation run for a given student step count.

    Called once per entry of --student_steps. Every run starts from a fresh behaviour-cloning init
    but reuses the same pool and the same cached teacher targets, so the 4-step vs 8-step comparison
    differs only in the discretisation.
    """
    run_dir = os.path.join(args.output_dir, args.run_name, f"steps{num_steps}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "num_steps": num_steps}, f, indent=2)

    # Student = behaviour-cloning init from the teacher weights; reference = frozen copy for the KL
    # term. eval() from the start: see the note in the PPO block about EDM's dropout.
    student = copy.deepcopy(base_net).to(device).eval().requires_grad_(True)
    ref_net = (copy.deepcopy(base_net).to(device).eval().requires_grad_(False)
               if args.kl_coeff > 0 else None)
    # Held-out slice for the deterministic progress metric: the LAST pairs of the pool, so it never
    # overlaps the first indices used for the sample grid.
    n_ev = min(args.eval_reward_n, pool.n)
    eval_idx = torch.arange(pool.n - n_ev, pool.n) if n_ev > 0 else None

    stat_tracker = PerKeyStatTracker(args.stat_buffer, args.stat_min_count)

    if args.lora_rank > 0:
        alpha = args.lora_alpha or args.lora_rank
        names = apply_lora(student, args.lora_rank, alpha, args.lora_targets)
        student.to(device)
        student.requires_grad_(False)
        trainable = []
        for n, prm in student.named_parameters():
            keep = n.endswith(".A") or n.endswith(".B")
            prm.requires_grad_(keep)
            if keep:
                trainable.append(prm)
        if not trainable:
            raise ValueError(f"--lora_targets '{args.lora_targets}' matched no wrappable modules.")
        print(f"[run steps={num_steps}] LoRA rank {args.lora_rank} on {len(names)} modules "
              f"(e.g. {names[:3]})")
    else:
        trainable = list(student.parameters())
    if args.lora_rank == 0 and args.trainable_regex:
        import re
        pat = re.compile(args.trainable_regex)
        trainable = []
        for name, prm in student.named_parameters():
            keep = bool(pat.search(name))
            prm.requires_grad_(keep)
            if keep:
                trainable.append(prm)
        if not trainable:
            raise ValueError(f"--trainable_regex '{args.trainable_regex}' matched no parameters.")
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in student.parameters())
    print(f"[run steps={num_steps}] trainable {n_train / 1e6:.1f}M / {n_total / 1e6:.1f}M params "
          f"({100 * n_train / n_total:.1f}%)")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr,
                                  betas=(args.adam_beta1, args.adam_beta2),
                                  weight_decay=args.weight_decay, eps=args.adam_eps)

    start_epoch = 0
    if args.resume_from:
        ck = torch.load(args.resume_from, map_location="cpu")
        student.load_state_dict(ck["model"])
        if "optimizer" in ck:
            optimizer.load_state_dict(ck["optimizer"])
        start_epoch = int(ck.get("epoch", -1)) + 1
        print(f"[resume] {args.resume_from} -> starting at epoch {start_epoch}")

    best_reward = -float("inf")
    best_epoch = None
    history = defaultdict(list)
    periodic_ckpts = []

    sigmas = karras_sigmas(num_steps + 1, args.sigma_min, args.sigma_max, args.rho,
                           device, end_at_sigma_min=True)
    # sigmas has num_steps+1 entries -> num_steps stochastic transitions.
    step_coeffs = [ancestral_coeffs(float(sigmas[i]), float(sigmas[i + 1]), args.eta)
                   for i in range(num_steps)]
    print(f"[run steps={num_steps}] sigmas:", [round(float(s), 4) for s in sigmas])
    print(f"[run steps={num_steps}] sigma_up per step:", [round(c[1], 4) for c in step_coeffs])

    ups = np.array([c[1] for c in step_coeffs], dtype=np.float64)
    if args.step_weighting == "sigma":
        step_w = ups / ups.mean()
        print(f"[run steps={num_steps}] loss weight per step:", [round(float(w), 4) for w in step_w])
        print(f"[run steps={num_steps}] without this weighting the last step's gradient would be "
              f"~{ups[0] / ups[-1]:.0f}x the first step's")
    else:
        step_w = np.ones_like(ups)

    wandb_run = None
    if args.use_wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=f"steps{num_steps}",
                               config={**vars(args), "num_steps": num_steps}, reinit=True)

    global_step = 0
    for epoch in range(start_epoch, args.num_epochs):
        # =========================================================================== ROLLOUT ===
        student.eval()
        buffers, epoch_rewards, epoch_parts = [], [], defaultdict(list)
        n_batches = args.num_batches_per_epoch

        for _ in range(n_batches):
            if args.online_teacher:
                # Fresh (z_T, class) every batch; the teacher target is generated on the spot.
                B = args.sample_batch_size
                cls = torch.randint(0, min(args.num_classes_used, student.label_dim), (B,))
                lat = torch.randn(B, student.img_channels, student.img_resolution,
                                  student.img_resolution, device=device)
                labels = one_hot_labels(cls, student.label_dim, device)
                target_u8 = to_uint8(teacher_sample(teacher_net, lat, labels, args.teacher_steps,
                                                    args.sigma_min, args.sigma_max, args.rho,
                                                    autocast_ctx))
            else:
                # Group sampling: draw n_groups distinct pairs and roll each out group_size times.
                # Same z_T, same class, different exploration noise -> a within-group baseline.
                n_groups = max(1, args.sample_batch_size // max(1, args.group_size))
                n_active = min(args.active_pairs, pool.n) if args.active_pairs > 0 else pool.n
                idx = torch.randint(0, n_active, (n_groups,)).repeat_interleave(
                    max(1, args.group_size))[:args.sample_batch_size]
                lat, cls = pool.batch(idx)
                lat = lat.to(device)
                labels = one_hot_labels(cls, student.label_dim, device)
                target_u8 = teacher_images[idx]

            traj = student_rollout(student, ref_net, lat, labels, sigmas, args.eta, autocast_ctx)
            rewards, parts = reward_fn(traj["images"], target_u8, classes=cls)

            # COCO-style regularizer: 0.5 * ||x_student_final - x_teacher||^2, one scalar per batch.
            # Computed here, in the no-grad rollout, exactly as latent_kl() is in the SD script.
            tgt = (to_unit(target_u8).to(device) * 2 - 1)
            latent_kl_batch = 0.5 * ((traj["images"].float() - tgt).flatten(1) ** 2).sum(-1).mean()

            buffers.append({
                "latents_cur": traj["latents_cur"].cpu(),
                "latents_next": traj["latents_next"].cpu(),
                "log_probs": traj["log_probs"].cpu(),
                "ref_means": traj["ref_means"].to(torch.float16).cpu(),
                "labels": labels.cpu(),
                "classes": cls.clone(),
                "keys": (idx.clone() if (not args.online_teacher and args.stat_key == "pair")
                         else cls.clone()),
                "rewards": rewards.cpu(),
                "latent_kl": float(latent_kl_batch),
            })
            epoch_rewards.append(rewards.cpu())
            for k, v in parts.items():
                epoch_parts[k].append(v.cpu())

        rewards_all = torch.cat(epoch_rewards).numpy()
        keys_all = torch.cat([b["keys"] for b in buffers]).numpy()
        within_std = 0.0
        if args.group_size > 1 and not args.online_teacher:
            advantages, within_std = group_advantages(keys_all, rewards_all, args.adv_normalize)
        elif stat_tracker is not None and args.stat_key != "batch":
            advantages = stat_tracker.update(keys_all, rewards_all)
        else:
            advantages = (rewards_all - rewards_all.mean()) / (rewards_all.std() + 1e-6)
        advantages = np.clip(advantages, -args.adv_clip_max, args.adv_clip_max)

        # Flatten the buffer into (sample, timestep) transitions.
        flat = {k: torch.cat([b[k] for b in buffers], dim=0)
                for k in ["latents_cur", "latents_next", "log_probs", "ref_means", "labels"]}
        flat["advantages"] = torch.from_numpy(advantages).float()
        n_samples = flat["latents_cur"].shape[0]

        # Slice the whitened advantages back onto the minibatch they came from, the way the SD
        # script slices `advantages_all` per rollout batch.
        cursor = 0
        for b in buffers:
            bs = b["rewards"].shape[0]
            b["advantages"] = flat["advantages"][cursor:cursor + bs]
            cursor += bs
        kl_reg = float(np.mean([b["latent_kl"] for b in buffers]))

        # =========================================================================== PPO =======
        # NOTE: the student stays in eval() mode on purpose. The EDM ImageNet-64 checkpoint was
        # trained with dropout=0.10, and its UNet blocks apply dropout whenever module.training is
        # True. Calling student.train() here would recompute the log-probs through a DIFFERENT
        # (dropout-perturbed) network than the one that actually produced the actions, so the
        # importance ratio would be biased rather than 1 on the first pass, and the update would
        # optimise the dropout-on network while every rollout and every evaluation runs dropout-off.
        # That mismatch degrades the policy monotonically. Gradients flow fine in eval mode --
        # requires_grad is what matters, not the training flag.
        # SD's UNet has no dropout, which is why the COCO branch never hit this.
        student.eval()
        stats = defaultdict(list)
        n_t_cfg = max(1, int(round(num_steps * args.timestep_fraction)))

        if args.ppo_style == "coco":
          # ---- exact port of the SD/COCO update -------------------------------------------
          # For each inner epoch: walk the rollout minibatches in a shuffled order; inside each,
          # walk timestep POSITIONS with an independently shuffled timestep order per sample;
          # accumulate the gradient over every (minibatch, timestep) and take ONE optimizer step.
          # That is the structure Accelerate produced there with
          # gradient_accumulation_steps = num_minibatches * num_train_timesteps.
          sig_cpu = sigmas.detach().cpu()
          down_cpu = torch.tensor([c[0] for c in step_coeffs])
          up_cpu = torch.tensor([c[1] for c in step_coeffs])

          for inner in range(args.inner_epochs):
                optimizer.zero_grad(set_to_none=True)
                n_accum = max(1, len(buffers) * n_t_cfg)
                order = torch.randperm(len(buffers)).tolist()
                for bi in order:
                    b = buffers[bi]
                    B = b["log_probs"].shape[0]
                    rows = torch.arange(B)
                    # per-sample shuffled timestep order, as in the SD script's t_perm
                    t_perm = torch.stack([torch.randperm(num_steps)[:n_t_cfg] for _ in range(B)])
                    labels_b = b["labels"].to(device)
                    adv_b = b["advantages"].to(device)

                    for j_idx in range(n_t_cfg):
                        j = t_perm[:, j_idx]
                        x_cur = b["latents_cur"][rows, j].to(device)
                        x_next = b["latents_next"][rows, j].to(device)
                        old_lp = b["log_probs"][rows, j].to(device)

                        with autocast_ctx():
                            mean, _ = edm_step_mean(student, x_cur, sig_cpu[j], down_cpu[j], labels_b)
                        mean = mean.float()
                        lp = gaussian_logprob(x_next, mean, up_cpu[j])
                        ratio = torch.exp(lp - old_lp)
                        surr1 = -adv_b * ratio
                        surr2 = -adv_b * torch.clamp(ratio, 1.0 - args.clip_range,
                                                     1.0 + args.clip_range)
                        policy_loss = torch.max(surr1, surr2).mean()

                        # kl_reg is a constant here, exactly as in the SD script where latent_kl is
                        # computed inside the no-grad rollout: it shifts the reported loss but
                        # contributes no gradient.
                        step_loss = (policy_loss + args.kl_coeff * kl_reg) / n_accum
                        step_loss.backward()

                        stats["pg_loss"].append(policy_loss.item())
                        stats["ratio"].append(ratio.mean().item())
                        stats["clipfrac"].append(
                            (torch.abs(ratio - 1.0) > args.clip_range).float().mean().item())
                        if inner == 0 and bi == order[0] and j_idx == 0:
                            stats["ratio_t0"] = [ratio.mean().item()]

                gn = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                stats["grad_norm"].append(float(gn))
                stats["opt_steps"] = [float(args.inner_epochs)]
                stats["kl"] = [kl_reg]
                global_step += 1

        else:
            for inner in range(args.inner_epochs):
                # timestep_fraction < 1 trains on a random subset of the trajectory each inner epoch.
                n_t = max(1, int(round(num_steps * args.timestep_fraction)))
                pairs = [(s, t) for s in range(n_samples)
                         for t in (range(num_steps) if n_t == num_steps
                                   else random.sample(range(num_steps), n_t))]
                random.shuffle(pairs)

                optimizer.zero_grad(set_to_none=True)
                n_minibatches = len(range(0, len(pairs) - args.train_batch_size + 1, args.train_batch_size))
                stats["opt_steps"] = [n_minibatches / max(1, args.grad_accum)]
                for mb_i in range(0, len(pairs) - args.train_batch_size + 1, args.train_batch_size):
                    mb = pairs[mb_i:mb_i + args.train_batch_size]
                    s_idx = torch.tensor([p[0] for p in mb])
                    t_idx = torch.tensor([p[1] for p in mb])

                    x_cur = flat["latents_cur"][s_idx, t_idx].to(device)
                    x_next = flat["latents_next"][s_idx, t_idx].to(device)
                    old_lp = flat["log_probs"][s_idx, t_idx].to(device)
                    ref_mean = flat["ref_means"][s_idx, t_idx].to(device).float()
                    labels = flat["labels"][s_idx].to(device)
                    adv = flat["advantages"][s_idx].to(device)

                    # All transitions in a minibatch may come from different timesteps; group them.
                    loss_terms, kl_terms, ratios = [], [], []
                    for t in t_idx.unique():
                        m = (t_idx == t).to(device)
                        s_from = float(sigmas[int(t)])
                        s_down, s_up = step_coeffs[int(t)]

                        with autocast_ctx():
                            mean, _ = edm_step_mean(student, x_cur[m], s_from, s_down, labels[m])
                        mean = mean.float()

                        lp = gaussian_logprob(x_next[m], mean, s_up)
                        ratio = torch.exp(lp - old_lp[m])
                        a = adv[m]
                        unclipped = -a * ratio
                        clipped = -a * torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range)
                        pg = torch.max(unclipped, clipped) * float(step_w[int(t)])

                        # Both forms measure drift from the reference policy; they differ only in how
                        # the per-step scale is handled (see --kl_mode).
                        kl = ((mean - ref_mean[m]) ** 2).mean(dim=(1, 2, 3))
                        if args.kl_mode == "analytic":
                            kl = kl / (2 * s_up ** 2)

                        loss_terms.append(pg)
                        kl_terms.append(kl)
                        ratios.append(ratio.detach())

                    pg_loss = torch.cat(loss_terms).mean()
                    kl_loss = torch.cat(kl_terms).mean()
                    loss = (pg_loss + args.kl_coeff * kl_loss) / args.grad_accum
                    loss.backward()

                    stats["pg_loss"].append(pg_loss.item())
                    stats["kl"].append(kl_loss.item())
                    r_mean = torch.cat(ratios).mean().item()
                    stats["ratio"].append(r_mean)
                    if mb_i == 0 and inner == 0:
                        # Before any optimizer step of this epoch the policy is still exactly the one
                        # that generated the actions, so this MUST be 1.0 to ~1e-6. Anything else means
                        # the recomputed log-prob comes from a different network than the rollout used
                        # (dropout left on, precision mismatch, a wrong sigma index...). The epoch-mean
                        # `ratio` cannot show this, because it mixes in genuine post-update drift.
                        stats["ratio_t0"] = [r_mean]

                    if ((mb_i // args.train_batch_size) + 1) % args.grad_accum == 0:
                        gn = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        stats["grad_norm"].append(float(gn))
                        global_step += 1

        # ================================================================ DETERMINISTIC EVAL ===
        eval_metrics = {}
        if eval_idx is not None:
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            torch.manual_seed(args.eval_seed)
            ev_rewards, ev_parts = [], defaultdict(list)
            for st in range(0, len(eval_idx), args.sample_batch_size):
                sub_idx = eval_idx[st:st + args.sample_batch_size]
                lat_e, cls_e = pool.batch(sub_idx)
                lab_e = one_hot_labels(cls_e, student.label_dim, device)
                with torch.no_grad():
                    img_e = student_sample_det(student, lat_e.to(device), lab_e, sigmas, args.eval_eta)
                r_e, p_e = reward_fn(img_e, teacher_images[sub_idx], classes=cls_e)
                ev_rewards.append(r_e)
                for k, v in p_e.items():
                    ev_parts[k].append(v)
            eval_metrics["eval_reward"] = float(torch.cat(ev_rewards).mean())
            for k, v in ev_parts.items():
                eval_metrics[f"eval_{k}"] = float(torch.cat(v).mean())
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

        if args.lora_rank > 0:
            eval_metrics["lora_drift"] = lora_drift(student)
        else:
            drift_ref = ref_net if ref_net is not None else base_net
            eval_metrics["param_drift"] = param_drift(student, drift_ref)

        # =========================================================================== LOG =======
        log = {
            "steps": num_steps,
            "epoch": epoch,
            "reward_mean": float(rewards_all.mean()),
            "reward_std": float(rewards_all.std()),
            # If adv_std collapses towards 0, the groups agree and there is nothing to learn from;
            # if reward_std >> the within-group spread, the baseline is doing the wrong job.
            "adv_std": float(advantages.std()),
            # within_group_std / reward_std is the signal-to-spread ratio. If it is ~0 the policy's
            # exploration produces no measurable reward difference and there is nothing to learn:
            # raise --eta (or drop to a reward that actually reacts to small perturbations).
            "within_group_std": within_std,
            "unique_pairs": int(len(np.unique(keys_all))),
            **{f"reward_{k}": float(torch.cat(v).mean()) for k, v in epoch_parts.items()},
            **{k: float(np.mean(v)) for k, v in stats.items() if len(v)},
            **eval_metrics,
        }
        print(json.dumps(log), flush=True)
        with open(os.path.join(run_dir, "log.jsonl"), "a") as f:
            f.write(json.dumps(log) + "\n")

        for key in ("epoch", "reward_mean", "reward_std", "pg_loss", "eval_reward"):
            history[key].append(log.get(key))
        if wandb_run is not None:
            wandb_run.log(log, step=global_step)

        if args.sample_every and (epoch % args.sample_every == 0):
            # Fixed seed + fixed pool slice so the only thing that changes between grids is the
            # student. CRITICAL: the RNG state must be saved and restored around this. Reseeding
            # the global RNG here would make the next epoch's rollout replay the exact same pool
            # indices and the exact same exploration noise as this one -- training would silently
            # collapse onto a handful of fixed samples.
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            torch.manual_seed(args.eval_seed)
            with torch.no_grad():
                idx = torch.arange(0, min(16, pool.n))
                lat, cls = pool.batch(idx)
                labels = one_hot_labels(cls, student.label_dim, device)
                traj = student_rollout(student, None, lat.to(device), labels, sigmas, args.eta, autocast_ctx)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            save_grid(traj["images"], os.path.join(run_dir, f"student_ep{epoch:04d}.png"))
            if teacher_images is not None:
                save_grid(to_unit(teacher_images[idx]).to(device) * 2 - 1,
                          os.path.join(run_dir, "teacher_reference.png"))

        # ---- checkpointing ---------------------------------------------------------------
        meta = {"epoch": epoch, "num_steps": num_steps, "sigmas": sigmas.cpu(),
                "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
                "lora_targets": args.lora_targets,
                "reward_mean": log["reward_mean"], "eval_reward": log.get("eval_reward"),
                "param_drift": log.get("param_drift"), "args": vars(args)}
        # best.pt is for evaluation, so it carries weights only: the AdamW state is 2x the model
        # size (~2.4 GB here) and is useless without resuming the exact same run. last.pt keeps it
        # so --resume_from still works.
        best_payload = {"model": student.state_dict(), **meta}
        payload = {**best_payload, "optimizer": optimizer.state_dict()}

        # "last" is always the newest epoch; "best" is the highest score SO FAR, written the moment
        # it is reached -- independent of --save_every -- so a run that later collapses (reward
        # hacking, drift) still leaves its best model behind.
        torch.save(payload, os.path.join(run_dir, "last.pt"))
        # Select on the deterministic held-out reward when we have it: picking "best" off the noisy
        # stochastic training reward mostly selects for a lucky epoch.
        select_on = log.get("eval_reward", log["reward_mean"])
        if select_on > best_reward:
            best_reward = select_on
            best_epoch = epoch
            best_path = os.path.join(run_dir, "best.pt")
            torch.save(best_payload, best_path)
            with open(os.path.join(run_dir, "best.json"), "w") as f:
                json.dump({"epoch": epoch, "reward_mean": best_reward,
                           **{f"reward_{k}": log.get(f"reward_{k}") for k in epoch_parts}}, f, indent=2)
            print(f"[ckpt] new best at epoch {epoch} "
                  f"({'eval_reward' if 'eval_reward' in log else 'reward'}={best_reward:.4f}) "
                  f"-> {os.path.abspath(best_path)}")

        save_training_curves(run_dir, history,
                             "+".join(t.upper() for t in reward_fn.terms), best_epoch)

        if args.save_every and (epoch % args.save_every == 0):
            ckpt = os.path.join(run_dir, f"student_ep{epoch:04d}.pt")
            torch.save(payload, ckpt)
            periodic_ckpts.append(ckpt)
            while len(periodic_ckpts) > args.num_checkpoint_limit:
                old = periodic_ckpts.pop(0)
                if os.path.exists(old):
                    os.remove(old)
            print(f"[ckpt] {ckpt}")

    if wandb_run is not None:
        wandb_run.finish()
    del student, ref_net, optimizer
    torch.cuda.empty_cache()
    print(f"[done] steps={num_steps}, best reward {best_reward:.4f} -> {run_dir}/best.pt")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.run_name:
        terms = args.reward_fn.replace("+", "_").replace(",", "_")
        args.run_name = (f"PPO_{terms}_kl{args.kl_coeff}_imagenet64"
                         f"_bs{args.sample_batch_size}_lr{args.lr}_eps{args.clip_range}")
    print(f"[init] run_name={args.run_name}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = args.device
    autocast_ctx = make_autocast(device, None)

    steps_list = [int(s) for s in str(args.student_steps).replace(" ", "").split(",") if s]
    print(f"[init] ablation over student step counts: {steps_list}")

    # --- teacher ------------------------------------------------------------------------------
    print("[init] loading EDM teacher ...")
    teacher = load_edm(args.teacher_pkl, device, cache_dir=args.ckpt_dir,
                       edm_repo=args.edm_repo).eval().requires_grad_(False)
    print(f"[init] resolution={teacher.img_resolution}, label_dim={teacher.label_dim}, "
          f"params={sum(p.numel() for p in teacher.parameters()) / 1e6:.0f}M")
    assert teacher.label_dim > 0, "The ImageNet-64 EDM checkpoint must be class-conditional."

    # EDM's native mixed precision: fp16 trunk, fp32 parameters and fp32 preconditioned output.
    teacher.use_fp16 = bool(args.teacher_fp16)
    print(f"[init] teacher use_fp16={teacher.use_fp16}")

    # --- pool + teacher targets (shared by every run) -----------------------------------------
    blob = None if args.online_teacher else peek_teacher_cache(args.teacher_cache)
    if (blob is not None and args.reuse_cache_pool
            and blob.get("pool_seed") == args.pool_seed
            and blob.get("num_pairs") != args.num_pairs):
        # Same seed, different size: adopt the cache's pool size instead of throwing away hours of
        # teacher sampling. Subsetting is not an option -- see peek_teacher_cache.
        print(f"[cache] adopting the cached pool size: --num_pairs {args.num_pairs} -> "
              f"{blob['num_pairs']} (use --reuse_cache_pool 0 to regenerate instead)")
        args.num_pairs = blob["num_pairs"]

    pool = PairPool(args.num_pairs, args.num_classes_used, teacher.label_dim,
                    teacher.img_channels, teacher.img_resolution, args.pool_seed)
    # Weight source for the behaviour-cloning init of each student (kept on CPU).
    base_net = copy.deepcopy(teacher).cpu()
    base_net.use_fp16 = bool(args.student_fp16)

    if args.online_teacher:
        print(f"[init] ONLINE teacher mode: {args.teacher_steps}-step Heun inside every rollout batch "
              f"(~{2 * args.teacher_steps - 1} NFE per sample per epoch). Consider --teacher_steps 32.")
        teacher_images, teacher_net = None, teacher
    else:
        teacher_images = build_teacher_cache(args, teacher, pool, device, autocast_ctx, blob=blob)
        if args.precompute_only:
            print("[done] teacher cache written; exiting (--precompute_only).")
            return
        teacher_net = None
        del teacher
    torch.cuda.empty_cache()

    reward_fn = RewardBank(args, device)

    for num_steps in steps_list:
        train_student(args, num_steps, base_net, pool, teacher_images, reward_fn, device, autocast_ctx,
                      teacher_net=teacher_net)

    print("[done] all runs finished")


if __name__ == "__main__":
    main()