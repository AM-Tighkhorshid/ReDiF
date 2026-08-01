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
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])

    # --- sampling schedule --------------------------------------------------------------------
    p.add_argument("--student_steps", type=str, default="4,8",
                   help="Comma-separated list of student step counts. Each entry is trained as a separate "
                        "run (fresh behaviour-cloning init) but shares the same (z_T, class) pool and the "
                        "same cached teacher targets, so the 4-step vs 8-step ablation is exactly matched.")
    p.add_argument("--teacher_steps", type=int, default=256, help="Teacher Heun steps (NFE = 2*steps-1).")
    p.add_argument("--sigma_min", type=float, default=0.002)
    p.add_argument("--sigma_max", type=float, default=80.0)
    p.add_argument("--rho", type=float, default=7.0)
    p.add_argument("--eta", type=float, default=0.3,
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
    p.add_argument("--teacher_cache", type=str, default="./teacher_cache_in64.pt")
    p.add_argument("--precompute_only", action="store_true")
    p.add_argument("--precompute_batch", type=int, default=64)

    # --- RL / PPO -----------------------------------------------------------------------------
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--sample_batch_size", type=int, default=16, help="Rollout batch size (no grad).")
    p.add_argument("--samples_per_epoch", type=int, default=256, help="Must be a multiple of sample_batch_size.")
    p.add_argument("--train_batch_size", type=int, default=8, help="PPO minibatch, in (sample, timestep) pairs.")
    p.add_argument("--grad_accum", type=int, default=4, help="Effective batch = train_batch_size * grad_accum.")
    p.add_argument("--inner_epochs", type=int, default=1, help="PPO epochs over each rollout buffer.")
    p.add_argument("--clip_range", type=float, default=1e-4,
                   help="PPO clip range. Small because log-probs are averaged over pixels (DDPO convention).")
    p.add_argument("--adv_clip_max", type=float, default=5.0)
    p.add_argument("--per_class_stats", action="store_true", default=True,
                   help="Per-class reward normalisation (analogue of DDPO's per-prompt stat tracker).")
    p.add_argument("--stat_buffer", type=int, default=32)
    p.add_argument("--stat_min_count", type=int, default=8)
    p.add_argument("--kl_coeff", type=float, default=1.0,
                   help="Coefficient of the per-step Gaussian KL to the frozen reference policy.")

    # --- optimisation -------------------------------------------------------------------------
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--adam_eps", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # --- reward -------------------------------------------------------------------------------
    p.add_argument("--reward_fn", type=str, default="clip+dino",
                   choices=["clip", "dino", "clip+dino"])
    p.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--dino_model", type=str, default="dinov2_vitb14")
    p.add_argument("--reward_resolution", type=int, default=224,
                   help="Encoders expect 224px; ImageNet-64 samples are bicubically upsampled.")

    # --- logging / io -------------------------------------------------------------------------
    p.add_argument("--output_dir", type=str, default="./logs/redif_edm_in64")
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--sample_every", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="redif-edm-imagenet")

    return p.parse_args()


# -----------------------------------------------------------------------------------------------
# EDM helpers
# -----------------------------------------------------------------------------------------------


def load_edm(pkl_path, device):
    """Load an EDM network from an NVlabs pickle. Returns the EMA precond model."""
    try:
        import dnnlib  # noqa: F401  (from the NVlabs/edm repo)
        from dnnlib.util import open_url
        opener = open_url
    except Exception:  # local file, no dnnlib available
        opener = lambda p: open(p, "rb")  # noqa: E731
    with opener(pkl_path) as f:
        data = pickle.load(f)
    net = data["ema"] if isinstance(data, dict) and "ema" in data else data
    net = net.to(device)
    # Precision is handled by autocast; disable the network's internal fp16 cast for stable training.
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


def edm_step_mean(net, x, sigma_from, sigma_down, labels):
    """Euler step towards sigma_down. Returns (mean, denoised).

    mean = x + (sigma_down - sigma_from) * (x - D_theta(x, sigma_from)) / sigma_from
    """
    sigma_b = torch.full((x.shape[0],), sigma_from, device=x.device, dtype=torch.float32)
    denoised = net(x, sigma_b, labels)
    d = (x - denoised) / sigma_from
    mean = x + (sigma_down - sigma_from) * d
    return mean, denoised


def gaussian_logprob(x_next, mean, sigma_up):
    """Isotropic Gaussian log-density, averaged over pixel dimensions (DDPO convention)."""
    var = sigma_up ** 2
    lp = -((x_next - mean) ** 2) / (2 * var) - math.log(sigma_up) - 0.5 * math.log(2 * math.pi)
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


def build_teacher_cache(args, teacher, pool, device, autocast_ctx):
    """Generate (and cache) one teacher image per pool entry. Runs once."""
    if os.path.exists(args.teacher_cache):
        blob = torch.load(args.teacher_cache, map_location="cpu")
        if blob["num_pairs"] == pool.n and blob["pool_seed"] == args.pool_seed:
            print(f"[cache] loaded {args.teacher_cache} "
                  f"({blob['num_pairs']} pairs, {blob['teacher_steps']} teacher steps)")
            return blob["images"]
        print("[cache] existing cache does not match the current pool -> regenerating")

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

    torch.save({"images": out, "num_pairs": pool.n, "pool_seed": args.pool_seed,
                "teacher_steps": args.teacher_steps}, args.teacher_cache)
    print(f"[cache] wrote {args.teacher_cache}")
    return out


def to_uint8(x):
    """EDM images live in [-1, 1]."""
    return (x.clamp(-1, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)


def to_unit(x_uint8):
    return x_uint8.to(torch.float32) / 255.0


# -----------------------------------------------------------------------------------------------
# Reward
# -----------------------------------------------------------------------------------------------


class SemanticReward:
    """Terminal reward: cosine similarity between teacher and student embeddings.

    Note on resolution: ImageNet-64 samples are bicubically upsampled to 224 before encoding.
    Both branches receive identical preprocessing, so the comparison stays fair, but absolute
    similarity values are not comparable with the 512px SD/COCO numbers.
    """

    IMNET_MEAN = (0.485, 0.456, 0.406)
    IMNET_STD = (0.229, 0.224, 0.225)
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, args, device):
        self.device = device
        self.res = args.reward_resolution
        self.use_clip = "clip" in args.reward_fn
        self.use_dino = "dino" in args.reward_fn
        self.clip, self.dino = None, None

        if self.use_clip:
            from transformers import CLIPVisionModelWithProjection
            self.clip = CLIPVisionModelWithProjection.from_pretrained(args.clip_model).to(device).eval()
            self.clip.requires_grad_(False)
        if self.use_dino:
            self.dino = torch.hub.load("facebookresearch/dinov2", args.dino_model).to(device).eval()
            self.dino.requires_grad_(False)

        self._buf = {}

    def _prep(self, x01, mean, std):
        x = F.interpolate(x01, size=(self.res, self.res), mode="bicubic", align_corners=False).clamp(0, 1)
        m = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
        s = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
        return (x - m) / s

    @torch.no_grad()
    def embed(self, x01):
        """x01: float tensor in [0, 1], shape [B, 3, H, W]. Returns dict of L2-normalised embeddings."""
        out = {}
        if self.use_clip:
            e = self.clip(pixel_values=self._prep(x01, self.CLIP_MEAN, self.CLIP_STD)).image_embeds
            out["clip"] = F.normalize(e.float(), dim=-1)
        if self.use_dino:
            e = self.dino(self._prep(x01, self.IMNET_MEAN, self.IMNET_STD))
            out["dino"] = F.normalize(e.float(), dim=-1)
        return out

    @torch.no_grad()
    def __call__(self, student_imgs_pm1, teacher_imgs_uint8):
        """Returns (total_reward [B], per-encoder dict of similarities)."""
        s = self.embed((student_imgs_pm1.clamp(-1, 1) + 1) / 2)
        t = self.embed(to_unit(teacher_imgs_uint8).to(self.device))
        parts = {k: (s[k] * t[k]).sum(dim=-1) for k in s}
        total = torch.stack(list(parts.values()), dim=0).sum(dim=0)  # sum, as in the COCO version
        return total, parts


class PerClassStatTracker:
    """Reward whitening per conditioning (per class here, per prompt in the COCO version)."""

    def __init__(self, buffer_size, min_count):
        self.buf = defaultdict(lambda: deque(maxlen=buffer_size))
        self.min_count = min_count

    def update(self, classes, rewards):
        classes = np.asarray(classes)
        rewards = np.asarray(rewards)
        adv = np.empty_like(rewards)
        for c in np.unique(classes):
            m = classes == c
            self.buf[int(c)].extend(rewards[m].tolist())
            if len(self.buf[int(c)]) < self.min_count:
                mu, sd = rewards.mean(), rewards.std()
            else:
                arr = np.asarray(self.buf[int(c)])
                mu, sd = arr.mean(), arr.std()
            adv[m] = (rewards[m] - mu) / (sd + 1e-6)
        return adv


# -----------------------------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------------------------


def make_autocast(device, dtype):
    if dtype == "fp32" or device == "cpu":
        return contextlib.nullcontext
    tdtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    return lambda: torch.autocast(device_type="cuda", dtype=tdtype)


def save_grid(images_pm1, path, nrow=8):
    try:
        import torchvision
        torchvision.utils.save_image((images_pm1.clamp(-1, 1) + 1) / 2, path, nrow=nrow)
    except Exception as e:
        print(f"[warn] could not save grid: {e}")


def train_student(args, num_steps, base_net, pool, teacher_images, reward_fn, device, autocast_ctx):
    """One complete PPO distillation run for a given student step count.

    Called once per entry of --student_steps. Every run starts from a fresh behaviour-cloning init
    but reuses the same pool and the same cached teacher targets, so the 4-step vs 8-step comparison
    differs only in the discretisation.
    """
    run_dir = os.path.join(args.output_dir, f"steps{num_steps}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "num_steps": num_steps}, f, indent=2)

    # Student = behaviour-cloning init from the teacher weights; reference = frozen copy for the KL term.
    student = copy.deepcopy(base_net).to(device).train().requires_grad_(True)
    ref_net = (copy.deepcopy(base_net).to(device).eval().requires_grad_(False)
               if args.kl_coeff > 0 else None)

    stat_tracker = PerClassStatTracker(args.stat_buffer, args.stat_min_count) if args.per_class_stats else None
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr,
                                  betas=(args.adam_beta1, args.adam_beta2),
                                  weight_decay=args.weight_decay, eps=args.adam_eps)

    sigmas = karras_sigmas(num_steps + 1, args.sigma_min, args.sigma_max, args.rho,
                           device, end_at_sigma_min=True)
    # sigmas has num_steps+1 entries -> num_steps stochastic transitions.
    step_coeffs = [ancestral_coeffs(float(sigmas[i]), float(sigmas[i + 1]), args.eta)
                   for i in range(num_steps)]
    print(f"[run steps={num_steps}] sigmas:", [round(float(s), 4) for s in sigmas])
    print(f"[run steps={num_steps}] sigma_up per step:", [round(c[1], 4) for c in step_coeffs])

    wandb_run = None
    if args.use_wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=f"steps{num_steps}",
                               config={**vars(args), "num_steps": num_steps}, reinit=True)

    global_step = 0
    for epoch in range(args.num_epochs):
        # =========================================================================== ROLLOUT ===
        student.eval()
        buffers, epoch_rewards, epoch_parts = [], [], defaultdict(list)
        n_batches = args.samples_per_epoch // args.sample_batch_size

        for _ in range(n_batches):
            idx = torch.randint(0, pool.n, (args.sample_batch_size,))
            lat, cls = pool.batch(idx)
            lat = lat.to(device)
            labels = one_hot_labels(cls, student.label_dim, device)

            traj = student_rollout(student, ref_net, lat, labels, sigmas, args.eta, autocast_ctx)
            rewards, parts = reward_fn(traj["images"], teacher_images[idx])

            buffers.append({
                "latents_cur": traj["latents_cur"].cpu(),
                "latents_next": traj["latents_next"].cpu(),
                "log_probs": traj["log_probs"].cpu(),
                "ref_means": traj["ref_means"].to(torch.float16).cpu(),
                "labels": labels.cpu(),
                "classes": cls.clone(),
                "rewards": rewards.cpu(),
            })
            epoch_rewards.append(rewards.cpu())
            for k, v in parts.items():
                epoch_parts[k].append(v.cpu())

        rewards_all = torch.cat(epoch_rewards).numpy()
        classes_all = torch.cat([b["classes"] for b in buffers]).numpy()
        if stat_tracker is not None:
            advantages = stat_tracker.update(classes_all, rewards_all)
        else:
            advantages = (rewards_all - rewards_all.mean()) / (rewards_all.std() + 1e-6)
        advantages = np.clip(advantages, -args.adv_clip_max, args.adv_clip_max)

        # Flatten the buffer into (sample, timestep) transitions.
        flat = {k: torch.cat([b[k] for b in buffers], dim=0)
                for k in ["latents_cur", "latents_next", "log_probs", "ref_means", "labels"]}
        flat["advantages"] = torch.from_numpy(advantages).float()
        n_samples = flat["latents_cur"].shape[0]

        # =========================================================================== PPO =======
        student.train()
        stats = defaultdict(list)
        for _ in range(args.inner_epochs):
            pairs = [(s, t) for s in range(n_samples) for t in range(num_steps)]
            random.shuffle(pairs)

            optimizer.zero_grad(set_to_none=True)
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
                    pg = torch.max(unclipped, clipped)

                    # Analytic KL between two Gaussians with equal (isotropic) variance:
                    # KL = ||mu_theta - mu_ref||^2 / (2 * sigma_up^2), averaged over pixels.
                    kl = ((mean - ref_mean[m]) ** 2).mean(dim=(1, 2, 3)) / (2 * s_up ** 2)

                    loss_terms.append(pg)
                    kl_terms.append(kl)
                    ratios.append(ratio.detach())

                pg_loss = torch.cat(loss_terms).mean()
                kl_loss = torch.cat(kl_terms).mean()
                loss = (pg_loss + args.kl_coeff * kl_loss) / args.grad_accum
                loss.backward()

                stats["pg_loss"].append(pg_loss.item())
                stats["kl"].append(kl_loss.item())
                stats["ratio"].append(torch.cat(ratios).mean().item())

                if ((mb_i // args.train_batch_size) + 1) % args.grad_accum == 0:
                    gn = torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    stats["grad_norm"].append(float(gn))
                    global_step += 1

        # =========================================================================== LOG =======
        log = {
            "steps": num_steps,
            "epoch": epoch,
            "reward_mean": float(rewards_all.mean()),
            "reward_std": float(rewards_all.std()),
            **{f"reward_{k}": float(torch.cat(v).mean()) for k, v in epoch_parts.items()},
            **{k: float(np.mean(v)) for k, v in stats.items() if len(v)},
        }
        print(json.dumps(log), flush=True)
        with open(os.path.join(run_dir, "log.jsonl"), "a") as f:
            f.write(json.dumps(log) + "\n")
        if wandb_run is not None:
            wandb_run.log(log, step=global_step)

        if args.sample_every and (epoch % args.sample_every == 0):
            with torch.no_grad():
                idx = torch.arange(0, min(16, pool.n))
                lat, cls = pool.batch(idx)
                labels = one_hot_labels(cls, student.label_dim, device)
                traj = student_rollout(student, None, lat.to(device), labels, sigmas, args.eta, autocast_ctx)
            save_grid(traj["images"], os.path.join(run_dir, f"student_ep{epoch:04d}.png"))
            save_grid(to_unit(teacher_images[idx]).to(device) * 2 - 1,
                      os.path.join(run_dir, "teacher_reference.png"))

        if args.save_every and (epoch % args.save_every == 0):
            ckpt = os.path.join(run_dir, f"student_ep{epoch:04d}.pt")
            torch.save({"model": student.state_dict(), "epoch": epoch,
                        "num_steps": num_steps, "args": vars(args)}, ckpt)
            print(f"[ckpt] {ckpt}")

    if wandb_run is not None:
        wandb_run.finish()
    del student, ref_net, optimizer
    torch.cuda.empty_cache()
    print(f"[done] steps={num_steps}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    autocast_ctx = make_autocast(device, args.dtype)

    steps_list = [int(s) for s in str(args.student_steps).replace(" ", "").split(",") if s]
    print(f"[init] ablation over student step counts: {steps_list}")

    # --- teacher ------------------------------------------------------------------------------
    print("[init] loading EDM teacher ...")
    teacher = load_edm(args.teacher_pkl, device).eval().requires_grad_(False)
    print(f"[init] resolution={teacher.img_resolution}, label_dim={teacher.label_dim}, "
          f"params={sum(p.numel() for p in teacher.parameters()) / 1e6:.0f}M")
    assert teacher.label_dim > 0, "The ImageNet-64 EDM checkpoint must be class-conditional."

    # --- pool + teacher targets (shared by every run) -----------------------------------------
    pool = PairPool(args.num_pairs, args.num_classes_used, teacher.label_dim,
                    teacher.img_channels, teacher.img_resolution, args.pool_seed)
    teacher_images = build_teacher_cache(args, teacher, pool, device, autocast_ctx)
    if args.precompute_only:
        print("[done] teacher cache written; exiting (--precompute_only).")
        return

    # The teacher is never needed on GPU again (all targets are cached); keep it on CPU purely as
    # the weight source for the behaviour-cloning init of each student.
    base_net = teacher.cpu()
    torch.cuda.empty_cache()

    reward_fn = SemanticReward(args, device)

    for num_steps in steps_list:
        train_student(args, num_steps, base_net, pool, teacher_images, reward_fn, device, autocast_ctx)

    print("[done] all runs finished")


if __name__ == "__main__":
    main()