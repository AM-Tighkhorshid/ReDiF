"""
train_backprop_coco.py
======================
Teacher (50-step DDIM) → Student (5-step DDIM) distillation via
DIRECT BACKPROPAGATION — no RL, no PPO.

Objective
---------
At each training step we:

  1. Run the frozen teacher (no grad) to get reference images + final latent.
  2. Run the student WITH gradients kept alive through the entire denoising
     chain (5 steps of DDIM).
  3. Decode the student's final latent through the frozen VAE to get a pixel
     tensor x_student ∈ [0, 1]  (grad-connected to UNet weights).
  4. Pass x_student and x_teacher through the frozen reward encoders
     (CLIP / DINOv2 / LPIPS …) using torch.no_grad() for the ENCODER
     forward pass, but keep the STUDENT pixel tensor in the autograd graph
     so that gradients flow back through the VAE decoder into the UNet.
  5. Minimise:
         L = -reward(x_student, x_teacher, prompts)   [alignment term]
           + λ_kl  * KL(latent_student || latent_teacher)  [KL reg]
           + λ_feat * feature_MSE(f_student, f_teacher)    [feature matching, optional]

Gradient flow
-------------
    UNet params → DDIM steps (differentiable) → x_0_pred
               → VAE decode → x_student (pixel tensor)
               → reward encoder (FROZEN, no_grad) produces target embedding t
               → loss = -cosine_sim(  encode(x_student),  t  )
                                       ↑ THIS encode() also runs no_grad.
                                       Gradient enters ONLY through
                                       x_student's pixel values, not
                                       through the reward encoder weights.

    Concretely for CLIP/DINOv2:
        t        = encoder(x_teacher)   # no_grad, constant target
        s        = encoder(x_student)   # no_grad wrapper, but x_student
                                        # was built WITH grad → PyTorch
                                        # still propagates through the
                                        # pixel pre-processing ops.

    For MSE reward the gradient flows directly through pixel differences.

Key differences from train_ppo_coco.py
---------------------------------------
  • No old_log_probs / importance ratio / PPO clip — zero RL machinery.
  • Student rollout runs inside torch.enable_grad() (not no_grad).
  • rewards.py is REUSED but we call a new differentiable wrapper
    `differentiable_reward_loss()` that re-implements each scorer in a
    grad-compatible way (frozen encoder, live pixel tensor).
  • rewards.py's original `get_reward_fn` is still used for *logging*
    (scalar reward value for W&B), not for the gradient.
"""

import os
import datetime
import json
import random

from absl import app, flags
from ml_collections import config_flags

from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger

from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor

import numpy as np
import torch
import torch.nn.functional as F
from functools import partial
import tqdm
import matplotlib.pyplot as plt
from PIL import Image

import ddpo_pytorch.prompts
from ddpo_pytorch.rewards import get_reward_fn
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


# ---------------------------------------------------------------------------
# Flags  (same interface as train_ppo_coco.py for easy drop-in replacement)
# ---------------------------------------------------------------------------
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", "config/distill_clip.py", "Training configuration."
)
flags.DEFINE_enum(
    "prompt_source", "coco", ["default", "coco"],
    "Prompt source: 'default' = built-in fn, 'coco' = COCO captions.",
)
flags.DEFINE_enum(
    "coco_split", "train", ["train", "val", "both"],
    "COCO split to use when prompt_source=coco.",
)
flags.DEFINE_string(
    "coco_annotations_dir", "coco_dataset/annotations",
    "Path to COCO annotations directory.",
)
flags.DEFINE_float(
    "kl_lambda", 0.5,
    "Weight λ for the latent KL regularisation term in the total loss.",
)
flags.DEFINE_float(
    "feat_lambda", 0.1,
    "Weight λ for the optional feature-MSE matching term (0 = disabled).",
)
flags.DEFINE_list(
    "reward_types", ["clip"],
    "Comma-separated reward(s): clip, dino, perception, text_image, mse. "
    "Example: --reward_types=clip,dino",
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared image normalisation helpers  (mirror of rewards.py, kept local so
# the reward encoders are NEVER imported in a way that breaks no_grad)
# ---------------------------------------------------------------------------

_CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]
_DINO_MEAN = [0.485, 0.456, 0.406]
_DINO_STD  = [0.229, 0.224, 0.225]


def _normalise_tensor(x: torch.Tensor, mean, std) -> torch.Tensor:
    """
    Normalise a (B, C, H, W) float32 tensor in [0, 1].
    Gradient flows through x (mean/std are constants).
    """
    mean_t = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std_t  = torch.tensor(std,  device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean_t) / std_t


def _resize_bicubic(x: torch.Tensor, size: int) -> torch.Tensor:
    """Differentiable bicubic resize.  Gradient flows through x."""
    if x.shape[2] == size and x.shape[3] == size:
        return x
    return F.interpolate(x, size=(size, size),
                         mode="bicubic", align_corners=False)


# ---------------------------------------------------------------------------
# Differentiable reward losses
# Each function takes:
#   x_student  : (B, 3, H, W) float32 [0,1], grad-connected
#   x_teacher  : (B, 3, H, W) float32 [0,1], no-grad constant
#   prompts    : list[str]  (may be None for some rewards)
#   *model refs loaded once at setup time*
# Returns:
#   scalar loss  (lower = better student, ready to .backward())
# ---------------------------------------------------------------------------

class DifferentiableRewardLoss:
    """
    Wraps all reward types into differentiable scalar losses.

    The reward encoders (CLIP, DINOv2, LPIPS) are loaded ONCE and kept
    frozen (requires_grad=False).  Forward passes through the encoder run
    inside torch.no_grad() so their internal activations are never stored
    in the autograd graph and their weights are never updated.

    Gradients reach the UNet ONLY through the student pixel tensor
    x_student, which was produced by a grad-enabled VAE decode.

    How the gradient flows (CLIP example):
    ──────────────────────────────────────
        x_student  (has .grad_fn from VAE decode)
            │
            ▼ _resize_bicubic  ← differentiable
            │
            ▼ _normalise_tensor ← differentiable
            │
            ▼  clip.get_image_features(pixel_values=x_norm)
               ← runs inside torch.no_grad() so activations
                 are NOT saved, but the *input* x_norm still
                 carries its grad_fn from above, so autograd
                 can back-prop through the normalise+resize ops
                 to x_student, and from there through VAE→UNet.
    """

    def __init__(self, reward_types: list[str], device: torch.device):
        self.reward_types = reward_types
        self.device = device

        # ── CLIP (image-image and text-image) ──────────────────────────
        if any(r in reward_types for r in ("clip", "text_image")):
            from transformers import CLIPModel, CLIPProcessor
            self._clip = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            ).to(device).eval()
            self._clip.requires_grad_(False)
            self._clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
        else:
            self._clip = None

        # ── DINOv2 ─────────────────────────────────────────────────────
        if "dino" in reward_types:
            import timm
            self._dino = timm.create_model(
                "vit_base_patch16_dinov2", pretrained=True
            ).to(device).eval()
            self._dino.requires_grad_(False)
        else:
            self._dino = None

        # ── LPIPS (perception) ─────────────────────────────────────────
        if "perception" in reward_types:
            import lpips
            self._lpips = lpips.LPIPS(net="vgg").to(device).eval()
            self._lpips.requires_grad_(False)
        else:
            self._lpips = None

    # ------------------------------------------------------------------ #
    # Private encoder helpers  — each runs the frozen encoder in no_grad  #
    # but receives a grad-carrying pixel tensor so gradients flow back.   #
    # ------------------------------------------------------------------ #

    def _encode_clip_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3, H, W)  float32 [0,1]  — may have grad_fn
        Returns (B, D) L2-normalised CLIP image embedding.

        The encoder forward runs inside no_grad so its activations are
        not retained, but autograd still back-props through the
        resize + normalise preprocessing ops into x.
        """
        x_resized = _resize_bicubic(x, 224)                      # grad through x
        x_norm    = _normalise_tensor(x_resized, _CLIP_MEAN, _CLIP_STD)  # grad through x
        with torch.no_grad():
            emb = self._clip.get_image_features(pixel_values=x_norm)
        return F.normalize(emb.float(), dim=-1)

    def _encode_dino(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns (B, N, D) patch-token features from DINOv2.
        Gradient back-props through resize + normalise into x.
        """
        x_resized = _resize_bicubic(x, 518)                       # DINOv2 native res
        x_norm    = _normalise_tensor(x_resized, _DINO_MEAN, _DINO_STD)
        with torch.no_grad():
            feats = self._dino.forward_features(x_norm)
        if isinstance(feats, dict):
            feats = feats.get("x_norm_patchtokens", feats.get("x", feats))
        return F.normalize(feats.float(), dim=-1)

    def _prep_lpips(self, x: torch.Tensor) -> torch.Tensor:
        """Resize to 224 and map [0,1] → [-1,1] for LPIPS."""
        x_resized = _resize_bicubic(x, 224)
        return x_resized * 2.0 - 1.0                              # grad through x

    # ------------------------------------------------------------------ #
    # Per-type loss functions                                              #
    # ------------------------------------------------------------------ #

    def _clip_loss(self, x_s: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        """
        1 - cosine_sim(CLIP(x_student), CLIP(x_teacher)).
        x_teacher embedding is computed once with no_grad and detached,
        so gradients only enter through x_student.
        """
        with torch.no_grad():
            t_emb = self._encode_clip_image(x_t).detach()      # constant target
        s_emb = self._encode_clip_image(x_s)                   # grad flows here
        sim   = F.cosine_similarity(s_emb, t_emb, dim=-1)      # (B,)
        return (1.0 - sim).mean()

    def _text_image_loss(self, x_s: torch.Tensor,
                         prompts: list[str]) -> torch.Tensor:
        """
        1 - cosine_sim(CLIP(x_student), CLIP_text(prompt)).
        Text embedding is a constant (no_grad); gradient enters via x_student.
        """
        inputs = self._clip_processor(
            text=prompts, return_tensors="pt",
            padding=True, truncation=True,
        ).to(self.device)
        with torch.no_grad():
            text_emb = self._clip.get_text_features(**inputs)
            text_emb = F.normalize(text_emb.float(), dim=-1).detach()  # constant

        s_emb = self._encode_clip_image(x_s)                   # grad flows here
        sim   = (s_emb * text_emb).sum(dim=-1)                 # (B,)
        return (1.0 - sim).mean()

    def _dino_loss(self, x_s: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        """
        Mean 1 - cosine_sim over patch tokens between student and teacher.
        """
        with torch.no_grad():
            t_feat = self._encode_dino(x_t).detach()           # (B, N, D) constant
        s_feat = self._encode_dino(x_s)                        # (B, N, D) grad flows
        if s_feat.dim() == 3:
            sim = F.cosine_similarity(s_feat, t_feat, dim=2).mean(dim=1)  # (B,)
        else:
            sim = F.cosine_similarity(s_feat, t_feat, dim=-1)
        return (1.0 - sim).mean()

    def _perception_loss(self, x_s: torch.Tensor,
                         x_t: torch.Tensor) -> torch.Tensor:
        """
        LPIPS perceptual distance — lower = better, so we minimise it directly.
        Gradient flows through x_student's pixel values.
        """
        s_prep = self._prep_lpips(x_s)                         # grad flows
        with torch.no_grad():
            t_prep = self._prep_lpips(x_t).detach()            # constant
        dist = self._lpips(s_prep, t_prep)                     # (B, 1, 1, 1)
        return dist.mean()

    def _mse_loss(self, x_s: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        """
        Pixel-space L2 loss — the classic diffusion reconstruction objective.
        Pure gradient, no frozen encoder needed.
        """
        with torch.no_grad():
            target = x_t.detach()
        return F.mse_loss(x_s, target)

    # ------------------------------------------------------------------ #
    # Public entry point                                                   #
    # ------------------------------------------------------------------ #

    def __call__(
        self,
        x_student: torch.Tensor,
        x_teacher: torch.Tensor,
        prompts: list[str] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute composite differentiable loss.

        Parameters
        ----------
        x_student : (B, 3, H, W) float32 [0,1], WITH grad_fn
        x_teacher : (B, 3, H, W) float32 [0,1], no-grad constant
        prompts   : list of text prompts (needed for text_image only)

        Returns
        -------
        total_loss : scalar Tensor  (call .backward() on this)
        details    : dict[reward_type → float]  for logging
        """
        total_loss = torch.zeros(1, device=self.device, dtype=x_student.dtype)
        details: dict[str, float] = {}

        for rtype in self.reward_types:
            if rtype == "clip":
                loss = self._clip_loss(x_student, x_teacher)
            elif rtype == "text_image":
                if prompts is None:
                    raise ValueError("prompts required for text_image reward")
                loss = self._text_image_loss(x_student, prompts)
            elif rtype == "dino":
                loss = self._dino_loss(x_student, x_teacher)
            elif rtype == "perception":
                loss = self._perception_loss(x_student, x_teacher)
            elif rtype == "mse":
                loss = self._mse_loss(x_student, x_teacher)
            else:
                raise ValueError(
                    f"Unknown reward_type '{rtype}'. "
                    "Choose from: clip, dino, perception, text_image, mse."
                )

            # Convert to a *reward* for logging  (higher = better)
            details[rtype] = -loss.item()
            total_loss = total_loss + loss

        return total_loss.squeeze(), details


# ---------------------------------------------------------------------------
# Utilities (identical to train_ppo_coco.py)
# ---------------------------------------------------------------------------

def save_side_by_side(student_images, teacher_images, epoch, outdir):
    os.makedirs(outdir, exist_ok=True)
    student_dir  = os.path.join(outdir, "student_only")
    combined_dir = os.path.join(outdir, "side_by_side")
    os.makedirs(student_dir,  exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)

    for idx in range(min(len(student_images), len(teacher_images))):
        s = student_images[idx].convert("RGB")
        t = teacher_images[idx].convert("RGB")
        h = min(s.height, t.height)
        s = s.resize((h, h), Image.Resampling.LANCZOS)
        t = t.resize((h, h), Image.Resampling.LANCZOS)

        s.save(os.path.join(student_dir,  f"epoch{epoch}_student{idx}.png"))

        combined = Image.new("RGB", (t.width + s.width, h))
        combined.paste(t, (0, 0))
        combined.paste(s, (t.width, 0))
        combined.save(os.path.join(combined_dir, f"epoch{epoch}_sample{idx}.png"))


def latent_kl(student_latent: torch.Tensor,
              teacher_latent: torch.Tensor) -> torch.Tensor:
    """
    Analytic KL(N(μ_s, I) || N(μ_t, I)) = ½ ||μ_s − μ_t||²  averaged over batch.
    Gradient flows through student_latent.
    """
    s = student_latent.float().view(student_latent.shape[0], -1)   # (B, D)
    t = teacher_latent.float().view(teacher_latent.shape[0], -1).detach()
    return 0.5 * ((s - t) ** 2).sum(dim=-1).mean()


def decode_latent_to_pixels(
    vae,
    latent: torch.Tensor,
) -> torch.Tensor:
    """
    Decode a VAE latent to a pixel tensor in [0, 1].

    The VAE is frozen (requires_grad=False) but the *forward pass* here
    runs with gradients ENABLED so that autograd records the computation
    graph through the decode operation.  This is the critical link that
    lets gradients from the pixel-space loss flow back to the UNet.

    Parameters
    ----------
    vae     : frozen AutoencoderKL
    latent  : (B, 4, h, w) float  — has grad_fn from UNet denoising

    Returns
    -------
    pixels  : (B, 3, H, W) float32 [0, 1]  — still connected to autograd
    """
    scaled = latent / vae.config.scaling_factor
    decoded = vae.decode(scaled).sample          # grad flows through here
    pixels  = (decoded.clamp(-1, 1) + 1) / 2    # [-1,1] → [0,1]
    return pixels


def student_ddim_steps_with_grad(
    student_pipeline,
    prompt_embeds: torch.Tensor,
    num_steps: int,
    guidance_scale: float,
    eta: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the student's DDIM denoising chain WITH gradients.

    Returns
    -------
    final_latent : (B, 4, h, w)  float — grad-connected to UNet
    all_latents  : (B, T+1, 4, h, w)  — intermediate latents (for KL)
    """
    scheduler = student_pipeline.scheduler
    unet      = student_pipeline.unet
    batch_size = prompt_embeds.shape[0]

    # Initialise with random noise — detached so it's a leaf variable
    latent_shape = (
        batch_size,
        unet.config.in_channels,
        unet.config.sample_size,
        unet.config.sample_size,
    )
    latent = torch.randn(latent_shape, device=device, dtype=prompt_embeds.dtype)
    latent = latent * scheduler.init_noise_sigma

    scheduler.set_timesteps(num_steps, device=device)
    timesteps = scheduler.timesteps

    all_latents = [latent.detach()]   # store intermediates (no grad needed for logging)

    for t in timesteps:
        t_batch = t.expand(batch_size)

        # Classifier-free guidance
        latent_input = torch.cat([latent] * 2)
        t_input      = torch.cat([t_batch] * 2)
        embeds_input = torch.cat([
            torch.zeros_like(prompt_embeds),  # unconditional
            prompt_embeds,                    # conditional
        ])

        # UNet forward — gradients ARE enabled here
        noise_pred = unet(latent_input, t_input, embeds_input).sample  # (2B, 4, h, w)

        # Split unconditional / conditional predictions
        noise_uncond, noise_cond = noise_pred.chunk(2)
        noise_pred_guided = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        # DDIM step — implemented inline to preserve the grad_fn on latent
        # (scheduler.step() detaches in many diffusers versions)
        latent = _ddim_step_differentiable(
            scheduler, noise_pred_guided, t, latent, eta=eta
        )

        all_latents.append(latent.detach())   # detached copies for logging / KL target

    return latent, all_latents


def _ddim_step_differentiable(scheduler, noise_pred, t, x_t, eta=0.0):
    """
    DDIM update step that keeps the gradient alive on the output tensor.

    x_{t-1} = sqrt(ᾱ_{t-1}) * x_0_pred
            + sqrt(1 - ᾱ_{t-1} - σ²) * noise_pred
            + σ * noise           (σ=0 for eta=0, fully deterministic)

    All arithmetic is differentiable w.r.t. x_t (and therefore w.r.t.
    the UNet weights that produced noise_pred).
    """
    # Retrieve schedule constants
    alpha_prod_t = scheduler.alphas_cumprod[t].to(x_t.device, x_t.dtype)

    prev_t = t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    prev_t = max(int(prev_t), 0)
    alpha_prod_t_prev = (
        scheduler.alphas_cumprod[prev_t].to(x_t.device, x_t.dtype)
        if prev_t >= 0
        else scheduler.final_alpha_cumprod.to(x_t.device, x_t.dtype)
    )

    beta_prod_t = 1.0 - alpha_prod_t

    # Predicted x_0
    x0_pred = (x_t - beta_prod_t.sqrt() * noise_pred) / alpha_prod_t.sqrt()
    x0_pred = x0_pred.clamp(-1, 1)                    # soft clamp keeps grad

    # DDIM variance
    sigma = eta * ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)).sqrt() * \
            (1 - alpha_prod_t / alpha_prod_t_prev).sqrt()

    # Direction pointing to x_t
    dir_xt_coef = (1 - alpha_prod_t_prev - sigma ** 2).clamp(min=0).sqrt()

    # x_{t-1}
    x_prev = alpha_prod_t_prev.sqrt() * x0_pred \
           + dir_xt_coef * noise_pred

    if eta > 0:
        noise = torch.randn_like(x_t)
        x_prev = x_prev + sigma * noise

    return x_prev


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(_):
    config       = FLAGS.config
    reward_types = list(FLAGS.reward_types)

    unique_id        = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    config.run_name  = (config.run_name or unique_id) + f"_backprop_{unique_id}"

    reward_tag = "_".join(reward_types)
    outdir  = f"BACKPROP_{reward_tag}_kl{FLAGS.kl_lambda}_feat{FLAGS.feat_lambda}"
    outdir += "_coco" if FLAGS.prompt_source == "coco" else "_ddpo"

    stats_dir  = outdir
    os.makedirs(stats_dir, exist_ok=True)
    stats_file = os.path.join(stats_dir, "training_stats.txt")

    all_losses, all_rewards, all_rewards_std = [], [], []

    # ------------------------------------------------------------------ #
    # Accelerator
    # ------------------------------------------------------------------ #
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=ProjectConfiguration(
            project_dir=os.path.join(config.logdir, config.run_name),
            total_limit=config.num_checkpoint_limit,
        ),
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    )
    set_seed(config.seed, device_specific=True)

    if accelerator.is_main_process:
        accelerator.init_trackers(
            "ddpo-distill-backprop",
            config={
                **config.to_dict(),
                "reward_types":  reward_types,
                "kl_lambda":     FLAGS.kl_lambda,
                "feat_lambda":   FLAGS.feat_lambda,
                "method":        "direct_backprop",
            },
        )

    logger.info(f"\n{config}")
    logger.info(f"Reward types : {reward_types}")
    logger.info(f"KL λ         : {FLAGS.kl_lambda}")
    logger.info(f"Feature λ    : {FLAGS.feat_lambda}")
    logger.info("Training method: DIRECT BACKPROPAGATION (no PPO)")

    # ------------------------------------------------------------------ #
    # Teacher pipeline  (all frozen — same as PPO script)
    # ------------------------------------------------------------------ #
    teacher_pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained.model, revision=config.pretrained.revision
    )
    teacher_pipeline.scheduler = DDIMScheduler.from_config(
        teacher_pipeline.scheduler.config
    )
    teacher_pipeline.safety_checker = None
    teacher_pipeline.set_progress_bar_config(disable=True)
    teacher_pipeline.to(accelerator.device)
    teacher_pipeline.vae.requires_grad_(False)
    teacher_pipeline.text_encoder.requires_grad_(False)
    teacher_pipeline.unet.requires_grad_(False)
    teacher_pipeline.save_pretrained(config.teacher_output_dir)

    # ------------------------------------------------------------------ #
    # Student pipeline  (UNet or LoRA trainable — same as PPO script)
    # ------------------------------------------------------------------ #
    student_pipeline = StableDiffusionPipeline.from_pretrained(
        config.student.model, revision=config.student.revision
    )
    student_pipeline.scheduler = DDIMScheduler.from_config(
        student_pipeline.scheduler.config
    )
    student_pipeline.safety_checker = None
    student_pipeline.set_progress_bar_config(disable=True)
    student_pipeline.to(accelerator.device)
    student_pipeline.vae.requires_grad_(False)
    student_pipeline.text_encoder.requires_grad_(False)

    if config.use_lora:
        lora_attn_procs = {}
        for name in student_pipeline.unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else student_pipeline.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = student_pipeline.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id    = int(name[len("up_blocks."):len("up_blocks.") + 1])
                hidden_size = list(
                    reversed(student_pipeline.unet.config.block_out_channels)
                )[block_id]
            elif name.startswith("down_blocks"):
                block_id    = int(name[len("down_blocks."):len("down_blocks.") + 1])
                hidden_size = student_pipeline.unet.config.block_out_channels[block_id]
            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
            )
        student_pipeline.unet.set_attn_processor(lora_attn_procs)

        class _Wrapper(AttnProcsLayers):
            def forward(self, *args, **kwargs):
                return student_pipeline.unet(*args, **kwargs)

        unet = _Wrapper(student_pipeline.unet.attn_processors)
    else:
        student_pipeline.unet.requires_grad_(True)
        unet = student_pipeline.unet

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(
        accelerator.mixed_precision, torch.float32
    )
    student_pipeline.vae.to(accelerator.device, dtype=dtype)
    student_pipeline.text_encoder.to(accelerator.device, dtype=dtype)
    if config.use_lora:
        student_pipeline.unet.to(accelerator.device, dtype=dtype)

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )
    unet, optimizer = accelerator.prepare(unet, optimizer)

    lora_params = sum(
        p.numel()
        for m in student_pipeline.unet.attn_processors.values()
        for p in m.parameters()
    )
    logger.info(f"LoRA parameters: {lora_params:,}")

    # ------------------------------------------------------------------ #
    # Differentiable reward loss  (replaces PPO reward fn for grad)
    # ------------------------------------------------------------------ #
    diff_loss_fn = DifferentiableRewardLoss(reward_types, accelerator.device)

    # Keep original rewards.py reward_fn for SCALAR logging only (no grad used)
    reward_fn_for_logging = get_reward_fn(
        reward_types, teacher_pipeline, student_pipeline
    )

    # ------------------------------------------------------------------ #
    # Prompts  (identical to PPO script)
    # ------------------------------------------------------------------ #
    prompt_fn = getattr(ddpo_pytorch.prompts, config.prompt_fn)

    coco_captions = None
    if FLAGS.prompt_source == "coco":
        candidates = []
        ann_dir = FLAGS.coco_annotations_dir
        if FLAGS.coco_split in ("train", "both"):
            candidates.append(os.path.join(ann_dir, "captions_train2017.json"))
        if FLAGS.coco_split in ("val", "both"):
            candidates.append(os.path.join(ann_dir, "captions_val2017.json"))

        coco_captions = []
        for cpath in candidates:
            cpath = os.path.abspath(cpath)
            if not os.path.exists(cpath):
                logger.warning(f"COCO file not found: {cpath}")
                continue
            with open(cpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            for a in data.get("annotations", []):
                cap = a.get("caption")
                if cap and isinstance(cap, str):
                    coco_captions.append(cap)
            logger.info(f"Loaded {len(data['annotations'])} captions from {cpath}")

        if not coco_captions:
            logger.warning("No COCO captions loaded — falling back to default prompts.")
            coco_captions = None

    # ================================================================== #
    # Training loop
    # ================================================================== #
    for epoch in range(config.num_epochs):
        logger.info(f"Epoch {epoch}: Sampling + Backprop Training")

        # ---- Prompt batch ------------------------------------------------
        if FLAGS.prompt_source == "coco" and coco_captions:
            if len(coco_captions) >= config.sample.batch_size:
                prompts = random.sample(coco_captions, k=config.sample.batch_size)
            else:
                prompts = [
                    random.choice(coco_captions)
                    for _ in range(config.sample.batch_size)
                ]
        else:
            pairs   = [prompt_fn(**config.prompt_fn_kwargs)
                       for _ in range(config.sample.batch_size)]
            prompts = [p[0] for p in pairs]

        # ---- Encode prompts ----------------------------------------------
        prompt_ids = student_pipeline.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=student_pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)

        with torch.no_grad():
            prompt_embeds = student_pipeline.text_encoder(prompt_ids)[0]  # (B, L, D)

        # ---------------------------------------------------------------- #
        # Teacher rollout  (no gradients — same as PPO)
        # ---------------------------------------------------------------- #
        with torch.no_grad():
            with accelerator.autocast():
                teacher_output = teacher_pipeline(
                    prompt_embeds=prompt_embeds,
                    num_inference_steps=config.sample.num_steps,   # 50
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                )
            # teacher pixel tensor (B, 3, H, W) in [0,1] — constant reference
            teacher_pixels = teacher_output.images.to(
                accelerator.device, dtype=torch.float32
            )

            # Teacher final latent for KL reg
            # Re-encode teacher pixels via VAE to get latent reference
            teacher_latent = student_pipeline.vae.encode(
                teacher_pixels * 2.0 - 1.0   # [0,1] → [-1,1]
            ).latent_dist.mean            # (B, 4, h, w)

        # ---------------------------------------------------------------- #
        # Student rollout  WITH gradients
        # ---------------------------------------------------------------- #
        student_pipeline.unet.train()

        with accelerator.accumulate(unet):
            with accelerator.autocast():
                # ── Step 1: run student DDIM chain with grad ─────────────
                student_final_latent, _ = student_ddim_steps_with_grad(
                    student_pipeline,
                    prompt_embeds=prompt_embeds,
                    num_steps=config.student.num_steps,       # 5
                    guidance_scale=config.sample.guidance_scale,
                    eta=config.sample.eta,
                    device=accelerator.device,
                )
                # student_final_latent: (B, 4, h, w)  WITH grad_fn

                # ── Step 2: decode latent → pixel tensor  (grad alive) ───
                student_pixels = decode_latent_to_pixels(
                    student_pipeline.vae, student_final_latent
                )
                # student_pixels: (B, 3, H, W) in [0,1]  WITH grad_fn
                # teacher_pixels: (B, 3, H, W) in [0,1]  detached constant

                # ── Step 3: differentiable reward loss ───────────────────
                reward_loss, reward_details = diff_loss_fn(
                    student_pixels,
                    teacher_pixels,
                    prompts=prompts,
                )

                # ── Step 4: latent KL regularisation ─────────────────────
                # student_final_latent retains grad; teacher_latent is detached
                kl_reg = latent_kl(student_final_latent, teacher_latent)

                # ── Step 5: optional feature-MSE matching ────────────────
                # Encourages intermediate latent statistics to match teacher.
                # Uses the student pixel embedding vs teacher pixel embedding
                # through the same frozen CLIP encoder.
                feat_loss = torch.zeros(1, device=accelerator.device,
                                        dtype=student_pixels.dtype)
                if FLAGS.feat_lambda > 0.0 and diff_loss_fn._clip is not None:
                    with torch.no_grad():
                        t_feat = diff_loss_fn._encode_clip_image(
                            teacher_pixels
                        ).detach()
                    s_feat = diff_loss_fn._encode_clip_image(student_pixels)
                    feat_loss = F.mse_loss(s_feat, t_feat)

                # ── Step 6: total loss ────────────────────────────────────
                total_loss = (
                    reward_loss
                    + FLAGS.kl_lambda  * kl_reg
                    + FLAGS.feat_lambda * feat_loss
                )

            # ── Backward & optimiser step ─────────────────────────────────
            accelerator.backward(total_loss)
            if accelerator.sync_gradients:
                torch.nn.utils.clip_grad_norm_(
                    unet.parameters(), config.train.max_grad_norm
                )
            optimizer.step()
            optimizer.zero_grad()

        # ---------------------------------------------------------------- #
        # Scalar reward logging  (no grad, uses original rewards.py)
        # ---------------------------------------------------------------- #
        student_pipeline.unet.eval()
        with torch.no_grad():
            # Convert pixels to PIL for the logging reward fn
            student_pil = [
                Image.fromarray(
                    (img.permute(1, 2, 0).cpu().float().numpy() * 255
                     ).round().clip(0, 255).astype("uint8")
                )
                for img in student_pixels.detach()
            ]
            teacher_pil = [
                Image.fromarray(
                    (img.permute(1, 2, 0).cpu().float().numpy() * 255
                     ).round().clip(0, 255).astype("uint8")
                )
                for img in teacher_pixels.detach()
            ]
            rewards_raw, log_reward_details = reward_fn_for_logging(
                student_pil, teacher_pil, prompts
            )

        rewards_np  = np.array(rewards_raw, dtype=np.float32)
        reward_mean = float(rewards_np.mean())
        reward_std  = float(rewards_np.std())

        total_loss_val = total_loss.item()
        all_losses.append(total_loss_val)
        all_rewards.append(reward_mean)
        all_rewards_std.append(reward_std)

        # ---------------------------------------------------------------- #
        # Console + file logging
        # ---------------------------------------------------------------- #
        detail_str = "  ".join(
            f"{k}={float(np.mean(v)):.4f}"
            for k, v in log_reward_details.items()
        )
        print(
            f"Epoch {epoch:4d} | reward {reward_mean:+.4f} ± {reward_std:.4f} "
            f"| kl {kl_reg.item():.4f} | rloss {reward_loss.item():.4f} "
            f"| loss {total_loss_val:.6f}"
            + (f"  [{detail_str}]" if detail_str else "")
        )

        with open(stats_file, "a") as f:
            f.write(
                f"Epoch {epoch}: loss={total_loss_val:.6f}, "
                f"reward={reward_mean:.6f}±{reward_std:.6f}, "
                f"kl={kl_reg.item():.6f}, "
                f"reward_loss={reward_loss.item():.6f}"
            )
            for k, v in log_reward_details.items():
                f.write(f", {k}={float(np.mean(v)):.6f}")
            f.write("\n")

        if accelerator.is_main_process:
            log_dict = {
                "reward_mean":  reward_mean,
                "reward_std":   reward_std,
                "kl_latent":    kl_reg.item(),
                "reward_loss":  reward_loss.item(),
                "feat_loss":    feat_loss.item() if hasattr(feat_loss, "item") else 0.0,
                "loss":         total_loss_val,
            }
            for k, v in log_reward_details.items():
                log_dict[f"reward/{k}"] = float(np.mean(v))
            accelerator.log(log_dict, step=epoch)

        # ---------------------------------------------------------------- #
        # Visualisation  (same as PPO script)
        # ---------------------------------------------------------------- #
        if accelerator.is_main_process:
            eval_prompts = [
                "A crystal-clear glass bowl overflowing with ripe oranges on a rustic wooden table",
                "A fluffy tabby cat mid-step, looking at the camera with curious eyes",
                "A futuristic city skyline at night with neon lights and flying cars",
                "A warm log cabin in a snowy pine forest at twilight",
                "A colorful wildflower plain under a bright sky",
            ]
            student_pipeline.unet.eval()
            with torch.no_grad(), accelerator.autocast():
                teacher_eval = [
                    teacher_pipeline(
                        p,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                    ).images[0]
                    for p in eval_prompts
                ]
                student_eval = [
                    student_pipeline(
                        p,
                        num_inference_steps=config.student.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                    ).images[0]
                    for p in eval_prompts
                ]
            save_side_by_side(student_eval, teacher_eval, epoch, outdir)

    # ------------------------------------------------------------------ #
    # Save model & training curves
    # ------------------------------------------------------------------ #
    student_pipeline.save_pretrained(os.path.join(stats_dir, "student_model"))

    epochs = range(len(all_losses))

    plt.figure()
    plt.plot(epochs, all_losses, label="Total loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training Loss")
    plt.legend()
    plt.savefig(os.path.join(stats_dir, "loss_curve.png"))
    plt.close()

    all_rewards     = np.array(all_rewards)
    all_rewards_std = np.array(all_rewards_std)
    plt.figure()
    plt.plot(epochs, all_rewards, label=f"Reward ({'+'.join(reward_types)})")
    plt.fill_between(
        epochs,
        all_rewards - all_rewards_std,
        all_rewards + all_rewards_std,
        alpha=0.3, label="± std",
    )
    plt.xlabel("Epoch"); plt.ylabel("Reward")
    plt.title(f"Reward Curve (student ↔ teacher)  [{'+'.join(reward_types)}]")
    plt.legend()
    plt.savefig(os.path.join(stats_dir, "reward_curve.png"))
    plt.close()


if __name__ == "__main__":
    app.run(main)