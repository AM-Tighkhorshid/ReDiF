# train_progressive_coco.py
# Progressive distillation version of your original train_ppo_coco.py
# Replaces the PPO distillation loop with Algorithm 2 (Progressive Distillation).
# Keeps model loading, LoRA injection, and evaluation image saving/printing intact.
# Based on "Progressive Distillation for Fast Sampling of Diffusion Models" (Salimans & Ho, ICLR 2022). :contentReference[oaicite:1]{index=1}

from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
from absl import app, flags
from ml_collections import config_flags
import json
import random
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import numpy as np
import ddpo_pytorch.prompts
from ddpo_pytorch.rewards import get_reward_fn
from ddpo_pytorch.stat_tracking import PerPromptStatTracker
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

def save_side_by_side(student_images, teacher_images, epoch, outdir):
    os.makedirs(outdir, exist_ok=True)

    for idx in range(min(len(student_images), len(teacher_images))):
        s_img = student_images[idx].convert("RGB")
        t_img = teacher_images[idx].convert("RGB")

        h = min(s_img.height, t_img.height)
        s_img = s_img.resize((h, h), Image.Resampling.LANCZOS)
        t_img = t_img.resize((h, h), Image.Resampling.LANCZOS)

        combined = Image.new("RGB", (t_img.width + s_img.width, h))
        combined.paste(t_img, (0, 0))
        combined.paste(s_img, (t_img.width, 0))

        outpath = os.path.join(outdir, f"epoch{epoch}_sample{idx}.png")
        combined.save(outpath)
        print(f"Saved {outpath}")

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/distill_clip.py", "Training configuration.")

# --- New flags for prompt source / COCO ---
flags.DEFINE_enum(
    "prompt_source", "coco", ["default", "coco"],
    "Source of prompts to use: 'default' uses built-in prompt function; 'coco' samples captions from COCO annotations."
)
flags.DEFINE_enum(
    "coco_split", "train", ["train", "val", "both"],
    "Which COCO captions split to use when prompt_source=coco."
)
flags.DEFINE_string(
    "coco_annotations_dir", "coco_dataset/annotations",
    "Path to COCO annotations directory (should contain captions_train2017.json and/or captions_val2017.json)"
)

logger = get_logger(__name__)

def cosine_alpha_sigma(t):
    """
    t may be a tensor or scalar in [0,1]
    α_t = cos(0.5 * pi * t)
    σ_t = sqrt(1 - α_t^2)
    """
    # Ensure torch tensor
    if isinstance(t, torch.Tensor):
        alpha = torch.cos(0.5 * math.pi * t)
        sigma = torch.sqrt(torch.clamp(1.0 - alpha * alpha, min=0.0))
        return alpha, sigma
    else:
        alpha = math.cos(0.5 * math.pi * t)
        sigma = math.sqrt(max(0.0, 1.0 - alpha * alpha))
        return alpha, sigma

def ddim_one_step_from_x_hat(zt, x_hat, alpha_s, sigma_s, alpha_t, sigma_t):
    """
    DDIM update from z_t to z_s (s < t) using a predicted x_hat (in latent space).
    z_s = alpha_s * x_hat + (sigma_s / sigma_t) * ( z_t - alpha_t * x_hat )
    Implemented in vectorized torch-friendly form.
    """
    return alpha_s * x_hat + (sigma_s / sigma_t) * (zt - alpha_t * x_hat)

def compute_x_hat_from_eps_pred(zt, eps_pred, alpha_t, sigma_t):
    """
    Convert predicted eps -> implied x_hat in latent space:
    x_hat = (z_t - sigma_t * eps_pred) / alpha_t
    Avoid division by zero by adding small eps when alpha_t==0 (should be rare; alpha=0 only at t=1).
    """
    # alpha_t may be tensor
    denom = alpha_t
    # if denom might be zero, guard with tiny epsilon
    denom_safe = denom.clone() if isinstance(denom, torch.Tensor) else denom
    if isinstance(denom_safe, torch.Tensor):
        denom_safe = torch.where(denom_safe == 0.0, torch.full_like(denom_safe, 1e-6), denom_safe)
    else:
        if denom_safe == 0.0:
            denom_safe = 1e-6
    return (zt - sigma_t * eps_pred) / denom_safe

def snr_plus_one_weight(alpha_t, sigma_t):
    # w = 1 + alpha_t^2 / sigma_t^2
    # avoid division by 0 if sigma_t == 0 (at t=0 alpha=1->sigma=0). add small eps
    eps = 1e-6
    return 1.0 + (alpha_t * alpha_t) / (sigma_t * sigma_t + eps)

def kl_divergence(p, q):
    p = p.float().view(p.shape[0], -1)
    q = q.float().view(q.shape[0], -1)
    p = F.log_softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    kl = F.kl_div(p, q, reduction='batchmean', log_target=False)
    return kl

def main(_):
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    config.run_name = config.run_name or unique_id
    config.run_name += f"_progressive_{unique_id}"

    outdir = "ProgressiveDistill"
    if FLAGS.prompt_source == "coco":
        outdir = outdir + "_coco_prompts"
    else:
        outdir = outdir + "_ddpo_prompts"

    stats_dir = outdir
    os.makedirs(stats_dir, exist_ok=True)
    stats_file = os.path.join(stats_dir, "distill_stats.txt")

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        total_limit=config.num_checkpoint_limit,
    )
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=getattr(config.train, "gradient_accumulation_steps", 1),
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("progressive-distill", config=config.to_dict())

    logger.info(f"\n{config}")

    # --- Load teacher pipeline (will remain frozen) ---
    teacher_pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained.model, revision=config.pretrained.revision
    )
    teacher_pipeline.scheduler = DDIMScheduler.from_config(teacher_pipeline.scheduler.config)
    teacher_pipeline.safety_checker = None
    teacher_pipeline.set_progress_bar_config(disable=False)
    teacher_pipeline.vae.to(accelerator.device)
    teacher_pipeline.text_encoder.to(accelerator.device)
    teacher_pipeline.unet.to(accelerator.device)
    teacher_pipeline.text_encoder.requires_grad_(False)
    teacher_pipeline.vae.requires_grad_(False)
    teacher_pipeline.unet.requires_grad_(False)
    teacher_pipeline.save_pretrained(config.teacher_output_dir)

    # --- Load student pipeline (we will train its UNet / LoRA) ---
    student_pipeline = StableDiffusionPipeline.from_pretrained(
        config.student.model, revision=config.student.revision
    )
    student_pipeline.scheduler = DDIMScheduler.from_config(student_pipeline.scheduler.config)
    student_pipeline.safety_checker = None
    student_pipeline.set_progress_bar_config(disable=False)
    student_pipeline.vae.to(accelerator.device)
    student_pipeline.text_encoder.to(accelerator.device)
    student_pipeline.unet.to(accelerator.device)
    student_pipeline.vae.requires_grad_(False)
    student_pipeline.text_encoder.requires_grad_(False)
    # UNet will be trainable unless using LoRA
    student_pipeline.unet.requires_grad_(not config.use_lora)

    if config.use_lora:
        lora_attn_procs = {}
        for name in student_pipeline.unet.attn_processors.keys():
            cross_attention_dim = (
                None if name.endswith("attn1.processor") else student_pipeline.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = student_pipeline.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks."):].split(".")[0])
                hidden_size = list(reversed(student_pipeline.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks."):].split(".")[0])
                hidden_size = student_pipeline.unet.config.block_out_channels[block_id]
            lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
        student_pipeline.unet.set_attn_processor(lora_attn_procs)

        class _Wrapper(AttnProcsLayers):
            def forward(self, *args, **kwargs):
                return student_pipeline.unet(*args, **kwargs)
        unet = _Wrapper(student_pipeline.unet.attn_processors)
    else:
        unet = student_pipeline.unet

    dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16

    student_pipeline.vae.to(accelerator.device, dtype=dtype)
    student_pipeline.text_encoder.to(accelerator.device, dtype=dtype)
    if config.use_lora:
        student_pipeline.unet.to(accelerator.device, dtype=dtype)

    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        unet.parameters(),
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    unet, optimizer = accelerator.prepare(unet, optimizer)

    # Default prompt function
    prompt_fn = getattr(ddpo_pytorch.prompts, config.prompt_fn)

    # --- Load COCO captions if selected ---
    coco_captions = None
    if FLAGS.prompt_source == "coco":
        ann_dir = FLAGS.coco_annotations_dir
        candidates = []
        if FLAGS.coco_split in ("train", "both"):
            candidates.append(os.path.join(ann_dir, "captions_train2017.json"))
        if FLAGS.coco_split in ("val", "both"):
            candidates.append(os.path.join(ann_dir, "captions_val2017.json"))

        coco_captions = []
        for cpath in candidates:
            if not os.path.isabs(cpath):
                cpath = os.path.abspath(cpath)
            if not os.path.exists(cpath):
                print(f"COCO caption file not found: {cpath}")
                continue
            try:
                with open(cpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                anns = data.get("annotations", [])
                for a in anns:
                    cap = a.get("caption")
                    if cap and isinstance(cap, str):
                        coco_captions.append(cap)
                print(f"Loaded {len(anns)} annotations from {cpath}")
            except Exception as e:
                print(f"Failed to load COCO captions from {cpath}: {e}")

        if len(coco_captions) == 0:
            print("Warning: --prompt_source=coco selected but no captions were loaded. Falling back to default prompts.")
            coco_captions = None


    # Distillation hyperparameters (can be set in config.distill.*)
    steps_list = getattr(config, "distill", None) and getattr(config.distill, "steps_list", None)
    if steps_list is None:
        steps_list = [50, 25, 12, 5]  # default, as you requested
    updates_per_stage = getattr(config, "distill", None) and getattr(config.distill, "updates_per_stage", None)
    if updates_per_stage is None:
        updates_per_stage = getattr(config.train, "distill_updates_per_stage", 5000)  # default, change in config

    batch_size = config.sample.batch_size

    # Main progressive distillation loop
    # For each student_steps in steps_list:
    #   copy teacher -> initialize student, then run updates for this stage
    teacher_unet = teacher_pipeline.unet
    student_unet = student_pipeline.unet  # keep a direct reference (wrapped version used for training)
    # Note: teacher unet is on accelerator.device from earlier; ensure in eval mode and no grad
    teacher_unet.eval()
    for stage_idx, student_steps in enumerate(steps_list):
        print(f"\n==== Distillation stage {stage_idx+1}/{len(steps_list)}: student_steps = {student_steps} ====\n")
        # Initialize student weights from teacher (copy)
        # Copying weights: state_dict copy (keeps LoRA layers intact if present in student)
        # If using LoRA, the student's base weights should also be initialized to teacher's weights.
        # We'll copy the state_dict for the underlying unet module in student_pipeline to match paper's "θ ← η".
        teacher_state = teacher_unet.state_dict()
        student_unet_state = student_unet.state_dict()
        # Only copy keys that exist in student (safer when LoRA or different wrappers present)
        copied = []
        for k in teacher_state:
            if k in student_unet_state and teacher_state[k].size() == student_unet_state[k].size():
                student_unet_state[k].copy_(teacher_state[k].to(student_unet_state[k].device))
                copied.append(k)
        print(f"Copied {len(copied)} matching parameters from teacher -> student")

        # Put student in train mode
        student_unet.train()

        # Distillation training loop for this stage
        updates_done = 0
        pbar = tqdm(range(updates_per_stage), desc=f"Stage {stage_idx+1} updates", disable=not accelerator.is_main_process)
        while updates_done < updates_per_stage:
            # --- Sample prompts (same as your original code's prompt selection) ---
            if FLAGS.prompt_source == "coco" and coco_captions is not None:
                if len(coco_captions) >= batch_size:
                    prompts = random.sample(coco_captions, k=batch_size)
                else:
                    prompts = [random.choice(coco_captions) for _ in range(batch_size)]
                prompt_pairs = [(p, None) for p in prompts]
                prompt_metadata = [None] * len(prompts)
            else:
                prompt_pairs = [prompt_fn(**config.prompt_fn_kwargs) for _ in range(batch_size)]
                prompts = [p[0] for p in prompt_pairs]
                prompt_metadata = [p[1] for p in prompt_pairs]

            # Encode prompts
            prompt_ids = student_pipeline.tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=student_pipeline.tokenizer.model_max_length,
            ).input_ids.to(accelerator.device)
            prompt_embeds = student_pipeline.text_encoder(prompt_ids)[0]

            # --- Generate teacher samples for these prompts (we'll use the final latents as 'clean x') ---
            with torch.no_grad():
                teacher_out = pipeline_with_logprob(
                    teacher_pipeline,
                    prompt_embeds=prompt_embeds,
                    num_inference_steps=config.sample.num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    eta=config.sample.eta,
                    output_type="pt",
                    return_all_latents=True,
                )
                teacher_images, _, teacher_latents_all, teacher_log_probs_all = teacher_out
                # teacher_latents_all is list (len=num_steps+1) or tensor list; convert to tensor
                # assume teacher_latents_all is list-of-tensors [batch, C, H, W] across timesteps
                # final clean latent x_latent for each prompt:
                # If teacher_latents_all is list where last element [-1] is final sample latent (z_0 ~ alpha_0 x)
                x_latents = teacher_latents_all[-1]  # shape: (batch, C, H, W)
                # If the VAE scaling is used by pipeline (as typical), the latents are in the same scaling as unet expects.

            # Prepare timesteps: sample discrete i in [1..N] uniformly per sample
            # We'll vectorize: sample i for each batch element
            N = int(student_steps)
            # Sample discrete i in 1..N (inclusive) - vector of size batch
            i_vals = torch.randint(low=1, high=N + 1, size=(batch_size,), device=accelerator.device)
            t_vals = i_vals.float() / float(N)  # in [1/N, ..., 1]
            # Convert to tensors shaped for unet: timesteps usually integers for scheduler, but we are using continuous t in cos schedule.
            # For unet, timesteps variable can be encoded via scheduler's timesteps embedding or passed in
            # We'll feed a scalar float as 'time' by mapping into an integer-like placeholder; the UNet expects a diffusion timestep integer.
            # To avoid mismatch we can pass timesteps as torch.full(...) with placeholder indices and rely on using the alpha/sigma schedule explicitly in our math.
            # However UNet from diffusers expects timesteps as integers and will embed them; to be consistent, we map t in [0,1] -> a pseudo-timestep integer by scaling to 1000.
            # This is a pragmatic choice that keeps the conditioning pipeline working.
            pseudo_timestep_scale = 1000  # arbitrary scale to create integer-like timesteps input
            timesteps_for_unet = (t_vals * (pseudo_timestep_scale - 1)).long()

            # Vectorize alpha_t and sigma_t for each element (latent space)
            alpha_t, sigma_t = cosine_alpha_sigma(t_vals)  # returns tensors
            # reshape to broadcast to latent shape
            # x_latents has shape [B, C, H, W]
            alpha_t = alpha_t.view(batch_size, *([1] * (x_latents.dim() - 1)))
            sigma_t = sigma_t.view(batch_size, *([1] * (x_latents.dim() - 1)))

            # Sample noise in latent space
            eps_latent = torch.randn_like(x_latents, device=accelerator.device)

            # Construct noisy latents z_t = alpha_t * x + sigma_t * eps
            zt = alpha_t * x_latents + sigma_t * eps_latent

            # For teacher predictions we need to call teacher_unet to get eps_pred at zt (teacher is deterministic DDIM-style)
            # Build the timesteps tensor in the scheduler-compatible format; we'll simply provide timesteps_for_unet
            # NOTE: the actual mapping of continuous t -> integer timesteps is approximate here; it's consistent across teacher & student calls.
            with torch.no_grad():
                # Teacher eps_pred at zt
                teacher_eps_pred = teacher_unet(zt, timesteps_for_unet, prompt_embeds).sample
                teacher_x_hat = compute_x_hat_from_eps_pred(zt, teacher_eps_pred, alpha_t, sigma_t)

                # Step 1: compute z_{t'} where t' = t - 0.5 / N
                t_prime_vals = (i_vals.float() - 0.5) / float(N)
                # Clamp to >= 0
                t_prime_vals = torch.clamp(t_prime_vals, min=0.0)
                alpha_t_prime, sigma_t_prime = cosine_alpha_sigma(t_prime_vals)
                alpha_t_prime = alpha_t_prime.view(batch_size, *([1] * (x_latents.dim() - 1)))
                sigma_t_prime = sigma_t_prime.view(batch_size, *([1] * (x_latents.dim() - 1)))

                z_tprime = ddim_one_step_from_x_hat(zt, teacher_x_hat, alpha_t_prime, sigma_t_prime, alpha_t, sigma_t)

                # Teacher eps_pred at z_tprime
                teacher_eps_pred_tprime = teacher_unet(z_tprime, (t_prime_vals * (pseudo_timestep_scale - 1)).long(), prompt_embeds).sample
                teacher_x_hat_tprime = compute_x_hat_from_eps_pred(z_tprime, teacher_eps_pred_tprime, alpha_t_prime, sigma_t_prime)

                # Step 2: compute z_{t''} where t'' = t - 1/N
                t_dprime_vals = (i_vals.float() - 1.0) / float(N)
                t_dprime_vals = torch.clamp(t_dprime_vals, min=0.0)
                alpha_t_dprime, sigma_t_dprime = cosine_alpha_sigma(t_dprime_vals)
                alpha_t_dprime = alpha_t_dprime.view(batch_size, *([1] * (x_latents.dim() - 1)))
                sigma_t_dprime = sigma_t_dprime.view(batch_size, *([1] * (x_latents.dim() - 1)))

                z_tdprime = ddim_one_step_from_x_hat(z_tprime, teacher_x_hat_tprime, alpha_t_dprime, sigma_t_dprime, alpha_t_prime, sigma_t_prime)

                # Now compute x_tilde per paper:
                # x_tilde = ( z_{t''} - (sigma_{t''} / sigma_t) * z_t ) / ( alpha_{t''} - (sigma_{t''} / sigma_t) * alpha_t )
                ratio = sigma_t_dprime / (sigma_t + 1e-8)
                numerator = z_tdprime - ratio * zt
                denom = alpha_t_dprime - ratio * alpha_t
                # denom might be small; add tiny eps for numerical stability
                denom_safe = denom.clone()
                denom_safe = torch.where(denom_safe.abs() < 1e-8, torch.sign(denom_safe) * 1e-8 + 1e-8, denom_safe)
                x_tilde = numerator / denom_safe

            # Now student predictions at zt
            # Student network is in train mode and being optimized
            with accelerator.accumulate(unet):
                # For student we must ensure timesteps_for_unet matches the same mapping
                noise_pred_student = unet(zt, timesteps_for_unet, prompt_embeds).sample
                x_hat_student = compute_x_hat_from_eps_pred(zt, noise_pred_student, alpha_t, sigma_t)

                # Compute loss in x-space weighted by SNR+1 (paper's SNR+1 weighting)
                weight = snr_plus_one_weight(alpha_t.view(batch_size, *([1] * (x_latents.dim() - 1))),
                                            sigma_t.view(batch_size, *([1] * (x_latents.dim() - 1))))
                # For broadcasting, ensure weight has same shape as latents
                weight = weight.to(x_tilde.dtype)

                # MSE per-element
                l2 = (x_tilde - x_hat_student).pow(2)
                # mean over latent dims, sum/mean over batch
                loss_per_sample = l2.view(batch_size, -1).mean(dim=1)
                weighted_loss_per_sample = (weight.view(batch_size, -1).mean(dim=1)) * loss_per_sample
                loss = weighted_loss_per_sample.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), config.train.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            updates_done += 1
            pbar.update(1)

        pbar.close()
        print(f"Finished distillation stage for student_steps={student_steps}. Updates done = {updates_done}")

        # After finishing training student for this N, promote student -> teacher (θ ← η)
        # We'll copy student's weights to teacher_unet state_dict (so next iteration teacher is the student)
        new_teacher_state = teacher_unet.state_dict()
        student_state = student_unet.state_dict()
        copied2 = []
        for k in student_state:
            if k in new_teacher_state and student_state[k].size() == new_teacher_state[k].size():
                new_teacher_state[k].copy_(student_state[k].to(new_teacher_state[k].device))
                copied2.append(k)
        print(f"Promoted student -> teacher by copying {len(copied2)} params")

        # Optionally halve N (but we're iterating over provided steps_list explicitly)

        # --- Save student model checkpoint for this stage ---
        stage_save_dir = os.path.join(stats_dir, f"student_steps_{student_steps}")
        os.makedirs(stage_save_dir, exist_ok=True)
        if accelerator.is_main_process:
            # Save using diffusers' save_pretrained interface; if LoRA is used, ensure proper saving of attn procs
            try:
                student_pipeline.save_pretrained(stage_save_dir)
                print(f"Saved student pipeline at {stage_save_dir}")
            except Exception as e:
                print(f"Warning: failed to save student pipeline: {e}")

        # --- Evaluation imaging (kept same as your original code style) ---
        if accelerator.is_main_process:
            eval_prompts = [
                "A cat on a chair",
                "A boy in a forest",
                "A futuristic city skyline at night",
                "A dragon flying over mountains",
                "A cozy cabin in a snowy forest",
            ]

            with torch.no_grad():
                with accelerator.autocast():
                    teacher_eval_images = [
                        teacher_pipeline(
                            prompt,
                            num_inference_steps=config.sample.num_steps,
                            guidance_scale=config.sample.guidance_scale,
                        ).images[0]
                        for prompt in eval_prompts
                    ]

                    student_eval_images = [
                        student_pipeline(
                            prompt,
                            num_inference_steps=student_steps,
                            guidance_scale=config.sample.guidance_scale,
                        ).images[0]
                        for prompt in eval_prompts
                    ]

        # Save side-by-side
        save_side_by_side(student_eval_images, teacher_eval_images, student_steps, outdir)

    print("Progressive distillation completed for all stages.")
    # Final save
    final_save_dir = os.path.join(stats_dir, "final_student")
    if accelerator.is_main_process:
        try:
            student_pipeline.save_pretrained(final_save_dir)
            print(f"Saved final student pipeline at {final_save_dir}")
        except Exception as e:
            print(f"Warning: failed to save final student pipeline: {e}")

if __name__ == "__main__":
    app.run(main)
