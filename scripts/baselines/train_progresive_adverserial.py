# train_progressive_adversarial_coco.py
# Progressive Adversarial Distillation (debugged)
# Keeps your model loading, LoRA injection and eval image saving exactly as before.
# Uses progressive_adverserial.pdf adversarial distillation loss.

from collections import defaultdict
import contextlib
import os
import datetime
import time
from absl import app, flags
from ml_collections import config_flags
import json
import random
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import numpy as np
import ddpo_pytorch.prompts
# The ddpo_pytorch utilities referenced in your earlier script:
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
import torch
import wandb
from functools import partial
import tqdm
from PIL import Image
import torch.nn.functional as F
import math
import torch.nn as nn

# -------------------------
# Flags and config
# -------------------------
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/distill_clip.py", "Training configuration.")

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

import tqdm as tqdm_lib
tqdm = partial(tqdm_lib.tqdm, dynamic_ncols=True)
logger = get_logger(__name__)

# -------------------------
# Small helper utilities
# -------------------------
def save_side_by_side(student_images, teacher_images, epoch, outdir):
    os.makedirs(outdir, exist_ok=True)

    # Separate directories for clarity
    student_dir = os.path.join(outdir, "student_only")
    teacher_dir = os.path.join(outdir, "teacher_only")
    combined_dir = os.path.join(outdir, "side_by_side")

    os.makedirs(student_dir, exist_ok=True)
    os.makedirs(teacher_dir, exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)

    for idx in range(min(len(student_images), len(teacher_images))):
        s_img = student_images[idx].convert("RGB")
        t_img = teacher_images[idx].convert("RGB")

        # Resize both to the same square size (keep consistent size)
        h = min(s_img.height, t_img.height)
        s_img = s_img.resize((h, h), Image.Resampling.LANCZOS)
        t_img = t_img.resize((h, h), Image.Resampling.LANCZOS)

        # --- Save teacher-only image ---
        teacher_path = os.path.join(teacher_dir, f"epoch{epoch}_teacher{idx}.png")
        t_img.save(teacher_path)
        print(f"Saved teacher image: {teacher_path}")

        # --- Save student-only image ---
        student_path = os.path.join(student_dir, f"epoch{epoch}_student{idx}.png")
        s_img.save(student_path)
        print(f"Saved student image: {student_path}")

        # --- Create side-by-side combined image ---
        combined = Image.new("RGB", (t_img.width + s_img.width, h))
        combined.paste(t_img, (0, 0))
        combined.paste(s_img, (t_img.width, 0))

        combined_path = os.path.join(combined_dir, f"epoch{epoch}_sample{idx}.png")
        combined.save(combined_path)
        print(f"Saved side-by-side image: {combined_path}")

def cosine_alpha_sigma(t):
    """t in [0,1] scalar or tensor -> (alpha, sigma)."""
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
    DDIM update from t->s using predicted x_hat:
      z_s = alpha_s * x_hat + (sigma_s / sigma_t) * ( z_t - alpha_t * x_hat )
    Vectorized; all inputs broadcastable to z shape.
    """
    return alpha_s * x_hat + (sigma_s / sigma_t) * (zt - alpha_t * x_hat)

def compute_x_hat_from_eps_pred(zt, eps_pred, alpha_t, sigma_t):
    denom = alpha_t
    if isinstance(denom, torch.Tensor):
        denom = torch.where(denom == 0.0, torch.full_like(denom, 1e-6), denom)
    else:
        if denom == 0.0:
            denom = 1e-6
    return (zt - sigma_t * eps_pred) / denom

def snr_plus_one_weight(alpha_t, sigma_t):
    eps = 1e-6
    return 1.0 + (alpha_t * alpha_t) / (sigma_t * sigma_t + eps)

def kl_divergence(p, q):
    p = p.float().view(p.shape[0], -1)
    q = q.float().view(q.shape[0], -1)
    p = F.log_softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    kl = F.kl_div(p, q, reduction='batchmean', log_target=False)
    return kl

# -------------------------
# Discriminator (lightweight)
# -------------------------
class SimpleDiscriminator(nn.Module):
    """
    Small conditional discriminator that pools a 4-channel latent input and conditions on prompt embeddings.
    This is intentionally simple and stable; you can replace with a deeper convnet if desired.
    """
    def __init__(self, in_channels=4, cond_dim=768, hidden=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, 1, 1),
            nn.GroupNorm(32, hidden), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, 2, 1),
            nn.GroupNorm(32, hidden), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, 2, 1),
            nn.GroupNorm(32, hidden), nn.SiLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden + cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x_latent, cond_embeds):
        # x_latent: (B,C,H,W), cond_embeds: (B,seq,L) or (B,L)
        feat = self.conv(x_latent).mean(dim=[2,3])  # global avg pool -> (B,hidden)
        if cond_embeds.dim() == 3:
            cond = cond_embeds.mean(dim=1)  # (B,L)
        else:
            cond = cond_embeds
        h = torch.cat([feat, cond], dim=1)
        return self.fc(h).squeeze(1)

# -------------------------
# Main training routine
# -------------------------
def main(_):
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    config.run_name = config.run_name or unique_id
    config.run_name += f"_pad_{unique_id}"

    outdir = "ProgressiveAdversarialDistill"
    if FLAGS.prompt_source == "coco":
        outdir = outdir + "_coco_prompts"
    else:
        outdir = outdir + "_ddpo_prompts"

    stats_dir = outdir
    os.makedirs(stats_dir, exist_ok=True)

    # Accelerator / logging
    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        total_limit=getattr(config, "num_checkpoint_limit", None),
    )
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=getattr(config.train, "gradient_accumulation_steps", 1),
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("prog_adv_distill", config=config.to_dict())

    logger.info(f"Config:\n{config}")

    # -------------------------
    # Load teacher pipeline
    # -------------------------
    teacher_pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained.model, revision=getattr(config.pretrained, "revision", None)
    )
    # ensure DDIM scheduler (consistent with progressive distillation math)
    teacher_pipeline.scheduler = DDIMScheduler.from_config(teacher_pipeline.scheduler.config)
    teacher_pipeline.safety_checker = None
    teacher_pipeline.set_progress_bar_config(disable=False)

    # move key modules to device
    teacher_pipeline.vae.to(accelerator.device)
    teacher_pipeline.text_encoder.to(accelerator.device)
    teacher_pipeline.unet.to(accelerator.device)
    teacher_pipeline.text_encoder.requires_grad_(False)
    teacher_pipeline.vae.requires_grad_(False)
    teacher_pipeline.unet.requires_grad_(False)
    # Save teacher snapshot (optional)
    try:
        teacher_pipeline.save_pretrained(getattr(config, "teacher_output_dir", "teacher_saved"))
    except Exception:
        pass

    # -------------------------
    # Load student pipeline
    # -------------------------
    student_pipeline = StableDiffusionPipeline.from_pretrained(
        config.student.model, revision=getattr(config.student, "revision", None)
    )
    student_pipeline.scheduler = DDIMScheduler.from_config(student_pipeline.scheduler.config)
    student_pipeline.safety_checker = None
    student_pipeline.set_progress_bar_config(disable=False)

    student_pipeline.vae.to(accelerator.device)
    student_pipeline.text_encoder.to(accelerator.device)
    student_pipeline.unet.to(accelerator.device)

    student_pipeline.vae.requires_grad_(False)
    student_pipeline.text_encoder.requires_grad_(False)
    student_pipeline.unet.requires_grad_(not getattr(config, "use_lora", False))

    # -------------------------
    # LoRA setup (kept as your original)
    # -------------------------
    if getattr(config, "use_lora", False):
        lora_attn_procs = {}
        for name in student_pipeline.unet.attn_processors.keys():
            cross_attention_dim = (
                None if name.endswith("attn1.processor") else student_pipeline.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = student_pipeline.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks."):].split(".")[0]) if "." in name[len("up_blocks."): ] else int(name[len("up_blocks."):])
                hidden_size = list(reversed(student_pipeline.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks."):].split(".")[0]) if "." in name[len("down_blocks."): ] else int(name[len("down_blocks."):])
                hidden_size = student_pipeline.unet.config.block_out_channels[block_id]
            else:
                hidden_size = student_pipeline.unet.config.block_out_channels[-1]
            lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
        student_pipeline.unet.set_attn_processor(lora_attn_procs)

        class _Wrapper(AttnProcsLayers):
            def forward(self, *args, **kwargs):
                return student_pipeline.unet(*args, **kwargs)

        unet = _Wrapper(student_pipeline.unet.attn_processors)
    else:
        unet = student_pipeline.unet

    # dtype placement for mixed precision
    dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16

    student_pipeline.vae.to(accelerator.device, dtype=dtype)
    student_pipeline.text_encoder.to(accelerator.device, dtype=dtype)
    if getattr(config, "use_lora", False):
        student_pipeline.unet.to(accelerator.device, dtype=dtype)

    # Optimizers
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # Discriminator + its optimizer
    disc = SimpleDiscriminator(
        in_channels=getattr(config, "latent_channels", 4),
        cond_dim=getattr(student_pipeline.text_encoder.config, "hidden_size", 768),
        hidden=getattr(config, "disc_hidden", 128),
    ).to(accelerator.device)

    opt_d = torch.optim.AdamW(disc.parameters(), lr=getattr(config.distill, "disc_lr", 1e-6), betas=(0.0, 0.99))

    # Prepare with accelerator
    unet, optimizer, disc, opt_d = accelerator.prepare(unet, optimizer, disc, opt_d)

    # Prompt function
    prompt_fn = getattr(ddpo_pytorch.prompts, config.prompt_fn)

    # Load COCO captions if required
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

    # Distillation hyperparams
    steps_list = getattr(config.distill, "steps_list", [50, 25, 12, 5])
    updates_per_stage = getattr(config.train, "distill_updates_per_stage", 10)
    batch_size = config.sample.batch_size
    guidance_scale = getattr(config.sample, "guidance_scale", 1.0)
    teacher_steps = getattr(config.sample, "num_steps", 50)

    # Sometimes tokenizers/text encoder need correct device/dtype
    # We'll call text_encoder with input_ids on accelerator.device

    # MAIN progressive-adversarial loop
    teacher_unet = teacher_pipeline.unet
    student_unet = student_pipeline.unet

    # Put teacher to eval and freeze grads
    teacher_unet.eval()
    for stage_idx, student_steps in enumerate(steps_list):
        print(f"\n==== Distillation stage {stage_idx+1}/{len(steps_list)}: student_steps = {student_steps} ====\n")
        # copy teacher -> student parameters (init student from teacher each stage)
        teacher_state = {k: v.cpu() for k, v in teacher_unet.state_dict().items()}
        student_state = student_unet.state_dict()
        copied = 0
        for k in teacher_state:
            if k in student_state and teacher_state[k].shape == student_state[k].shape:
                # copy into student (device will be handled by to(device) calls later)
                student_state[k].copy_(teacher_state[k].to(student_state[k].device))
                copied += 1
        print(f"Copied {copied} matching parameters from teacher -> student")

        # Train student_unet (unet wrapper trains attn procs / LoRA if used)
        student_unet.train()
        # re-init discriminator at stage start (optional per paper)
        def reinit(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
        disc.apply(reinit)

        bce_loss = nn.BCEWithLogitsLoss()

        pbar = tqdm(range(updates_per_stage), desc=f"Stage {stage_idx+1} updates", disable=not accelerator.is_main_process)
        updates_done = 0
        while updates_done < updates_per_stage:
            # ---- sample prompts ----
            if FLAGS.prompt_source == "coco" and coco_captions is not None:
                if len(coco_captions) >= batch_size:
                    prompts = random.sample(coco_captions, k=batch_size)
                else:
                    prompts = [random.choice(coco_captions) for _ in range(batch_size)]
            else:
                prompt_pairs = [prompt_fn(**getattr(config, "prompt_fn_kwargs", {})) for _ in range(batch_size)]
                prompts = [p[0] for p in prompt_pairs]

            # encode prompts
            prompt_ids = student_pipeline.tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=student_pipeline.tokenizer.model_max_length,
            ).input_ids.to(accelerator.device)
            prompt_embeds = student_pipeline.text_encoder(prompt_ids)[0]

            # ---- teacher forward: obtain latents and compute x_tilde (two-step target) ----
            with torch.no_grad():
                teacher_out = pipeline_with_logprob(
                    teacher_pipeline,
                    prompt_embeds=prompt_embeds,
                    num_inference_steps=teacher_steps,
                    guidance_scale=guidance_scale,
                    eta=getattr(config.sample, "eta", 0.0),
                    output_type="pt",
                    return_all_latents=True,
                )
                # pipeline_with_logprob returns tuple-like: images, ..., latents_list, ...
                # We expect teacher_out[2] is list-of-latents across timesteps
                teacher_latents_all = teacher_out[2]
                # convert to tensor shaped (B, T, C, H, W) if list
                if isinstance(teacher_latents_all, list):
                    # each element is (B,C,H,W)
                    teacher_latents_tensor = torch.stack(teacher_latents_all, dim=1)
                else:
                    teacher_latents_tensor = teacher_latents_all

                # get "clean" latent x (final) as last element if available
                x_latents = teacher_latents_tensor[:, -1].to(accelerator.device)

            # prepare i's per sample
            N = int(student_steps)
            i_vals = torch.randint(low=1, high=N + 1, size=(batch_size,), device=accelerator.device)
            t_vals = i_vals.float() / float(N)  # in (1/N .. 1)
            # convert t_vals into shaped alpha/sigma for broadcast
            alpha_t, sigma_t = cosine_alpha_sigma(t_vals)
            # reshape to (B,1,1,1) to broadcast to latents' (B,C,H,W)
            alpha_t_b = alpha_t.view(batch_size, *([1] * (x_latents.dim() - 1)))
            sigma_t_b = sigma_t.view(batch_size, *([1] * (x_latents.dim() - 1)))

            # sample noise and create z_t
            eps_latent = torch.randn_like(x_latents, device=accelerator.device)
            zt = alpha_t_b * x_latents + sigma_t_b * eps_latent

            # teacher predicts eps at zt -> compute x_hat_t
            with torch.no_grad():
                # map continuous t to pseudo-integer timesteps for UNet call (heuristic)
                pseudo_t = (t_vals * 999).long()
                teacher_eps_t = teacher_unet(zt, pseudo_t, prompt_embeds).sample
                teacher_x_hat_t = compute_x_hat_from_eps_pred(zt, teacher_eps_t, alpha_t_b, sigma_t_b)

                # compute z_{t'} where t' = t - 0.5/N
                t_prime_vals = torch.clamp((i_vals.float() - 0.5) / float(N), min=0.0)
                a_p, s_p = cosine_alpha_sigma(t_prime_vals)
                a_p_b = a_p.view(batch_size, *([1] * (x_latents.dim() - 1)))
                s_p_b = s_p.view(batch_size, *([1] * (x_latents.dim() - 1)))
                z_tprime = ddim_one_step_from_x_hat(zt, teacher_x_hat_t, a_p_b, s_p_b, alpha_t_b, sigma_t_b)

                teacher_eps_tprime = teacher_unet(z_tprime, (t_prime_vals * 999).long(), prompt_embeds).sample
                teacher_x_hat_tprime = compute_x_hat_from_eps_pred(z_tprime, teacher_eps_tprime, a_p_b, s_p_b)

                # compute z_{t''} where t'' = t - 1/N
                t_dprime_vals = torch.clamp((i_vals.float() - 1.0) / float(N), min=0.0)
                a_pp, s_pp = cosine_alpha_sigma(t_dprime_vals)
                a_pp_b = a_pp.view(batch_size, *([1] * (x_latents.dim() - 1)))
                s_pp_b = s_pp.view(batch_size, *([1] * (x_latents.dim() - 1)))
                z_tdprime = ddim_one_step_from_x_hat(z_tprime, teacher_x_hat_tprime, a_pp_b, s_pp_b, a_p_b, s_p_b)

                # compute x_tilde as paper eq.43
                ratio = s_pp_b / (sigma_t_b + 1e-8)
                numerator = z_tdprime - ratio * zt
                denom = a_pp_b - ratio * alpha_t_b
                denom_safe = torch.where(denom.abs() < 1e-8, torch.sign(denom) * 1e-8 + 1e-8, denom)
                x_tilde = numerator / denom_safe  # target student should produce this x_hat

            # ---- Student forward (predict eps, compute x_hat_student) ----
            # We run the unet (which will be the LoRA wrapper if used)
            noise_pred_student = unet(zt, (t_vals * 999).long(), prompt_embeds).sample
            x_hat_student = compute_x_hat_from_eps_pred(zt, noise_pred_student, alpha_t_b, sigma_t_b)

            # --- Adversarial training ---
            # Train discriminator to distinguish teacher (x_tilde) vs student x_hat_student
            disc.train()
            real_labels = torch.ones(batch_size, device=accelerator.device)
            fake_labels = torch.zeros(batch_size, device=accelerator.device)

            # discriminator on real (teacher) example
            real_logits = disc(x_tilde.detach(), prompt_embeds.detach())
            fake_logits = disc(x_hat_student.detach(), prompt_embeds.detach())
            loss_d = 0.5 * (bce_loss(real_logits, real_labels) + bce_loss(fake_logits, fake_labels))

            accelerator.backward(loss_d)
            opt_d.step()
            opt_d.zero_grad()

            # generator (student) step: fool discriminator
            gen_logits = disc(x_hat_student, prompt_embeds)
            loss_g_adv = bce_loss(gen_logits, real_labels)

            # Optionally combine with reconstruction loss (paper sometimes uses hybrid). We'll use adversarial-only per PAD recipe.
            # If you want a hybrid: loss = lambda_adv * loss_g_adv + lambda_mse * mse(x_hat_student, x_tilde)

            accelerator.backward(loss_g_adv)
            if accelerator.sync_gradients:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), getattr(config.train, "max_grad_norm", 1.0))
            optimizer.step()
            optimizer.zero_grad()

            updates_done += 1
            pbar.update(1)

        pbar.close()
        print(f"Finished distillation stage for student_steps={student_steps}. Updates done = {updates_done}")

        # Promote student -> teacher (copy student weights into teacher)
        new_teacher_state = teacher_unet.state_dict()
        student_state = student_unet.state_dict()
        copied2 = 0
        for k in student_state:
            if k in new_teacher_state and student_state[k].size() == new_teacher_state[k].size():
                new_teacher_state[k].copy_(student_state[k].to(new_teacher_state[k].device))
                copied2 += 1
        print(f"Promoted student -> teacher by copying {copied2} params")

        # Save student pipeline checkpoint for this stage
        stage_save_dir = os.path.join(stats_dir, f"student_steps_{student_steps}")
        os.makedirs(stage_save_dir, exist_ok=True)
        if accelerator.is_main_process:
            try:
                student_pipeline.save_pretrained(stage_save_dir)
                print(f"Saved student pipeline at {stage_save_dir}")
            except Exception as e:
                print(f"Warning: failed to save student pipeline: {e}")

        # Evaluation (kept unchanged style)
        if accelerator.is_main_process:
            eval_prompts = [
                    "A crystal-clear glass bowl overflowing with ripe, vibrant oranges on a rustic wooden table, sunlight streaming through a nearby window, warm golden reflections and soft shadows",
                    "A fluffy tabby cat caught mid-step, looking directly at the camera with curious eyes, sunlight highlighting its fur, cozy home interior in soft focus behind it",
                    "A sprawling futuristic city skyline at night, glowing neon lights reflecting off glass skyscrapers, flying cars streaking through the sky, a misty cyberpunk atmosphere in vivid blues and pinks",
                    "A U.S. Marine in desert camouflage standing under a setting sun, gazing at his smartphone with a thoughtful expression, soft golden light and dust in the air",
                    "A warm, glowing log cabin nestled in a snowy pine forest at twilight, smoke rising gently from the chimney, soft snowflakes falling under a purple and orange winter sky",
                    "A sleek motorcycle parked beside a rain puddle reflecting a nearby vintage van, wet asphalt glistening under streetlights, dramatic evening sky with lingering clouds",
                    "A colorful plain under a bright sky, filled with wildflowers of red, yellow, and purple, rolling green hills stretching to the horizon, soft sunlight and a gentle breeze – یک دشت کالرفول"
                    ]
            with torch.no_grad():
                teacher_eval_images = [
                    teacher_pipeline(
                        prompt,
                        num_inference_steps=teacher_steps,
                        guidance_scale=guidance_scale,
                    ).images[0]
                    for prompt in eval_prompts
                ]
                student_eval_images = [
                    student_pipeline(
                        prompt,
                        num_inference_steps=student_steps,
                        guidance_scale=guidance_scale,
                    ).images[0]
                    for prompt in eval_prompts
                ]
            save_side_by_side(student_eval_images, teacher_eval_images, student_steps, outdir)

    print("Progressive adversarial distillation completed for all stages.")

    final_save_dir = os.path.join(stats_dir, "final_student")
    if accelerator.is_main_process:
        try:
            student_pipeline.save_pretrained(final_save_dir)
            print(f"Saved final student pipeline at {final_save_dir}")
        except Exception as e:
            print(f"Warning: failed to save final student pipeline: {e}")

if __name__ == "__main__":
    app.run(main)
