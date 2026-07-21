# train_dmd2_in_template.py
# Drop-in script preserving train_ppo_coco.py structure but replacing PPO core with DMD2 distillation.
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
# from ddpo_pytorch.rewards import get_reward_fn  # Not used in DMD2
from ddpo_pytorch.stat_tracking import PerPromptStatTracker
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
from ddpo_pytorch import flop_budget
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
from ddpo_pytorch.flop_budget import safe_get

def save_side_by_side(student_images, teacher_images, epoch, outdir):
    os.makedirs(outdir, exist_ok=True)
    student_dir = os.path.join(outdir, "student_only")
    combined_dir = os.path.join(outdir, "side_by_side")

    os.makedirs(student_dir, exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)

    for idx in range(min(len(student_images), len(teacher_images))):
        s_img = student_images[idx].convert("RGB")
        t_img = teacher_images[idx].convert("RGB")

        # Resize both to the same square size
        h = min(s_img.height, t_img.height)
        s_img = s_img.resize((h, h), Image.Resampling.LANCZOS)
        t_img = t_img.resize((h, h), Image.Resampling.LANCZOS)

        # Save student-only image
        student_path = os.path.join(student_dir, f"epoch{epoch}_student{idx}.png")
        s_img.save(student_path)
        print(f"Saved student image: {student_path}")

        # Create side-by-side combined image
        combined = Image.new("RGB", (t_img.width + s_img.width, h))
        combined.paste(t_img, (0, 0))
        combined.paste(s_img, (t_img.width, 0))

        combined_path = os.path.join(combined_dir, f"epoch{epoch}_sample{idx}.png")
        combined.save(combined_path)
        print(f"Saved side-by-side image: {combined_path}")

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
flags.DEFINE_string(
    "reference_config", "config/distill_clip.py",
    "Config file for scripts/train_ppo_coco.py's run - used ONLY to compute "
    "the FLOP budget this baseline should match, not to change this "
    "baseline's own hyperparameters."
)

logger = get_logger(__name__)

def kl_divergence(p, q):
    p = p.float().view(p.shape[0], -1)
    q = q.float().view(q.shape[0], -1)
    p = F.log_softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    kl = F.kl_div(p, q, reduction='batchmean', log_target=False)
    return kl

def align_teacher_student_steps(teacher_latents, student_latents, teacher_steps, student_steps):
    if isinstance(teacher_latents, list):
        teacher_latents = torch.stack(teacher_latents, dim=1)
    align_indices = torch.linspace(teacher_steps//student_steps, teacher_steps, steps=student_steps).long()
    aligned_teacher_latents = teacher_latents[:, align_indices]
    return aligned_teacher_latents, student_latents[:,1:]

# Simple bottleneck discriminator used for DMD2
class BottleneckDisc(nn.Module):
    def __init__(self, in_channels, hidden_channels=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(hidden_channels, 1)

    def forward(self, feat):
        x = self.net(feat)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def main(_):
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    config.run_name = config.run_name or unique_id
    config.run_name += f"_dmd2_{unique_id}"

    # Keep the same outdir naming style as PPO code (adapt to DMD2)
    outdir = "DMD2_distill"
    if FLAGS.prompt_source == "coco":
        outdir = outdir + "_coco_prompts"
    else:
        outdir = outdir + "_ddpo_prompts"

    stats_dir = outdir
    os.makedirs(stats_dir, exist_ok=True)
    stats_file = os.path.join(stats_dir, "training_stats.txt")

    all_losses = []
    all_dmd_losses = []
    all_gen_gan_losses = []
    all_disc_losses = []

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        total_limit=config.num_checkpoint_limit,
    )
    accelerator = Accelerator(
        log_with="wandb" if getattr(config, "log_with_wandb", True) else None,
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=max(1, config.train.gradient_accumulation_steps * max(1, getattr(config.student, "num_steps", 1))),
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("dmd2-distill", config=config.to_dict())

    logger.info(f"\n{config}")

    # ------------------------------
    # Load teacher pipeline (frozen)
    # ------------------------------
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

    # ------------------------------
    # Load student pipeline (trainable)
    # ------------------------------
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
    student_pipeline.unet.requires_grad_(not config.use_lora)

    # LoRA handling preserved exactly as PPO template
    if config.use_lora:
        lora_attn_procs = {}
        for name in student_pipeline.unet.attn_processors.keys():
            cross_attention_dim = (
                None if name.endswith("attn1.processor") else student_pipeline.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = student_pipeline.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                # --- FIX --- #
                block_id = int(name[len("up_blocks."):].split(".")[0])
                # --- END FIX --- #
                hidden_size = list(reversed(student_pipeline.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                # --- FIX --- #
                block_id = int(name[len("down_blocks."):].split(".")[0])
                # --- END FIX --- #
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

    # optimizer for generator (student UNet / LoRA params)
    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        unet.parameters(),
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # Prepare unet and optimizer with accelerator (same as PPO template)
    unet, optimizer = accelerator.prepare(unet, optimizer)

    # Prompt function (same as PPO)
    prompt_fn = getattr(ddpo_pytorch.prompts, config.prompt_fn)

    # --- Load COCO captions if selected (same as PPO) ---
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

    # ------------------------------
    # Build mu_fake (full UNet copy) and its optimizer
    # ------------------------------
    from copy import deepcopy

    # deep copy the entire UNet module (parameters, buffers, submodules)
    mu_fake = deepcopy(student_pipeline.unet)

    # Ensure mu_fake parameters are trainable (and independent)
    for p in mu_fake.parameters():
        p.requires_grad = True

    # If the student UNet was using LoRA processors, make sure mu_fake has its own copies.
    try:
        if hasattr(student_pipeline.unet, "attn_processors"):
            # clone processors dict so mu_fake doesn't reference the same processors
            mu_fake.set_attn_processor(deepcopy(student_pipeline.unet.attn_processors))
    except Exception:
        pass

    fake_lr = getattr(config.train, "fake_lr", 1e-4)
    fake_optimizer = torch.optim.AdamW(mu_fake.parameters(), lr=fake_lr, betas=(0.9, 0.99))

    # ------------------------------
    # Attach hook to capture mid-block bottleneck features of student UNet
    # ------------------------------
    mid_feat_store = {"feat": None}
    def mid_block_hook(module, inp, out):
        feat = out
        if hasattr(out, "sample"):
            feat = out.sample
        mid_feat_store["feat"] = feat

    # Try register hook on mid_block (diffusers typical)
    try:
        if hasattr(student_pipeline.unet, "mid_block"):
            student_pipeline.unet.mid_block.register_forward_hook(mid_block_hook)
        elif hasattr(student_pipeline.unet, "down_blocks"):
            # fallback: register on last down block
            student_pipeline.unet.down_blocks[-1].register_forward_hook(mid_block_hook)
        else:
            logger.warning("Unable to register UNet mid_block hook; discriminator will be disabled if hook fails.")
    except Exception as e:
        logger.warning(f"Failed to register mid_block hook: {e}")

    # Dry-forward a tiny tensor to determine mid feature channels if possible
    disc = None
    disc_optimizer = None
    if safe_get(config, "gan", "use_gan", True):
        try:
            dummy_bs = 1
            height = getattr(config.sample, "height", 512)
            width = getattr(config.sample, "width", 512)
            lat_h = max(4, height // 8)
            lat_w = max(4, width // 8)
            dummy_lat = torch.randn((dummy_bs, student_pipeline.unet.in_channels, lat_h, lat_w), device=accelerator.device, dtype=dtype)
            # run a forward to populate mid_feat_store
            with torch.no_grad():
                _ = student_pipeline.unet(dummy_lat, torch.tensor([student_pipeline.scheduler.timesteps[0]], device=accelerator.device), encoder_hidden_states=torch.zeros((1,1,student_pipeline.text_encoder.config.hidden_size), device=accelerator.device, dtype=dtype))
            mid_feat = mid_feat_store.get("feat", None)
            if mid_feat is not None:
                mid_ch = mid_feat.shape[1]
                disc = BottleneckDisc(in_channels=mid_ch, hidden_channels=safe_get(config, "gan", "disc_hidden", 128)).to(accelerator.device)
                disc_optimizer = torch.optim.AdamW(disc.parameters(), lr=getattr(config.train, "disc_lr", 1e-4), betas=(0.5, 0.9))
            else:
                logger.warning("mid-block hook did not capture features; disabling discriminator.")
                disc = None
                disc_optimizer = None
        except Exception as e:
            logger.warning(f"Failed to create discriminator: {e}")
            disc = None
            disc_optimizer = None

    # Prepare mu_fake, disc, and their optimizers with accelerator
    if mu_fake is not None and fake_optimizer is not None:
        mu_fake, fake_optimizer = accelerator.prepare(mu_fake, fake_optimizer)
    if disc is not None and disc_optimizer is not None:
        disc, disc_optimizer = accelerator.prepare(disc, disc_optimizer)

    # Training hyperparams
    bs = config.sample.batch_size
    fake_updates = getattr(config.train, "fake_updates_per_gen", 5)
    gan_weight = safe_get(config, "gan", "weight", 1.0)
    use_gan = safe_get(config, "gan", "use_gan", False) and (disc is not None)

    # ------------------------------------------------------------
    # FLOP-matching: DMD2 does far more compute per epoch than plain PPO
    # (an extra full auxiliary UNet `mu_fake` trained `fake_updates` times,
    # plus a discriminator, plus a second/third generator-side UNet call),
    # so matching *epoch counts* across baselines would NOT match compute.
    # Calibrate every distinct call site this loop actually uses and derive
    # num_epochs from the same total-FLOP target as train_ppo_coco.py.
    # ------------------------------------------------------------
    cross_attn_dim = teacher_pipeline.unet.config.cross_attention_dim
    teacher_flops = flop_budget.calibrate_unet_call(
        teacher_pipeline.unet, bs, cross_attn_dim, device=accelerator.device, dtype=dtype, backward=False,
    )
    student_flops = flop_budget.calibrate_unet_call(
        accelerator.unwrap_model(student_pipeline.unet), bs, cross_attn_dim,
        device=accelerator.device, dtype=dtype, backward=True,
    )
    mu_fake_flops = flop_budget.calibrate_unet_call(
        accelerator.unwrap_model(mu_fake), bs, cross_attn_dim, device=accelerator.device, dtype=dtype, backward=True,
    )
    disc_flops = 0.0
    if use_gan:
        mid_ch = disc.net[0].in_channels if hasattr(disc, "net") else None
        try:
            unwrapped_disc = accelerator.unwrap_model(disc)
            in_ch = unwrapped_disc.net[0].in_channels
            def disc_inputs():
                return (torch.randn(bs, in_ch, 8, 8, device=accelerator.device, dtype=dtype, requires_grad=True),)
            disc_flops = flop_budget.calibrate_call(unwrapped_disc, disc_inputs, backward=True)
        except Exception as e:
            logger.warning(f"[flop-match] could not calibrate discriminator, treating its cost as 0: {e}")

    ref_config = flop_budget.load_reference_config(FLAGS.reference_config)
    target_flops = flop_budget.reference_budget_flops(ref_config, teacher_flops, student_flops)

    # Per-epoch call count for THIS script:
    #   - 1 teacher rollout (sample.num_steps, no_grad) + 1 student rollout
    #     (student.num_steps, no_grad)                      -> teacher-cost class
    #   - 1 extra single-step teacher call (no_grad)          -> teacher-cost class
    #   - `fake_updates` mu_fake forward+backward calls
    #   - (if GAN) 2 extra no_grad student forward calls for disc features,
    #     1 disc forward+backward for the disc update, 1 more no_grad student
    #     forward + 1 disc forward (no grad into disc, but counted at its
    #     forward+backward calibration for a conservative/simple estimate)
    #     for the generator's adversarial term
    #   - 1 student forward+backward for the generator update
    rollout_calls = config.sample.num_steps + config.student.num_steps + 1
    gan_student_fwd_calls = 3 if use_gan else 0  # no_grad student calls used only to read mid features
    flops_per_epoch = (
        rollout_calls * teacher_flops
        + gan_student_fwd_calls * teacher_flops
        + fake_updates * mu_fake_flops
        + (2 * disc_flops if use_gan else 0.0)
        + student_flops
    )
    matched_num_epochs = flop_budget.units_needed(target_flops, flops_per_epoch)
    flop_budget.report(accelerator.print, target_flops, flops_per_epoch, "epoch", matched_num_epochs)
    num_epochs = matched_num_epochs

    logger.info("Beginning DMD2 training loop")

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch}: Sampling and DMD2 Training")

        # --- Pick prompts (same as PPO) ---
        if FLAGS.prompt_source == "coco" and coco_captions is not None:
            if len(coco_captions) >= config.sample.batch_size:
                prompts = random.sample(coco_captions, k=config.sample.batch_size)
            else:
                prompts = [random.choice(coco_captions) for _ in range(config.sample.batch_size)]
            prompt_pairs = [(p, None) for p in prompts]
            prompt_metadata = [None] * len(prompts)
        else:
            prompt_pairs = [prompt_fn(**config.prompt_fn_kwargs) for _ in range(config.sample.batch_size)]
            prompts = [p[0] for p in prompt_pairs]
            prompt_metadata = [p[1] for p in prompt_pairs]

        # --- Encode prompts (same as PPO) ---
        prompt_ids = student_pipeline.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=student_pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
        prompt_embeds = student_pipeline.text_encoder(prompt_ids)[0]

        # --- Produce teacher and student latents/images (same pipeline interface as PPO) ---
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

        student_pipeline.unet.eval()
        with accelerator.autocast():
            # Note: We must run student sampling *without* grads
            # to get the *actual* student distribution for fake_mu and disc training.
            with torch.no_grad():
                student_out = pipeline_with_logprob(
                    student_pipeline,
                    prompt_embeds=prompt_embeds,
                    num_inference_steps=config.student.num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    eta=config.sample.eta,
                    output_type="pt",
                    return_all_latents=True,
                )
                student_images, _, student_latents_all, student_log_probs_all = student_out

        # Stack latents: shape [B, steps, C, H, W]
        latents = torch.stack(student_latents_all, dim=1)
        # We'll use the last latent (most denoised) as student's generated clean latent for DMD operations
        student_clean_latent = latents[:, -1].detach()  # detach for mu_fake updates

        # Choose a timestep t from teacher scheduler's timesteps
        teacher_sched = teacher_pipeline.scheduler
        t_choices = list(teacher_sched.timesteps)
        t_chosen = int(random.choice(t_choices))
        timesteps = torch.tensor([t_chosen] * student_clean_latent.shape[0], device=accelerator.device)

        # Sample noise
        noise = torch.randn_like(student_clean_latent)

        # Create noisy latents using teacher scheduler
        try:
            noisy_latents = teacher_sched.add_noise(student_clean_latent, noise, timesteps)
        except Exception:
            # Fallback if scheduler doesn't have add_noise
            alphas_cumprod = teacher_sched.alphas_cumprod
            alpha_val = alphas_cumprod[t_chosen].to(accelerator.device, dtype=dtype)
            alpha_sqrt = alpha_val.sqrt()
            sigma_sqrt = (1.0 - alpha_val).sqrt()
            noisy_latents = alpha_sqrt * student_clean_latent + sigma_sqrt * noise


        # Compute teacher score (denoiser output) at noisy latents (no grad)
        with torch.no_grad():
            teacher_pred = teacher_pipeline.unet(noisy_latents, timesteps, encoder_hidden_states=prompt_embeds).sample

        # --- Update mu_fake (fake score model) multiple times (TTUR) ---
        mu_fake.train()
        for k in range(fake_updates):
            with accelerator.accumulate(mu_fake):
                pred_noise = mu_fake(noisy_latents, timesteps, encoder_hidden_states=prompt_embeds).sample
                loss_mu = F.mse_loss(pred_noise, noise)
                fake_optimizer.zero_grad()
                accelerator.backward(loss_mu)
                fake_optimizer.step()

        # --- Update discriminator (on bottleneck features) if enabled ---
        disc_loss_val = 0.0
        if use_gan:
            disc.train()
            with accelerator.accumulate(disc):
                # try to obtain bottleneck features for real images: use teacher images as "real"
                try:
                    with torch.no_grad():
                        real_enc_dist = student_pipeline.vae.encode(teacher_images)
                        real_enc = real_enc_dist.latent_dist.sample()
                except Exception:
                    # fallback by converting and encoding
                    real_t = []
                    for im in teacher_images:
                        arr = np.array(im).astype(np.float32) / 255.0
                        t = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(accelerator.device, dtype=dtype)
                        real_t.append(t)
                    real_t = torch.cat(real_t, dim=0)
                    with torch.no_grad():
                        real_enc = student_pipeline.vae.encode(real_t).latent_dist.sample()
                
                real_enc = real_enc * student_pipeline.vae.config.scaling_factor

                # run forward through student unet to capture mid features
                mid_feat_store["feat"] = None
                with torch.no_grad(): # Don't need grads for disc input
                    _ = student_pipeline.unet(real_enc, timesteps, encoder_hidden_states=prompt_embeds)
                real_feat = mid_feat_store.get("feat", None)

                # fake features from student_clean_latent (generated latents)
                mid_feat_store["feat"] = None
                with torch.no_grad(): # Don't need grads for disc input
                    _ = student_pipeline.unet(student_clean_latent.detach(), timesteps, encoder_hidden_states=prompt_embeds)
                fake_feat = mid_feat_store.get("feat", None)

                if (real_feat is not None) and (fake_feat is not None):
                    real_logit = disc(real_feat.detach())
                    fake_logit = disc(fake_feat.detach())
                    d_loss = F.softplus(-real_logit).mean() + F.softplus(fake_logit).mean()
                    disc_optimizer.zero_grad()
                    accelerator.backward(d_loss)
                    disc_optimizer.step()
                    disc_loss_val = d_loss.item()
                else:
                    disc_loss_val = 0.0
                    if real_feat is None: logger.warning("Disc update skipped: real_feat is None.")
                    if fake_feat is None: logger.warning("Disc update skipped: fake_feat is None.")


        # --- Generator update (student) ---
        # Set student model to train
        student_pipeline.unet.train()
        if config.use_lora:
            unet.train()
            
        optimizer.zero_grad()
        with accelerator.accumulate(unet):
            # random latents z (start point for 1-step differentiable generation)
            z = torch.randn_like(student_clean_latent, device=accelerator.device)
            
            # student predicts noise / denoised output from z at t
            # This is the 1-step generation pass where grads are kept
            pred_eps = unet(z, timesteps, encoder_hidden_states=prompt_embeds).sample

            # reconstruct x0_est from predicted eps
            pred_x0 = None
            try:
                alphas_cumprod = teacher_sched.alphas_cumprod
                alpha_val = alphas_cumprod[t_chosen].to(accelerator.device, dtype=z.dtype)
                alpha_tensor = alpha_val.sqrt()
                sigma_tensor = (1.0 - alpha_val).sqrt()
                # reconstruct x0
                pred_x0 = (z - sigma_tensor * pred_eps) / (alpha_tensor + 1e-12)
            except Exception as e:
                logger.warning(f"Failed to use scheduler alphas for x0 recon: {e}. Falling back.")
                pred_x0 = z - pred_eps

            # create noisy_for_gen via teacher scheduler
            try:
                noisy_for_gen = teacher_sched.add_noise(pred_x0, noise, timesteps)
            except Exception:
                # Fallback
                alphas_cumprod = teacher_sched.alphas_cumprod
                alpha_val = alphas_cumprod[t_chosen].to(accelerator.device, dtype=dtype)
                alpha_sqrt = alpha_val.sqrt()
                sigma_sqrt = (1.0 - alpha_val).sqrt()
                noisy_for_gen = alpha_sqrt * pred_x0 + sigma_sqrt * noise


            # mu_fake prediction (train mode for gradient path). We need
            # gradients to flow BACK THROUGH mu_fake's activations into the
            # generator's parameters (that's how the DMD loss trains the
            # generator), but mu_fake's OWN parameters should not receive or
            # accumulate gradients here - only `fake_optimizer` is allowed to
            # update them, in the earlier "Update mu_fake" block above. Toggle
            # requires_grad off/back-on around this single call instead of
            # `torch.no_grad()` (which would also block the gradient the
            # generator needs).
            mu_fake.train()
            for p in mu_fake.parameters():
                p.requires_grad_(False)
            mu_pred_for_gen = mu_fake(noisy_for_gen, timesteps, encoder_hidden_states=prompt_embeds).sample
            for p in mu_fake.parameters():
                p.requires_grad_(True)

            # Distribution matching loss (MSE between teacher_pred and mu_pred_for_gen)
            dmd_loss = F.mse_loss(mu_pred_for_gen, teacher_pred.detach())

            # GAN generator loss (if enabled)
            gen_gan_loss_val = 0.0
            if use_gan:
                disc.eval() # Disc is frozen during gen update
                mid_feat_store["feat"] = None
                # Run student unet *again* on the differentiable pred_x0
                _ = student_pipeline.unet(pred_x0, timesteps, encoder_hidden_states=prompt_embeds)
                mid_feat_for_gen = mid_feat_store.get("feat", None)
                
                if mid_feat_for_gen is not None:
                    fake_logit_for_gen = disc(mid_feat_for_gen)
                    gen_gan_loss = F.softplus(-fake_logit_for_gen).mean()
                    gen_gan_loss_val = gen_gan_loss.item()
                else:
                    logger.warning("Gen GAN loss skipped: mid_feat_for_gen is None.")
                    gen_gan_loss = torch.tensor(0.0, device=accelerator.device)
            else:
                gen_gan_loss = torch.tensor(0.0, device=accelerator.device)

            gen_loss = dmd_loss + gan_weight * gen_gan_loss

            accelerator.backward(gen_loss)
            # clip grads same as PPO template style
            if accelerator.sync_gradients:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), config.train.max_grad_norm)
            optimizer.step()

        # save stats
        all_losses.append(gen_loss.item() if isinstance(gen_loss, torch.Tensor) else float(gen_loss))
        all_dmd_losses.append(dmd_loss.item() if isinstance(dmd_loss, torch.Tensor) else float(dmd_loss))
        all_gen_gan_losses.append(gen_gan_loss_val)
        all_disc_losses.append(disc_loss_val)

        with open(stats_file, "a") as f:
            f.write(f"Epoch {epoch}: gen_loss={all_losses[-1]:.6f}, dmd_loss={all_dmd_losses[-1]:.6f}, gen_gan={all_gen_gan_losses[-1]:.6f}, disc_loss={all_disc_losses[-1]:.6f}\n")

        print(f"Epoch {epoch}: gen_loss={all_losses[-1]:.6f}, dmd_loss={all_dmd_losses[-1]:.6f}, gen_gan={all_gen_gan_losses[-1]:.6f}, disc_loss={all_disc_losses[-1]:.6f}")

        # Monitoring images & saving — preserved exactly like PPO template
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
                            num_inference_steps=getattr(config.student, "num_steps", 1),
                            guidance_scale=config.sample.guidance_scale,
                        ).images[0]
                        for prompt in eval_prompts
                    ]

            save_side_by_side(student_eval_images, teacher_eval_images, epoch, outdir)

            # Save student
            try:
                student_pipeline.save_pretrained(stats_dir + "/student_model/")
            except Exception as e:
                logger.warning(f"Failed to save student pipeline at epoch {epoch}: {e}")

    # After training: plot losses (same plotting style as PPO)
    if accelerator.is_main_process:
        epochs = range(len(all_losses))
        plt.figure()
        plt.plot(epochs, all_losses, label="Gen loss")
        plt.plot(epochs, all_dmd_losses, label="DMD loss")
        plt.plot(epochs, all_gen_gan_losses, label="Gen GAN loss")
        plt.plot(epochs, all_disc_losses, label="Disc loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.savefig(os.path.join(stats_dir, "loss_curve.png"))
        plt.close()

        # Save training stats json
        try:
            import json as _json
            with open(os.path.join(stats_dir, "training_losses.json"), "w") as f:
                _json.dump({"gen": all_losses, "dmd": all_dmd_losses, "gen_gan": all_gen_gan_losses, "disc": all_disc_losses}, f)
        except Exception:
            pass

if __name__ == "__main__":
    app.run(main)