# train_dmd2_coco_full.py
# DMD2 distillation training re-implemented in the exact structure of train_ppo_coco.py
# - Uses Accelerate (no ProjectConfiguration)
# - Uses teacher.scheduler.add_noise when available (diffusers 0.17.1)
# - mu_fake := UNet2DConditionModel with same config as student UNet
# - Discriminator attached to UNet bottleneck via forward hook
# - Optional FID/CLIP (disabled by default)
#
# Usage: same CLI as train_ppo_coco.py (same config pattern)
# Ensure your config file contains sensible fields; missing optional fields are given safe defaults.

import os
import sys
import time
import json
import random
import datetime
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.logging import get_logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import tqdm
from functools import partial

# Keep your project imports if train_ppo_coco.py uses them; adapt names if different
import ddpo_pytorch.prompts
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/distill_clip.py", "Training configuration.")


# ---------------------------
# Small helper utilities
# ---------------------------
def save_side_by_side(student_images, teacher_images, epoch, outdir, max_samples=4):
    os.makedirs(outdir, exist_ok=True)
    n = min(len(student_images), len(teacher_images), max_samples)
    for i in range(n):
        s = student_images[i].convert("RGB")
        t = teacher_images[i].convert("RGB")
        h = min(s.height, t.height)
        s = s.resize((h, h), Image.Resampling.LANCZOS)
        t = t.resize((h, h), Image.Resampling.LANCZOS)
        combined = Image.new("RGB", (t.width + s.width, h))
        combined.paste(t, (0, 0))
        combined.paste(s, (t.width, 0))
        path = os.path.join(outdir, f"epoch_{epoch:04d}_sample_{i}.png")
        combined.save(path)
        logger.info(f"Saved sample monitor: {path}")

def pil_to_tensor(img):
    arr = np.array(img).astype(np.float32) / 255.0
    # CHW
    t = torch.from_numpy(arr).permute(2, 0, 1)
    return t

def images_to_tensor_list(imgs, device, dtype=torch.float32):
    tensors = [pil_to_tensor(img).unsqueeze(0).to(device=device, dtype=dtype) for img in imgs]
    return torch.cat(tensors, dim=0)

# ---------------------------
# Discriminator on bottleneck features
# ---------------------------
class BottleneckDisc(nn.Module):
    def __init__(self, in_channels, hidden_channels=128):
        super().__init__()
        # simple conv head that reduces spatial dims and outputs scalar logit
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
        # feat: [B, C, H, W]
        x = self.net(feat)  # [B, hidden, 1, 1]
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ---------------------------
# Main
# ---------------------------
def main(argv):
    del argv
    cfg = FLAGS.config
    # Keep run name consistent
    unique = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (getattr(cfg, "run_name", "dmd2_run")) + f"_{unique}"
    outdir = os.path.join(getattr(cfg, "logdir", "logs"), run_name)
    os.makedirs(outdir, exist_ok=True)
    logger.info(f"Output dir: {outdir}")

    # Accelerator init (mirror train_ppo_coco.py style)
    log_backend = "wandb" if getattr(cfg, "log_with_wandb", False) else None
    accelerator = Accelerator(log_with=log_backend, mixed_precision=getattr(cfg, "mixed_precision", "no"))
    if accelerator.is_main_process:
        try:
            accelerator.init_trackers(run_name, config=cfg.to_dict())
        except Exception:
            # older accelerate versions: still fine
            pass

    device = accelerator.device
    dtype = torch.float16 if accelerator.mixed_precision == "fp16" else (torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float32)
    logger.info(f"Using device {device} dtype {dtype}")

    # ---------------------------
    # Load teacher and student pipelines (keep same as your PPO script)
    # ---------------------------
    teacher_pipeline = StableDiffusionPipeline.from_pretrained(cfg.pretrained.model, revision=getattr(cfg.pretrained, "revision", None))
    # ensure DDIM-like scheduler for exact add_noise support; try to re-create scheduler from its config
    try:
        teacher_pipeline.scheduler = DDIMScheduler.from_config(teacher_pipeline.scheduler.config)
    except Exception:
        # fallback: keep existing scheduler
        pass
    teacher_pipeline.safety_checker = None
    teacher_pipeline.vae.to(device, dtype=dtype)
    teacher_pipeline.text_encoder.to(device, dtype=dtype)
    teacher_pipeline.unet.to(device, dtype=dtype)
    teacher_pipeline.text_encoder.requires_grad_(False)
    teacher_pipeline.vae.requires_grad_(False)
    teacher_pipeline.unet.requires_grad_(False)

    student_pipeline = StableDiffusionPipeline.from_pretrained(cfg.student.model, revision=getattr(cfg.student, "revision", None))
    try:
        student_pipeline.scheduler = DDIMScheduler.from_config(student_pipeline.scheduler.config)
    except Exception:
        pass
    student_pipeline.safety_checker = None
    # move to device (we'll use prepare later with accelerator)
    student_pipeline.vae.to(device, dtype=dtype)
    student_pipeline.text_encoder.to(device, dtype=dtype)
    student_pipeline.unet.to(device, dtype=dtype)
    # LoRA handling preserved from PPO script:
    if getattr(cfg, "use_lora", False):
        # replicate LoRA processor logic (same as train_ppo_coco.py)
        lora_attn_procs = {}
        for name in student_pipeline.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else student_pipeline.unet.config.cross_attention_dim
            # hidden size selection (adapted from typical diffusers pattern)
            if "mid_block" in name or name.startswith("mid_block"):
                hidden_size = student_pipeline.unet.config.block_out_channels[-1]
            elif name.startswith("down_blocks"):
                block_id = int(name.split('.')[1]) if '.' in name else 0
                hidden_size = student_pipeline.unet.config.block_out_channels[block_id]
            elif name.startswith("up_blocks"):
                block_id = int(name.split('.')[1]) if '.' in name else 0
                hidden_size = list(reversed(student_pipeline.unet.config.block_out_channels))[block_id]
            else:
                hidden_size = student_pipeline.unet.config.block_out_channels[-1]
            lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
        student_pipeline.unet.set_attn_processor(lora_attn_procs)

        class _Wrapper(AttnProcsLayers):
            def forward(self, *args, **kwargs):
                return student_pipeline.unet(*args, **kwargs)
        unet_for_optimizer = _Wrapper(student_pipeline.unet.attn_processors)
    else:
        unet_for_optimizer = student_pipeline.unet

    # ---------------------------
    # Create mu_fake as a full UNet (same config as student UNet)
    # ---------------------------
    fake_unet_config = student_pipeline.unet.config
    # instantiate a UNet with same config (fresh weights)
    mu_fake = UNet2DConditionModel(fake_unet_config)
    # Important: do NOT share weights by default (separate model)
    # Move mu_fake to device through accelerator later.

    # ---------------------------
    # Discriminator on UNet bottleneck: attach via forward hook
    # We'll register a forward hook on the student's unet.mid_block (typical location)
    # Capture mid features during forward passes and pass them to the discriminator.
    # ---------------------------
    # We'll create a simple placeholder disc later after we know feature channels.
    disc = None
    disc_hook_storage = {"feat": None}

    def mid_block_hook(module, input, output):
        # some UNet versions return a tuple/dict; handle robustly
        # output is often a tensor, possibly a ModelOutput; we'll convert if needed
        feat = output
        # If ModelOutput with .sample or similar, try to use .sample
        if hasattr(output, "sample"):
            feat = output.sample
        # store
        disc_hook_storage["feat"] = feat

    # Attach hook to student_pipeline.unet.mid_block if exists; else try to locate mid_block by name
    if hasattr(student_pipeline.unet, "mid_block"):
        student_pipeline.unet.mid_block.register_forward_hook(mid_block_hook)
    else:
        # attempt to register on the deepest block (best-effort)
        try:
            # try to find attribute named 'down_blocks' and use last block's resnets
            if hasattr(student_pipeline.unet, "down_blocks"):
                last = student_pipeline.unet.down_blocks[-1]
                last.register_forward_hook(mid_block_hook)
            else:
                logger.warning("Unable to locate mid_block hook insertion point in UNet. Discriminator-on-bottleneck won't run.")
        except Exception as e:
            logger.warning(f"Failed to register mid-block hook: {e}")

    # ---------------------------
    # Optimizers (generator uses existing optimizer in PPO script style)
    # ---------------------------
    gen_lr = getattr(cfg.train, "gen_lr", 2e-5)
    fake_lr = getattr(cfg.train, "fake_lr", 1e-4)
    disc_lr = getattr(cfg.train, "disc_lr", 1e-4)
    # generator optimizer acts on LoRA params if LoRA used, else whole unet
    gen_params = [p for p in unet_for_optimizer.parameters() if p.requires_grad]
    gen_optimizer = torch.optim.AdamW(gen_params, lr=gen_lr, betas=(0.9, 0.999), weight_decay=0.01)
    fake_optimizer = torch.optim.AdamW(mu_fake.parameters(), lr=fake_lr, betas=(0.9, 0.99))
    # disc will be created after knowing mid_channel dims; create later

    # Prepare models + optimizers with accelerator
    models_and_opts = [student_pipeline.unet, student_pipeline.text_encoder, student_pipeline.vae,
                       teacher_pipeline.unet, mu_fake, gen_optimizer, fake_optimizer]
    # We will prepare disc and its optimizer later (after creating disc)
    models_and_opts = accelerator.prepare(*models_and_opts)
    # Unpack back
    student_pipeline.unet = models_and_opts[0]
    student_pipeline.text_encoder = models_and_opts[1]
    student_pipeline.vae = models_and_opts[2]
    teacher_pipeline.unet = models_and_opts[3]
    mu_fake = models_and_opts[4]
    gen_optimizer = models_and_opts[5]
    fake_optimizer = models_and_opts[6]

    # Now create disc once we can run one forward to know mid-block channels
    # We'll do a quick dry-run with a dummy tensor to extract shape (no grads)
    try:
        dummy_bs = 1
        # create dummy latents consistent with VAE shapes
        latent_ch = student_pipeline.unet.config.in_channels
        # VAE latent spatial size: try from student_pipeline.vae.config
        vae_cfg = getattr(student_pipeline.vae, "config", None)
        sample_size = getattr(vae_cfg, "sample_size", None) or getattr(vae_cfg, "sample_sizes", [64])[-1]
        # but to be safe use 64x64 latent spatial dims if present in config
        height = getattr(cfg.sample, "height", 512)
        width = getattr(cfg.sample, "width", 512)
        down_factor = student_pipeline.unet.sample_size if hasattr(student_pipeline.unet, "sample_size") else 64
        # build dummy latents with plausible shape: [B, C, H_lat, W_lat]
        # Determining exact latent spatial dims is tricky across pipelines; we will rely on encoder/vae API later if hook fails
        lat_h = max(4, height // 8)
        lat_w = max(4, width // 8)
        dummy_lat = torch.randn((dummy_bs, student_pipeline.unet.in_channels, lat_h, lat_w), device=device, dtype=dtype)
        # run a forward pass through student's unet to trigger the hook
        _ = student_pipeline.unet(dummy_lat, torch.tensor([student_pipeline.scheduler.timesteps[0]], device=device), encoder_hidden_states=torch.zeros((1,1,student_pipeline.text_encoder.config.hidden_size), device=device))
        # check storage
        mid_feat = disc_hook_storage.get("feat", None)
        if mid_feat is not None:
            mid_channels = mid_feat.shape[1]
            disc = BottleneckDisc(in_channels=mid_channels, hidden_channels=getattr(cfg.gan, "disc_hidden", 128)).to(device)
            disc_optimizer = torch.optim.AdamW(disc.parameters(), lr=disc_lr, betas=(0.5, 0.9))
            # prepare with accelerator
            disc, disc_optimizer = accelerator.prepare(disc, disc_optimizer)
        else:
            logger.warning("mid_block hook didn't capture a feature. Discriminator will be disabled.")
            disc = None
            disc_optimizer = None
    except Exception as e:
        logger.warning(f"Failed to create discriminator due to: {e}")
        disc = None
        disc_optimizer = None

    # ---------------------------
    # Training hyperparams & helpers
    # ---------------------------
    num_epochs = getattr(cfg, "num_epochs", getattr(cfg.train, "num_epochs", 30))
    batch_size = getattr(cfg.sample, "batch_size", 4)
    fake_updates_per_gen = getattr(cfg.train, "fake_updates_per_gen", 5)
    gan_weight = getattr(cfg.gan, "weight", 1.0)
    use_gan = getattr(cfg.gan, "use_gan", False) and (disc is not None)
    # The teacher scheduler object (for add_noise)
    teacher_sched = teacher_pipeline.scheduler
    student_sched = student_pipeline.scheduler

    # Prompt sampling util (reuse from your PPO script code)
    prompt_fn = getattr(ddpo_pytorch.prompts, cfg.prompt_fn)

    # ---------------------------
    # Training loop
    # ---------------------------
    logger.info("Starting training loop")
    monitor_dir = os.path.join(outdir, "monitor")
    os.makedirs(monitor_dir, exist_ok=True)
    training_losses = []

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        # Sample prompts (reuse your existing logic)
        prompts = []
        if getattr(cfg, "prompt_source", "default") == "coco":
            # load coarse captions if config points to COCO (left simple here)
            # We assume the PPO script already had a loader; user can adapt.
            prompts = [prompt_fn() for _ in range(batch_size)]
        else:
            for _ in range(batch_size):
                p, _ = prompt_fn(**getattr(cfg, "prompt_fn_kwargs", {}))
                prompts.append(p)

        # Encode prompts with student's tokenizer & text encoder
        tokenized = student_pipeline.tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=student_pipeline.tokenizer.model_max_length)
        input_ids = tokenized.input_ids.to(device)
        with torch.no_grad():
            prompt_embeds = student_pipeline.text_encoder(input_ids)[0]

        # ---------------------------
        # Generate teacher images and teacher scores on student latents for DMD loss
        # ---------------------------
        # Generate student images (we use pipeline sampling to ensure deterministic behavior)
        student_pipeline.unet.eval()
        with torch.no_grad():
            # Use the pipeline's standard call to produce images (it internally runs the student's UNet)
            student_out = student_pipeline(prompt=prompts, num_inference_steps=getattr(cfg.student, "num_steps", 1),
                                           guidance_scale=getattr(cfg.sample, "guidance_scale", 7.5))
            student_images = student_out.images
        student_pipeline.unet.train()

        # Encode student images to latents using VAE
        # The encode API differs across VAE versions; use the common pattern: vae.encode(images).latent_dist.sample()
        with torch.no_grad():
            try:
                enc = student_pipeline.vae.encode(student_images).latent_dist.sample()
            except Exception:
                # fallback: convert with pixel values and pass to vae.encode
                t = images_to_tensor_list(student_images, device=device, dtype=dtype)
                enc = student_pipeline.vae.encode(t).latent_dist.sample()

        # Sample a random timestep t for DMD computations
        # Choose from teacher scheduler's timesteps
        t_choices = list(teacher_sched.timesteps)
        t_sample = int(random.choice(t_choices))
        timesteps_tensor = torch.tensor([t_sample] * enc.shape[0], device=device)

        # sample gaussian noise of the same shape
        noise = torch.randn_like(enc)

        # create noisy latents using teacher scheduler exact API if available
        noisy_latents = None
        try:
            # Many diffusers schedulers implement add_noise(x, noise, timesteps)
            noisy_latents = teacher_sched.add_noise(enc, noise, timesteps_tensor)
        except Exception:
            # fallback: simple scaling (this is approximate; for correctness prefer add_noise)
            logger.debug("teacher_sched.add_noise not available; using approximate add_noise fallback.")
            noisy_latents = enc + noise

        # Compute teacher's denoiser output (teacher score) at noisy latents
        with torch.no_grad():
            teacher_pred = teacher_pipeline.unet(noisy_latents, timesteps_tensor, encoder_hidden_states=prompt_embeds).sample

        # ---------------------------
        # Train mu_fake (DSM) several times (TTUR)
        # ---------------------------
        for k in range(fake_updates_per_gen):
            # mu_fake predicts noise (denoising score parameterized as predicting epsilon)
            mu_fake.train()
            pred_noise = mu_fake(noisy_latents, timesteps_tensor, encoder_hidden_states=prompt_embeds).sample
            # DSM loss: MSE between predicted noise and actual noise
            loss_mu = F.mse_loss(pred_noise, noise)
            fake_optimizer.zero_grad()
            accelerator.backward(loss_mu)
            fake_optimizer.step()

        # ---------------------------
        # Train discriminator on bottleneck features (if enabled)
        # ---------------------------
        d_loss = torch.tensor(0.0, device=device)
        if use_gan and disc is not None:
            # real images: we try to load a small batch from real_images_dir if provided in flags
            real_images_dir = getattr(cfg, "real_images_dir", None) or getattr(FLAGS, "real_images_dir", None)
            real_imgs = []
            if real_images_dir and os.path.isdir(real_images_dir):
                # pick same count as batch
                files = [os.path.join(real_images_dir, f) for f in os.listdir(real_images_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
                random.shuffle(files)
                sel = files[:enc.shape[0]]
                for p in sel:
                    try:
                        img = Image.open(p).convert("RGB").resize((getattr(cfg.sample, "width", 512), getattr(cfg.sample, "height", 512)))
                        real_imgs.append(img)
                    except Exception:
                        continue
            # if real images not provided, skip GAN step
            if len(real_imgs) >= 1:
                # encode real images to latents
                real_t = images_to_tensor_list(real_imgs, device=device, dtype=dtype)
                with torch.no_grad():
                    try:
                        real_lat = student_pipeline.vae.encode(real_t).latent_dist.sample()
                    except Exception:
                        real_lat = student_pipeline.vae.encode(real_t).latent_dist.sample()
                # forward pass through student UNet to capture mid features (we rely on the hook)
                disc_hook_storage["feat"] = None
                _ = student_pipeline.unet(real_lat, timesteps_tensor, encoder_hidden_states=prompt_embeds)
                real_feat = disc_hook_storage.get("feat", None)
                # fake features from student's latents (we already have enc)
                disc_hook_storage["feat"] = None
                _ = student_pipeline.unet(enc.detach(), timesteps_tensor, encoder_hidden_states=prompt_embeds)
                fake_feat = disc_hook_storage.get("feat", None)
                if real_feat is not None and fake_feat is not None:
                    real_logit = disc(real_feat.detach())
                    fake_logit = disc(fake_feat.detach())
                    # discriminator loss: softplus(-real) + softplus(fake)
                    d_loss = F.softplus(-real_logit).mean() + F.softplus(fake_logit).mean()
                    disc_optimizer.zero_grad()
                    accelerator.backward(d_loss)
                    disc_optimizer.step()

        # ---------------------------
        # Generator update: distribution matching + optional GAN generator loss
        # ---------------------------
        # mu_fake is now an approximation of s_fake; compute dmd loss between teacher_pred and mu_fake(noisy)
        mu_fake.eval()
        with torch.no_grad():
            fake_pred_no_grad = mu_fake(noisy_latents, timesteps_tensor, encoder_hidden_states=prompt_embeds).sample
        # For generator update we need mu_fake predictions that allow gradient to flow into the student parameters.
        # A common practical surrogate is to minimize ||teacher_pred - mu_fake(noisy_latents_detached_backprop_through_student)||^2
        # We'll re-compute noisy latents as a differentiable function of student outputs:
        # Strategy: treat enc as function( student(...) ) — but as we used pipeline to get student images then encoded them,
        # we don't have a direct differentiable pipeline call here. A practical alternative is to backprop through the student's UNet
        # by reconstructing the chain: start from random latents and run one student denoising step; this is approximate but keeps training end-to-end.
        #
        # Simpler empirical approach used in implementations: compute loss L = MSE(teacher_pred.detach(), mu_fake(noisy_latents)), then
        # backpropagate through mu_fake parameters only? That wouldn't update student. To push student, we instead compute a surrogate gradient:
        # compute L_gen = MSE(teacher_pred.detach(), mu_fake(noisy_latents_generated_by_student)) and backprop through student by reconstructing noisy_latents
        # via student unet on trainable path. For pragmatic balance we do a single differentiable student forward to produce latents and compare.
        #
        # Re-run a differentiable student forward from a random noise z to reconstruct noisy latents used above:
        # Create z0 (random latents), run student's single-step denoiser (if student.num_steps==1) or limited steps depending on config.
        gen_loss = torch.tensor(0.0, device=device)
        student_pipeline.unet.train()

        # We'll create a differentiable path: sample z ~ N(0,1) and run one student denoising step to produce latents, then add noise via teacher_sched.add_noise
        bs = enc.shape[0]
        z = torch.randn_like(enc, device=device)
        # For simplicity, take t_sample and run student's UNet once to get predicted noise/pi and build a 'reconstructed' clean latent
        # This block is approximate but creates a path for gradients to flow to UNet params
        t_tensor = timesteps_tensor
        pred_eps_by_student = student_pipeline.unet(z, t_tensor, encoder_hidden_states=prompt_embeds).sample
        # naive reconstruction (one-step): x0_est = (z - sigma_t * pred_eps) / alpha_t if alphas/sigmas available.
        # Try to use scheduler's alpha/sigma if available
        alpha_t = None
        sigma_t = None
        try:
            # Some schedulers expose 'alphas_cumprod' or 'alpha_to_t' mapping; we try the common access pattern
            if hasattr(teacher_sched, "alphas_cumprod"):
                # approximate scalar per t; timesteps may be large array — we take index
                # We attempt to map t_sample index -> alpha
                # If scheduler.timesteps is decreasing, find index
                tt_idx = list(teacher_sched.timesteps).index(t_sample) if t_sample in teacher_sched.timesteps else 0
                alpha_t = teacher_sched.alphas_cumprod[tt_idx] if hasattr(teacher_sched, "alphas_cumprod") else None
            # fallback: None
        except Exception:
            alpha_t = None

        # Fall back to simpler mixing if we cannot compute exact alpha/sigma
        if alpha_t is not None:
            # compute x0_est using standard formula (this may require converting to float tensors)
            # shape handling
            alpha_t = torch.tensor(alpha_t, device=device, dtype=z.dtype)
            # make broadcastable
            pred_x0 = (z - sigma_t * pred_eps_by_student) / (alpha_t.sqrt() if alpha_t is not None else 1.0)
        else:
            # fallback: consider student produces a denoised latent directly
            pred_x0 = z - pred_eps_by_student

        # add noise according to teacher scheduler to get noisy_latents_for_gen
        try:
            noisy_for_gen = teacher_sched.add_noise(pred_x0, noise, t_tensor)
        except Exception:
            noisy_for_gen = pred_x0 + noise

        # get mu_fake prediction on noisy_for_gen (allow gradients to flow through student via pred_x0)
        mu_fake.train()
        mu_pred_for_gen = mu_fake(noisy_for_gen, t_tensor, encoder_hidden_states=prompt_embeds).sample

        # distribution matching loss (MSE between teacher_pred and mu_pred_for_gen)
        dmd_loss = F.mse_loss(mu_pred_for_gen, teacher_pred.detach())

        # optional generator GAN loss: encourage discriminator to label generated mid features as real
        gen_gan_loss = torch.tensor(0.0, device=device)
        if use_gan and disc is not None:
            # run student unet on pred_x0 to capture mid feature (we rely on the hook)
            disc_hook_storage["feat"] = None
            _ = student_pipeline.unet(pred_x0, t_tensor, encoder_hidden_states=prompt_embeds)
            mid_feat_for_gen = disc_hook_storage.get("feat", None)
            if mid_feat_for_gen is not None:
                fake_logit_for_gen = disc(mid_feat_for_gen)
                # generator non-saturating loss
                gen_gan_loss = F.softplus(-fake_logit_for_gen).mean()
        gen_loss = dmd_loss + gan_weight * gen_gan_loss

        # optimizer step for generator (student)
        gen_optimizer.zero_grad()
        accelerator.backward(gen_loss)
        gen_optimizer.step()

        # logging
        training_losses.append({
            "epoch": epoch,
            "dmd_loss": dmd_loss.item() if isinstance(dmd_loss, torch.Tensor) else float(dmd_loss),
            "gen_gan_loss": gen_gan_loss.item() if isinstance(gen_gan_loss, torch.Tensor) else float(gen_gan_loss),
            "disc_loss": d_loss.item() if isinstance(d_loss, torch.Tensor) else 0.0
        })
        # print summary
        logger.info(f"Epoch {epoch} DMD_loss={training_losses[-1]['dmd_loss']:.6f} gen_gan={training_losses[-1]['gen_gan_loss']:.6f} disc_loss={training_losses[-1]['disc_loss']:.6f}")

        # Save checkpoint and monitor images (teacher vs student)
        if accelerator.is_main_process:
            # sample a small eval set
            eval_prompts = getattr(cfg, "eval_prompts", [
                "A cat on a chair",
                "A boy in a forest",
                "A futuristic city skyline at night",
                "A dragon flying over mountains",
            ])
            n_eval = min(len(eval_prompts), 4)
            teacher_images = []
            student_images_eval = []
            with torch.no_grad():
                for p in eval_prompts[:n_eval]:
                    t_img = teacher_pipeline(p, num_inference_steps=getattr(cfg.sample, "num_steps", 50),
                                             guidance_scale=getattr(cfg.sample, "guidance_scale", 7.5)).images[0]
                    s_img = student_pipeline(p, num_inference_steps=getattr(cfg.student, "num_steps", 1),
                                              guidance_scale=getattr(cfg.sample, "guidance_scale", 7.5)).images[0]
                    teacher_images.append(t_img)
                    student_images_eval.append(s_img)
            save_side_by_side(student_images_eval, teacher_images, epoch, monitor_dir)

            # save student pipeline weights (this will save LoRA adapters if used)
            try:
                save_path = os.path.join(outdir, f"student_epoch_{epoch:04d}")
                student_pipeline.save_pretrained(save_path)
                logger.info(f"Saved student pipeline at {save_path}")
            except Exception as e:
                logger.warning(f"Failed to save student pipeline: {e}")

    # training end
    if accelerator.is_main_process:
        try:
            import json
            stats_path = os.path.join(outdir, "training_losses.json")
            with open(stats_path, "w") as f:
                json.dump(training_losses, f, indent=2)
            logger.info(f"Training losses written to {stats_path}")
        except Exception:
            pass
        accelerator.end_training()

if __name__ == "__main__":
    app.run(main)
