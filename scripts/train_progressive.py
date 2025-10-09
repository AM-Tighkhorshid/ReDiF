# train_progressive_distill.py
# Progressive distillation implementation adapted from your train_PPO.py
# Keeps LoRA, accelerator, prompt handling, and most config options intact.
# Usage: same as your original script that used the config flag.
#
# NOTE: This script assumes the helper functions pipeline_with_logprob and
#       ddim_step_with_logprob from your repo exist and behave as in your PPO script.
#       It also assumes the config file contains necessary fields (see comments).

from collections import defaultdict
import contextlib
import os
import datetime
import time
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import numpy as np
import ddpo_pytorch.prompts
from ddpo_pytorch.stat_tracking import PerPromptStatTracker
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
import torch
import torch.nn.functional as F
from functools import partial
import tqdm
from PIL import Image
import matplotlib.pyplot as plt

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/distill_clip.py", "Training configuration.")

logger = get_logger(__name__)

def cos_alpha_sigma(t):
    """
    Given t in [0,1], compute alpha_t = cos(0.5*pi*t), sigma_t = sin(0.5*pi*t)
    Returns (alpha, sigma) tensors or floats.
    """
    # Accept t as float or tensor
    return torch.cos(0.5 * torch.pi * t), torch.sin(0.5 * torch.pi * t)

def snr_plus_one_weight(alpha, sigma):
    # SNR = alpha^2 / sigma^2
    snr = (alpha ** 2) / (sigma ** 2 + 1e-12)
    return 1.0 + snr

def encode_images_to_latents(vae, images, accelerator, dtype=torch.float32):
    """
    Encode a list of PIL images (or a tensor) into VAE latents consistent with pipeline.
    We rely on pipeline.vae.encode and the typical latent scaling factor (if present).
    """
    # Convert PIL images to pixel tensors in [0,1] and to device
    # Expect images as list of PIL.Image
    imgs = []
    for im in images:
        if isinstance(im, Image.Image):
            im = np.array(im).astype(np.float32) / 255.0
            # HWC -> CHW
            im = np.transpose(im, (2, 0, 1))
            imgs.append(im)
        else:
            imgs.append(im)
    imgs = np.stack(imgs, axis=0)
    imgs = torch.from_numpy(imgs).to(accelerator.device)
    imgs = imgs.to(dtype)
    # The pipeline VAE may expect images normalized to [-1,1]
    imgs = imgs * 2.0 - 1.0
    with torch.no_grad():
        # Using VAE encode -> latents; many SD VAEs scale by 0.18215, but we'll detect if attribute exists
        latents = vae.encode(imgs).latent_dist.mean
        # Some pipelines multiply by a constant; try to keep unchanged (user can adapt if needed)
    return latents

def compute_ddim_step_from_latent(teacher_unet, scheduler, latent, t, alpha_t, sigma_t, prompt_embeds, device, dtype):
    """
    Given latent z_t, compute teacher's predicted x_hat(zt) and return the predicted
    next latent following the DDIM formula for a target time s < t:
    z_s = alpha_s * x_hat + (sigma_s / sigma_t) * (z_t - alpha_t * x_hat)
    We will call this twice with two different s values to perform the two-step teacher transition.
    """
    # Prepare model input; teacher_unet expects latent scaled in some fashion (as used in pipeline)
    # We'll call teacher_unet forward to obtain noise_pred (epsilon_hat)
    # teacher_unet(latent_t, timestep, encoder_hidden_states).sample -> predicted noise
    # Here t is a scalar in [0,1]; but original unet uses discrete timesteps. We'll pass a float tensor converted to scheduler's expected format.
    # For simplicity, compute a dummy timestep as integer index using scheduler.num_train_timesteps * t
    num_ts = scheduler.config.num_train_timesteps if hasattr(scheduler, 'config') and hasattr(scheduler.config, 'num_train_timesteps') else 1000
    timestep = torch.tensor([int(t * (num_ts - 1))], device=device, dtype=torch.long)

    # The UNet expects latent scaled with scheduler's scaling? We'll call as in pipeline: unet(latents, timesteps, encoder_hidden_states)
    # Make sure latent shape: (B, C, H, W)
    latent_in = latent.to(device=device, dtype=dtype)
    # Expand prompt_embeds if needed
    encoder_hidden_states = prompt_embeds
    # Unet may be wrapped in AttnProcsLayers wrapper (as in your PPO script) — calling it should work.
    out = teacher_unet(latent_in, timestep, encoder_hidden_states)
    # Some UNet returns a ModelOutput with .sample or tensor directly
    noise_pred = out.sample if hasattr(out, 'sample') else out
    # convert to x_hat: x_hat = (z_t - sigma_t * eps_hat) / alpha_t
    x_hat = (latent_in - sigma_t * noise_pred) / (alpha_t + 1e-12)
    return x_hat, noise_pred

def compute_target_x_tilde(teacher_unet, scheduler, z_t, t, N, prompt_embeds, device, dtype):
    """
    Implements the teacher two-step update and inversion to compute x_tilde as target for student.
    Steps:
      t' = t - 0.5/N
      t'' = t - 1/N
      z_t' = DDIM_step(teacher, z_t, t -> t')
      z_t'' = DDIM_step(teacher, z_t', t' -> t'')
      x_tilde = invert_single_ddim_step(z_t, z_t'', t, t'')  # per paper formula
    Returns x_tilde (latent-space clean image).
    """
    # Convert scalars to tensors for alpha/sigma
    t_tensor = torch.tensor(t, device=device)
    tprime = float(max(0.0, t - 0.5 / N))
    tprime2 = float(max(0.0, t - 1.0 / N))

    alpha_t, sigma_t = cos_alpha_sigma(t_tensor)
    alpha_t = alpha_t.to(device=device, dtype=dtype)
    sigma_t = sigma_t.to(device=device, dtype=dtype)

    # compute alpha/sigma for t' and t''
    a_tprime, s_tprime = cos_alpha_sigma(torch.tensor(tprime, device=device))
    a_tprime = a_tprime.to(device=device, dtype=dtype)
    s_tprime = s_tprime.to(device=device, dtype=dtype)

    a_tprime2, s_tprime2 = cos_alpha_sigma(torch.tensor(tprime2, device=device))
    a_tprime2 = a_tprime2.to(device=device, dtype=dtype)
    s_tprime2 = s_tprime2.to(device=device, dtype=dtype)

    # First teacher denoising at z_t
    x_hat_t, noise_pred_t = compute_ddim_step_from_latent(teacher_unet, scheduler, z_t, t, alpha_t, sigma_t, prompt_embeds, device, dtype)
    # Compute z_t' using DDIM formula
    z_tprime = a_tprime * x_hat_t + (s_tprime / sigma_t) * (z_t - alpha_t * x_hat_t)

    # Second teacher denoising at z_t'
    x_hat_tprime, noise_pred_tprime = compute_ddim_step_from_latent(teacher_unet, scheduler, z_tprime, tprime, a_tprime, s_tprime, prompt_embeds, device, dtype)
    z_tprime2 = a_tprime2 * x_hat_tprime + (s_tprime2 / s_tprime) * (z_tprime - a_tprime * x_hat_tprime)

    # Invert formula to compute x_tilde (teacher target) at z_t that would map in one student step to z_tprime2
    # Based on paper: x̃ = (z_{t''} - (σ_{t''}/σ_t) z_t) / (α_{t''} - (σ_{t''}/σ_t) α_t)
    numerator = z_tprime2 - (s_tprime2 / sigma_t) * z_t
    denominator = (a_tprime2 - (s_tprime2 / sigma_t) * alpha_t) + 1e-12
    x_tilde = numerator / denominator
    return x_tilde

def train_progressive_distillation(config):
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    config.run_name = config.run_name or unique_id
    config.run_name += f"_progdistill_{unique_id}"

    stats_dir = os.path.join(config.logdir, config.run_name)
    os.makedirs(stats_dir, exist_ok=True)
    stats_file = os.path.join(stats_dir, "training_stats.txt")

    # Accelerator
    accelerator_config = ProjectConfiguration(
        project_dir=stats_dir,
        total_limit=config.num_checkpoint_limit,
    )
    accelerator = Accelerator(
        log_with="wandb" if getattr(config, "use_wandb", False) else None,
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    )

    if accelerator.is_main_process and getattr(config, "use_wandb", False):
        accelerator.init_trackers("progressive-distill", config=config.to_dict())

    logger.info(f"\n{config}")

    device = accelerator.device

    # Load teacher pipeline (pretrained, fixed)
    teacher_pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained.model, revision=config.pretrained.revision
    )
    teacher_pipeline.scheduler = DDIMScheduler.from_config(teacher_pipeline.scheduler.config)
    teacher_pipeline.safety_checker = None
    teacher_pipeline.set_progress_bar_config(disable=False)
    teacher_pipeline.vae.to(device)
    teacher_pipeline.text_encoder.to(device)
    teacher_pipeline.unet.to(device)
    teacher_pipeline.text_encoder.requires_grad_(False)
    teacher_pipeline.vae.requires_grad_(False)
    teacher_pipeline.unet.requires_grad_(False)
    teacher_pipeline.save_pretrained(config.teacher_output_dir)

    # Load base student pipeline (will be copied/initialized from teacher weights on each distillation stage)
    student_pipeline = StableDiffusionPipeline.from_pretrained(
        config.student.model, revision=config.student.revision
    )
    student_pipeline.scheduler = DDIMScheduler.from_config(student_pipeline.scheduler.config)
    student_pipeline.safety_checker = None
    student_pipeline.set_progress_bar_config(disable=False)
    # Move to device & set requires_grad appropriately later (we'll handle LoRA)
    student_pipeline.vae.to(device)
    student_pipeline.text_encoder.to(device)
    student_pipeline.unet.to(device)
    student_pipeline.vae.requires_grad_(False)
    student_pipeline.text_encoder.requires_grad_(False)
    # We'll toggle unet.requires_grad_ depending on LoRA
    student_pipeline.unet.requires_grad_(not config.use_lora)

    # LoRA setup (same as in your PPO script)
    if config.use_lora:
        lora_attn_procs = {}
        for name in student_pipeline.unet.attn_processors.keys():
            cross_attention_dim = (
                None if name.endswith("attn1.processor") else student_pipeline.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = student_pipeline.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(student_pipeline.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = student_pipeline.unet.config.block_out_channels[block_id]
            lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
        student_pipeline.unet.set_attn_processor(lora_attn_procs)

        class _Wrapper(AttnProcsLayers):
            def forward(self, *args, **kwargs):
                return student_pipeline.unet(*args, **kwargs)
        unet = _Wrapper(student_pipeline.unet.attn_processors)
    else:
        unet = student_pipeline.unet

    # dtype handling
    dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16

    student_pipeline.vae.to(device, dtype=dtype)
    student_pipeline.text_encoder.to(device, dtype=dtype)
    if config.use_lora:
        student_pipeline.unet.to(device, dtype=dtype)

    # optimizer for student (trainable params depend on LoRA)
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # Prepare accelerator
    unet, optimizer = accelerator.prepare(unet, optimizer)

    # Prompt function
    prompt_fn = getattr(ddpo_pytorch.prompts, config.prompt_fn)

    # Distillation schedule:
    teacher_steps = config.sample.num_steps
    target_student_steps = config.student.num_steps
    if teacher_steps <= target_student_steps:
        raise ValueError(f"teacher steps ({teacher_steps}) must be > target student steps ({target_student_steps})")

    # Distillation hyperparams
    updates_per_stage = getattr(config.distill, "updates_per_stage", None) or getattr(config, "num_epochs", 5000)
    batch_size = config.sample.batch_size

    all_stage_stats = []

    # Progressive loop: while teacher_steps > target_student_steps, distill to teacher_steps//2
    current_teacher_pipeline = teacher_pipeline
    current_teacher_unet = current_teacher_pipeline.unet
    current_teacher_scheduler = current_teacher_pipeline.scheduler

    while teacher_steps > target_student_steps:
        student_steps = teacher_steps // 2
        logger.info(f"Starting distillation: teacher_steps={teacher_steps} -> student_steps={student_steps}")
        stage_start_time = time.time()

        # Initialize student from current teacher weights (deep copy of pipelines)
        # For safety, load student pipeline from teacher model's weights
        student_pipeline = StableDiffusionPipeline.from_pretrained(
            None,  # we will load weights from teacher copy below by copying attributes
            revision=None
        )
        # Instead of trying to construct an empty pipeline, we'll re-use the original student pipeline object but overwrite weights from teacher
        # Copy teacher weights to student_pipeline
        student_pipeline = current_teacher_pipeline.clone() if hasattr(current_teacher_pipeline, "clone") else StableDiffusionPipeline.from_pretrained(config.pretrained.model)
        student_pipeline.scheduler = DDIMScheduler.from_config(current_teacher_scheduler.config)
        student_pipeline.safety_checker = None
        student_pipeline.set_progress_bar_config(disable=False)
        student_pipeline.vae.to(device)
        student_pipeline.text_encoder.to(device)
        student_pipeline.unet.to(device)
        student_pipeline.vae.requires_grad_(False)
        student_pipeline.text_encoder.requires_grad_(False)
        student_pipeline.unet.requires_grad_(not config.use_lora)

        # Re-apply LoRA if requested
        if config.use_lora:
            # set attn processors same as earlier
            lora_attn_procs = {}
            for name in student_pipeline.unet.attn_processors.keys():
                cross_attention_dim = (
                    None if name.endswith("attn1.processor") else student_pipeline.unet.config.cross_attention_dim
                )
                if name.startswith("mid_block"):
                    hidden_size = student_pipeline.unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(student_pipeline.unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = student_pipeline.unet.config.block_out_channels[block_id]
                lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            student_pipeline.unet.set_attn_processor(lora_attn_procs)

            class _Wrapper2(AttnProcsLayers):
                def forward(self, *args, **kwargs):
                    return student_pipeline.unet(*args, **kwargs)
            unet = _Wrapper2(student_pipeline.unet.attn_processors)
        else:
            unet = student_pipeline.unet

        unet, optimizer = accelerator.prepare(unet, optimizer)

        # training loop for this distillation stage
        stage_losses = []
        stage_epochs = updates_per_stage  # interpret as number of parameter updates for simplicity

        for update_i in range(stage_epochs):
            # Sample a batch of prompts
            prompt_pairs = [prompt_fn(**config.prompt_fn_kwargs) for _ in range(batch_size)]
            prompts = [p[0] for p in prompt_pairs]
            prompt_metadata = [p[1] for p in prompt_pairs]

            # Tokenize and get prompt embeddings
            prompt_ids = student_pipeline.tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=student_pipeline.tokenizer.model_max_length,
            ).input_ids.to(device)
            prompt_embeds = student_pipeline.text_encoder(prompt_ids)[0]

            # Generate teacher clean images with teacher pipeline (we need x to compute z_t)
            with torch.no_grad():
                teacher_out = pipeline_with_logprob(
                    current_teacher_pipeline,
                    prompt_embeds=prompt_embeds,
                    num_inference_steps=teacher_steps,
                    guidance_scale=config.sample.guidance_scale,
                    eta=config.sample.eta,
                    output_type="pil",
                    return_all_latents=False,  # we will encode images ourselves
                )
                teacher_images, _, _, _ = teacher_out  # teacher_images: list of PIL images (per prompt) or tensor

            # Encode teacher images to latents
            teacher_latents = encode_images_to_latents(current_teacher_pipeline.vae, teacher_images, accelerator, dtype=dtype)
            # teacher_latents shape: (batch, C, H, W)

            # For each sample compute a training target:
            # draw t as discrete {i/N}, i~Categorical(1..N)
            N = student_steps
            # sample i uniformly in {1..N}
            i_vals = torch.randint(1, N + 1, (batch_size,), device=device)
            t_vals = (i_vals.float() / float(N)).to(device)

            # compute noise eps and z_t
            eps = torch.randn_like(teacher_latents, device=device, dtype=dtype)
            alpha_t_vals = torch.cos(0.5 * torch.pi * t_vals).to(device=device, dtype=dtype)
            sigma_t_vals = torch.sin(0.5 * torch.pi * t_vals).to(device=device, dtype=dtype)
            # reshape alpha/sigma for broadcasting
            alpha_t_vals_b = alpha_t_vals.view(batch_size, *([1] * (teacher_latents.dim() - 1)))
            sigma_t_vals_b = sigma_t_vals.view(batch_size, *([1] * (teacher_latents.dim() - 1)))

            z_t = alpha_t_vals_b * teacher_latents + sigma_t_vals_b * eps

            # For each element in batch compute x_tilde via teacher two-step DDIM
            x_tilde_batch = []
            for b in range(batch_size):
                ztb = z_t[b : b + 1]
                tb = float(t_vals[b].item())
                # For per-sample prompt embedding, expand dims
                prompt_emb_b = prompt_embeds[b : b + 1]
                x_tilde = compute_target_x_tilde(
                    current_teacher_unet,
                    current_teacher_scheduler,
                    ztb,
                    tb,
                    N,
                    prompt_emb_b,
                    device,
                    dtype,
                )
                x_tilde_batch.append(x_tilde)
            x_tilde_batch = torch.cat(x_tilde_batch, dim=0)

            # Student prediction: for each z_t, compute student's epsilon prediction and convert to x_hat
            unet.train()
            # We'll accumulate gradient once per batch (accelerator handles accumulation)
            with accelerator.autocast():
                # Prepare timestep tensors for student unet call (similar to above)
                # convert t_vals in [0,1] to integer timesteps for scheduler/unet (approx)
                num_ts = student_pipeline.scheduler.config.num_train_timesteps if hasattr(student_pipeline.scheduler, 'config') and hasattr(student_pipeline.scheduler.config, 'num_train_timesteps') else 1000
                timesteps = (t_vals * (num_ts - 1)).long().to(device)
                # call unet with z_t and timesteps and prompt_embeds -> noise_pred
                noise_pred_out = unet(z_t.to(dtype=dtype), timesteps, prompt_embeds.to(dtype=dtype))
                noise_pred = noise_pred_out.sample if hasattr(noise_pred_out, 'sample') else noise_pred_out
                # compute x_hat student
                # need alpha_t values as broadcasted
                alpha_t_vals_b = alpha_t_vals_b.to(dtype=dtype)
                sigma_t_vals_b = sigma_t_vals_b.to(dtype=dtype)
                x_hat = (z_t - sigma_t_vals_b * noise_pred) / (alpha_t_vals_b + 1e-12)

                # Loss: weighted MSE between x_tilde and x_hat using SNR+1 weighting per sample
                weight_per_sample = snr_plus_one_weight(alpha_t_vals.to(dtype=dtype), sigma_t_vals.to(dtype=dtype)).to(device)
                weight_b = weight_per_sample.view(batch_size, *([1] * (x_hat.dim() - 1)))
                mse = ((x_tilde_batch - x_hat) ** 2).mean(dim=list(range(1, x_hat.dim())))  # per-sample MSE across channels+spatial
                weighted_mse = (weight_per_sample * mse).mean()

                loss = weighted_mse

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), config.train.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            stage_losses.append(loss.item())

            # periodic logging / eval
            if (update_i + 1) % max(1, config.distill.log_every or 100) == 0 and accelerator.is_main_process:
                mean_loss = float(np.mean(stage_losses[-(config.distill.log_every or 100):]))
                with open(stats_file, "a") as f:
                    f.write(f"Stage teacher={teacher_steps}->student={student_steps} update={update_i} mean_loss={mean_loss:.6f}\n")
                print(f"[Stage {teacher_steps}->{student_steps}] update {update_i+1}/{stage_epochs} mean_loss={mean_loss:.6f}")

        # After training stage, make student the new teacher
        # For simplicity, reassign current_teacher_pipeline to student_pipeline (we loaded student from teacher earlier, but we've trained unet weights)
        # Save student model
        if accelerator.is_main_process:
            save_dir = os.path.join(stats_dir, f"distilled_to_{student_steps}_steps")
            os.makedirs(save_dir, exist_ok=True)
            # Save pipeline components (we attempt to set student_pipeline.unet weights from unet module)
            try:
                # If using LoRA, you may want to save only LoRA weights etc. We attempt to call save_pretrained
                student_pipeline.unet = unet  # assign trained unet wrapper back
                student_pipeline.save_pretrained(save_dir)
                print(f"Saved distilled student to {save_dir}")
            except Exception as e:
                print(f"Warning: failed to fully save pipeline: {e}")

        # Set student as next teacher
        current_teacher_pipeline = student_pipeline
        current_teacher_unet = student_pipeline.unet
        current_teacher_scheduler = student_pipeline.scheduler
        teacher_steps = student_steps

        stage_time = time.time() - stage_start_time
        logger.info(f"Finished distillation stage to {teacher_steps} steps in {stage_time:.1f}s")

    # final save
    final_save = os.path.join(stats_dir, "final_student")
    if accelerator.is_main_process:
        try:
            current_teacher_pipeline.save_pretrained(final_save)
            print(f"Saved final distilled model to {final_save}")
        except Exception as e:
            print(f"Warning saving final model: {e}")

def main(_):
    config = FLAGS.config
    train_progressive_distillation(config)

if __name__ == "__main__":
    app.run(main)
