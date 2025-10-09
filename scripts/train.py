# train1.py
"""
Distillation script supporting:
 - prompt-based (Stable Diffusion)
 - conditional CIFAR (class-conditional)
 - unconditional CIFAR (google/ddpm-cifar10-32)

This version is hardened for the DDPM CIFAR-10 32x32 pipeline and avoids:
 - shape mismatches passed to UNet
 - repeated-backward issues (detach KL/log_probs)
 - std() dof warnings for batch_size == 1
"""

from collections import defaultdict
import os
import datetime
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    DDPMPipeline,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import numpy as np
import ddpo_pytorch.prompts
from ddpo_pytorch.rewards import get_reward_fn
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
import torch
from functools import partial
import tqdm
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import inspect

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/distill_clip.py", "Training configuration.")

logger = get_logger(__name__)

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

def build_cifar_label_embedding(num_classes, hidden_dim, device):
    return nn.Embedding(num_classes, hidden_dim).to(device)

def main(_):
    config = FLAGS.config
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    config.run_name = (config.run_name or unique_id) + f"_distill_{unique_id}"
    outdir = getattr(config, "outdir", "distill_outputs")
    os.makedirs(outdir, exist_ok=True)
    stats_file = os.path.join(outdir, "training_stats.txt")

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(getattr(config, "logdir", "."), config.run_name),
        total_limit=getattr(config, "num_checkpoint_limit", None),
    )
    accelerator = Accelerator(
        log_with="wandb" if getattr(config, "use_wandb", False) else None,
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * config.student.num_steps,
    )
    if accelerator.is_main_process and getattr(config, "use_wandb", False):
        accelerator.init_trackers("ddpo-distill", config=config.to_dict())

    logger.info(f"Config: {config}")

    # mode: "prompt" | "conditional" | "unconditional"
    mode = getattr(config, "training_mode", "unconditional")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16

    # load pipelines depending on mode
    if mode == "prompt":
        teacher_pipeline = StableDiffusionPipeline.from_pretrained(config.pretrained.model, revision=getattr(config.pretrained, "revision", None))
        teacher_pipeline.scheduler = DDIMScheduler.from_config(teacher_pipeline.scheduler.config)
        teacher_pipeline.safety_checker = None
        teacher_pipeline.set_progress_bar_config(disable=False)
        to_device = lambda p: (p.vae.to(device, dtype=dtype), p.text_encoder.to(device, dtype=dtype), p.unet.to(device, dtype=dtype))
        teacher_pipeline.vae.requires_grad_(False); teacher_pipeline.text_encoder.requires_grad_(False); teacher_pipeline.unet.requires_grad_(False)

        student_pipeline = StableDiffusionPipeline.from_pretrained(config.student.model, revision=getattr(config.student, "revision", None))
        student_pipeline.scheduler = DDIMScheduler.from_config(student_pipeline.scheduler.config)
        student_pipeline.safety_checker = None
        student_pipeline.set_progress_bar_config(disable=False)
        student_pipeline.vae.requires_grad_(False); student_pipeline.text_encoder.requires_grad_(False)
    else:
        # CIFAR paths (DDPMPipeline)
        teacher_pipeline = DDPMPipeline.from_pretrained(config.pretrained.model, revision=getattr(config.pretrained, "revision", None))
        # try converting scheduler to DDIMScheduler for compatibility with logprob wrapper
        try:
            teacher_pipeline.scheduler = DDIMScheduler.from_config(teacher_pipeline.scheduler.config)
        except Exception as e:
            raise RuntimeError("Failed to convert teacher scheduler to DDIMScheduler. Ensure pipeline/scheduler configs are compatible.") from e

        student_pipeline = DDPMPipeline.from_pretrained(config.student.model, revision=getattr(config.student, "revision", None))
        try:
            student_pipeline.scheduler = DDIMScheduler.from_config(student_pipeline.scheduler.config)
        except Exception as e:
            raise RuntimeError("Failed to convert student scheduler to DDIMScheduler. Ensure pipeline/scheduler configs are compatible.") from e

        # DDPMPipeline typically does not have VAE/text_encoder; unet is main module
        for p in teacher_pipeline.unet.parameters(): p.requires_grad = False
        for p in student_pipeline.unet.parameters(): p.requires_grad = not getattr(config, "use_lora", False)

    # optional class embedding for conditional mode
    class_embedding = None
    if mode == "conditional":
        class_embedding = build_cifar_label_embedding(getattr(config, "num_classes", 10), getattr(config, "label_embed_dim", 128), device)

    # LoRA support (optional; only if underlying UNet has attention processors)
    unet = student_pipeline.unet
    if getattr(config, "use_lora", False):
        # build LoRA processors similarly to original code (try/except in case structure differs)
        lora_attn_procs = {}
        for name in student_pipeline.unet.attn_processors.keys():
            # determine hidden_size robustly
            try:
                if name.startswith("mid_block"):
                    hidden_size = student_pipeline.unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name.split(".")[1]) if "." in name else int(name[len("up_blocks."):])
                    hidden_size = list(reversed(student_pipeline.unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name.split(".")[1]) if "." in name else int(name[len("down_blocks."):])
                    hidden_size = student_pipeline.unet.config.block_out_channels[block_id]
                else:
                    hidden_size = getattr(student_pipeline.unet.config, "sample_size", 512)
            except Exception:
                hidden_size = getattr(student_pipeline.unet.config, "in_channels", 128)
            cross_attention_dim = None if name.endswith("attn1.processor") else getattr(student_pipeline.unet.config, "cross_attention_dim", None)
            lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
        student_pipeline.unet.set_attn_processor(lora_attn_procs)
        class _Wrapper(AttnProcsLayers):
            def forward(self, *args, **kwargs):
                return student_pipeline.unet(*args, **kwargs)
        unet = _Wrapper(student_pipeline.unet.attn_processors)

    # move required parts to device and dtype
    if hasattr(student_pipeline, "vae") and student_pipeline.vae is not None:
        student_pipeline.vae.to(device, dtype=dtype)
    if mode == "prompt" and hasattr(student_pipeline, "text_encoder") and student_pipeline.text_encoder is not None:
        student_pipeline.text_encoder.to(device, dtype=dtype)
    # UNet to device in correct dtype
    if getattr(config, "use_lora", False):
        student_pipeline.unet.to(device, dtype=dtype)
    else:
        student_pipeline.unet.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=config.train.learning_rate, betas=(config.train.adam_beta1, config.train.adam_beta2), weight_decay=config.train.adam_weight_decay, eps=config.train.adam_epsilon)
    unet, optimizer = accelerator.prepare(unet, optimizer)

    prompt_fn = getattr(ddpo_pytorch.prompts, config.prompt_fn)
    reward_fn = get_reward_fn(getattr(config, "reward_types", ["dino"]), teacher_pipeline, student_pipeline)

    # helper: prepare prompt embeddings / conditioning depending on mode
    def prepare_prompt_embeddings(prompts_or_labels):
        if mode == "prompt":
            ids = student_pipeline.tokenizer(prompts_or_labels, return_tensors="pt", padding="max_length", truncation=True, max_length=student_pipeline.tokenizer.model_max_length).input_ids.to(device)
            return student_pipeline.text_encoder(ids)[0].to(device)
        elif mode == "conditional":
            labels = torch.tensor(prompts_or_labels, device=device, dtype=torch.long)
            return class_embedding(labels)
        else:
            return None

    # training loop
    all_losses, all_rewards, all_rewards_std = [], [], []
    for epoch in range(config.num_epochs):
        logger.info(f"Epoch {epoch}: Sampling and Training")

        # prepare batch prompts / labels
        if mode == "prompt":
            prompt_pairs = [prompt_fn(**config.prompt_fn_kwargs) for _ in range(config.sample.batch_size)]
            prompts = [p[0] for p in prompt_pairs]
            prompt_embeds = prepare_prompt_embeddings(prompts)
        elif mode == "conditional":
            if getattr(config, "use_prompt_fn_labels", False):
                prompt_pairs = [prompt_fn(**config.prompt_fn_kwargs) for _ in range(config.sample.batch_size)]
                labels = [p[0] for p in prompt_pairs]
            else:
                labels = np.random.randint(0, getattr(config, "num_classes", 10), size=(config.sample.batch_size,)).tolist()
            prompt_embeds = prepare_prompt_embeddings(labels)
        else:
            prompts = None
            prompt_embeds = None

        # teacher sampling
        with torch.no_grad():
            teacher_out = pipeline_with_logprob(
                teacher_pipeline,
                prompt_embeds=prompt_embeds,
                num_inference_steps=config.sample.num_steps,
                guidance_scale=getattr(config.sample, "guidance_scale", 1.0),
                eta=getattr(config.sample, "eta", 0.0),
                output_type="pt",
                return_all_latents=True,
            )
            teacher_images, _, teacher_latents_all, teacher_log_probs_all = teacher_out

        # student sampling
        student_pipeline.unet.eval()
        with accelerator.autocast():
            student_out = pipeline_with_logprob(
                student_pipeline,
                prompt_embeds=prompt_embeds,
                num_inference_steps=config.student.num_steps,
                guidance_scale=getattr(config.sample, "guidance_scale", 1.0),
                eta=getattr(config.sample, "eta", 0.0),
                output_type="pt",
                return_all_latents=True,
            )
            student_images, _, student_latents_all, student_log_probs_all = student_out
            # baseline log-probs (detach to avoid double-backward)
            log_probs = student_log_probs_all[-1].detach()

        # stack latents -> (batch, steps+1, C, H, W)
        # student_latents_all is a list of (batch, C, H, W)
        latents = torch.stack(student_latents_all, dim=1).to(device)
        # timesteps: scheduler.timesteps is 1D; repeat to (batch, steps)
        timesteps = student_pipeline.scheduler.timesteps.to(device).repeat(latents.shape[0], 1)

        # compute rewards and advantages (guard batch==1)
        rewards_np = reward_fn(student_images, teacher_images, prompts if mode == "prompt" else None)[0]
        rewards = torch.tensor(rewards_np, device=device, dtype=torch.float32)
        if rewards.numel() > 1:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            advantages = torch.zeros_like(rewards)

        # Align log probs shapes and compute KL regularizer, detach it
        aligned_teacher_log_probs = torch.stack(teacher_log_probs_all, dim=1)
        aligned_student_log_probs = torch.stack(student_log_probs_all, dim=1)
        if aligned_teacher_log_probs.shape[1] != aligned_student_log_probs.shape[1]:
            aligned_teacher_log_probs = torch.nn.functional.interpolate(
                aligned_teacher_log_probs.unsqueeze(1),
                size=aligned_student_log_probs.shape[1],
                mode="linear",
                align_corners=True,
            ).squeeze(1)
        
        # Ensure both on same device and dtype
        aligned_teacher_log_probs = aligned_teacher_log_probs.to(aligned_student_log_probs.device, dtype=aligned_student_log_probs.dtype)

        kl_loss_total = torch.nn.functional.kl_div(
            aligned_student_log_probs,
            aligned_teacher_log_probs,
            reduction="batchmean",
            log_target=True
        ).detach()

        # set up UNet expected spatial
        expected_spatial = getattr(student_pipeline.unet.config, "sample_size", None)
        if expected_spatial is None:
            # some UNet configs store spatial differently; default to 32 for CIFAR
            expected_spatial = 32

        # train UNet across student timesteps
        student_pipeline.unet.train()
        for j in range(config.student.num_steps):
            with accelerator.accumulate(unet):
                with accelerator.autocast():
                    # Select latent for step j (always keep batch dim)
                    if latents.ndim == 5:
                        latent_input = latents[:, j, :, :, :]  # (B, C, H, W)
                    elif latents.ndim == 4:
                        latent_input = latents
                    else:
                        raise ValueError(f"Unexpected latents shape: {latents.shape}")

                    # Make sure spatial size matches UNet expected size
                    cur_h, cur_w = latent_input.shape[-2], latent_input.shape[-1]
                    if (cur_h, cur_w) != (expected_spatial, expected_spatial):
                        latent_input = F.interpolate(latent_input, size=(expected_spatial, expected_spatial), mode="bilinear", align_corners=False)

                    # timestep for this step: shape (batch,)
                    timestep_input = timesteps[:, j] if timesteps.ndim > 1 else timesteps[j]

                    # safe UNet call (only pass encoder_hidden_states if supported)
                    try:
                        sig = inspect.signature(unet.forward)
                        params = set(sig.parameters.keys())
                    except Exception:
                        params = set(getattr(unet.forward, "__code__", object()).co_varnames if hasattr(unet.forward, "__code__") else [])

                    unet_kwargs = {}
                    if "encoder_hidden_states" in params and prompt_embeds is not None:
                        unet_kwargs["encoder_hidden_states"] = prompt_embeds
                    if "cross_attention_kwargs" in params and 'cross_attention_kwargs' in locals() and locals()['cross_attention_kwargs'] is not None:
                        unet_kwargs["cross_attention_kwargs"] = locals()['cross_attention_kwargs']
                    if "return_dict" in params:
                        unet_kwargs["return_dict"] = False

                    unet_out = unet(latent_input.to(next(unet.parameters()).device), timestep_input.to(next(unet.parameters()).device), **unet_kwargs)

                    # extract noise_pred robustly
                    if isinstance(unet_out, dict):
                        noise_pred = unet_out.get("sample", None) or list(unet_out.values())[0]
                    elif hasattr(unet_out, "sample"):
                        noise_pred = unet_out.sample
                    elif isinstance(unet_out, tuple):
                        noise_pred = unet_out[0]
                    else:
                        noise_pred = unet_out

                    # compute prev_sample & log_prob for this step (use latents with batch/time)
                    prev_sample = latents[:, j+1] if (j + 1) < latents.shape[1] else latents[:, j]
                    _, log_prob = ddim_step_with_logprob(
                        student_pipeline.scheduler,
                        noise_pred,
                        timesteps[:, j] if timesteps.ndim > 1 else timesteps[j],
                        latents[:, j],
                        eta=getattr(config.sample, "eta", 0.0),
                        prev_sample=prev_sample,
                    )

                # detach baseline log_probs (already detached earlier)
                ratio = torch.exp(log_prob - log_probs)  # log_probs detached already
                adv = torch.clamp(advantages, -config.train.adv_clip_max, config.train.adv_clip_max)
                unclipped = -adv * ratio
                clipped = -adv * torch.clamp(ratio, 1.0 - config.train.clip_range, 1.0 + config.train.clip_range)
                loss = torch.mean(torch.maximum(unclipped, clipped))

                total_loss = loss + getattr(config.train, "kl_lambda", 1.0) * kl_loss_total

                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), config.train.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

        # logging & saving
        total_loss_val = total_loss.item() if "total_loss" in locals() else 0.0
        all_losses.append(total_loss_val)
        all_rewards.append(rewards.mean().item())
        all_rewards_std.append(rewards.std().item() if rewards.numel() > 1 else 0.0)

        with open(stats_file, "a") as f:
            f.write(f"Epoch {epoch}: Loss={total_loss_val:.6f}, Reward_mean={rewards.mean().item():.6f}, Reward_std={(rewards.std().item() if rewards.numel()>1 else 0.0):.6f}\n")

        print(f"Epoch {epoch}: Loss={total_loss_val:.6f}, Reward mean = {rewards.mean().item():.4f}, Reward std = {(rewards.std().item() if rewards.numel()>1 else 0.0):.4f}")

        if epoch % config.save_freq == 0 and accelerator.is_main_process:
            student_pipeline.save_pretrained(config.student_output_dir)

        # simple eval visuals (unconditional/conditional/prompt)
        if accelerator.is_main_process:
            if mode == "prompt":
                eval_prompts = ["A cat on a chair", "A boy in a forest", "A futuristic city skyline at night"]
                with torch.no_grad():
                    student_eval_images = [student_pipeline(p, num_inference_steps=config.student.num_steps).images[0] for p in eval_prompts]
                    teacher_eval_images = [teacher_pipeline(p, num_inference_steps=config.sample.num_steps).images[0] for p in eval_prompts]
            elif mode == "conditional":
                eval_labels = list(range(min(5, getattr(config, "num_classes", 10))))
                with torch.no_grad():
                    student_eval_images = [student_pipeline(class_labels=l, num_inference_steps=config.student.num_steps).images[0] for l in eval_labels]
                    teacher_eval_images = [teacher_pipeline(class_labels=l, num_inference_steps=config.sample.num_steps).images[0] for l in eval_labels]
            else:
                with torch.no_grad():
                    student_eval_images = [student_pipeline(num_inference_steps=config.student.num_steps).images[0] for _ in range(5)]
                    teacher_eval_images = [teacher_pipeline(num_inference_steps=config.sample.num_steps).images[0] for _ in range(5)]
            save_side_by_side(student_eval_images, teacher_eval_images, epoch, outdir)

    # save final student
    student_pipeline.save_pretrained(config.student_output_dir)

    # plot losses & rewards
    epochs = range(len(all_losses))
    plt.figure(); plt.plot(epochs, all_losses); plt.title("Loss"); plt.savefig(os.path.join(outdir, "loss_curve.png")); plt.close()
    all_rewards = np.array(all_rewards); all_rewards_std = np.array(all_rewards_std)
    plt.figure(); plt.plot(epochs, all_rewards); plt.fill_between(epochs, all_rewards - all_rewards_std, all_rewards + all_rewards_std, alpha=0.3); plt.title("Reward"); plt.savefig(os.path.join(outdir, "reward_curve.png")); plt.close()

if __name__ == "__main__":
    app.run(main)
