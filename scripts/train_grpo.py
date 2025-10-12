from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
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

logger = get_logger(__name__)


def main(_):
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    config.run_name = config.run_name or unique_id
    config.run_name += f"_distill_{unique_id}"

    # <--- CORRECTION: Add PPO clipping and KL beta from paper's equations to config
    config.train.clip_epsilon = getattr(config.train, "clip_epsilon", 0.2) # Epsilon for PPO clipping
    config.train.kl_beta = getattr(config.train, "kl_beta", 0.04) # Beta for KL penalty


    reward_types = ["clip", "dino", "text_image"]
    

    outdir = "GRPO_" + "_".join(reward_types)
    kl_lambda = getattr(config.train, "kl_lambda", 1.0) 
    if kl_lambda != 0:
        outdir = outdir + "_kl_" + str(kl_lambda)
    stats_dir = outdir
    os.makedirs(stats_dir, exist_ok=True)
    stats_file = os.path.join(stats_dir, "training_stats.txt")

    all_losses = []
    all_rewards = []
    all_rewards_std = []

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        total_limit=config.num_checkpoint_limit,
    )
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * config.student.num_steps,
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("ddpo-distill", config=config.to_dict())

    logger.info(f"\n{config}")

    teacher_pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained.model
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

    student_pipeline = StableDiffusionPipeline.from_pretrained(
        config.student.model
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

    prompt_fn = getattr(ddpo_pytorch.prompts, config.prompt_fn)

    reward_fn = get_reward_fn(reward_types, teacher_pipeline, student_pipeline)

    group_size = getattr(config.train, "group_size", 2)  # default group size

    for epoch in range(config.num_epochs):
        logger.info(f"Epoch {epoch}: Sampling and Training")

        prompt_pairs = [prompt_fn(**config.prompt_fn_kwargs) for _ in range(config.sample.batch_size)]
        prompts = [p[0] for p in prompt_pairs]

        # repeat prompts group_size times and get embeddings for all samples
        prompt_ids = student_pipeline.tokenizer(
            prompts * group_size,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=student_pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
        prompt_embeds = student_pipeline.text_encoder(prompt_ids)[0]
        print(prompt_embeds.shape)

        # Teacher sampling (repeated to match group repeats so teacher latents/logprobs align)
        with torch.no_grad():
            teacher_images = teacher_pipeline(
                prompt_embeds=prompt_embeds,
                num_inference_steps=config.sample.num_steps,
                guidance_scale=config.sample.guidance_scale,
                eta=config.sample.eta,
                output_type="pil",
            ).images

        # Student sampling (inference mode) to get images, latents, and baseline log-probs
        student_pipeline.unet.eval()
        with accelerator.autocast():
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
            # baseline log_probs from the pipeline (usually final-step logprob per sample)
            baseline_log_probs = student_log_probs_all[-1]

        # Compute rewards per sample
        rewards = torch.tensor(
            reward_fn(student_images, teacher_images, prompts * group_size)[0],
            device=accelerator.device,
        )
        
        del teacher_images
        del student_images

        batch_size = config.sample.batch_size

        # <--- CORRECTION: Implement advantage normalization as per Section 4.1.2
        rewards_grouped = rewards.view(batch_size, group_size)
        rewards_mean = rewards_grouped.mean(dim=1, keepdim=True)
        rewards_std = rewards_grouped.std(dim=1, keepdim=True) + 1e-8 # Add epsilon for numerical stability
        advantages_grouped = (rewards_grouped - rewards_mean)
        advantages_flat = advantages_grouped.view(-1) # Flatten back to per-sample shape

        latents = torch.stack(student_latents_all, dim=1)
        # <--- CORRECTION: Store the log probabilities from the sampling step (`π_θ_old`)
        old_log_probs = torch.stack(student_log_probs_all, dim=1)
        
        timesteps = student_pipeline.scheduler.timesteps.repeat(batch_size * group_size, 1)

        # <--- CORRECTION: The entire training logic is refactored below
        logger.info(f"Epoch {epoch}: Training")
        student_pipeline.unet.train()
        
        # Accumulate loss over all timesteps
        total_policy_loss = 0
        total_kl_loss = 0

        old_log_probs = old_log_probs.detach()

        # Training loop over timesteps: keep the DDIM step with logprob as in PPO flow
        student_pipeline.unet.train()
        for j in range(config.student.num_steps):
            with accelerator.accumulate(unet):
                with accelerator.autocast():
                    # unet expects latents for current step; latents shape [batch_all, seq_len, ...]
                    noise_pred = unet(latents[:, j], timesteps[:, j], prompt_embeds).sample

                    # compute the log_prob for this step using the DDIM helper (SDE->ODE conversion)
                    # prev_sample argument matches the pipeline order: next latent if available, else current
                    _, current_log_prob = ddim_step_with_logprob(
                        student_pipeline.scheduler,
                        noise_pred,
                        timesteps[:, j],
                        latents[:, j],
                        eta=config.sample.eta,
                        prev_sample=latents[:, j+1] if j + 1 < latents.shape[1] else latents[:, j],
                    )                    

                current_log_prob = current_log_prob.view(-1)
            
                # --- Policy Loss with PPO Clipping (Equation 3) ---
                # Ratio of probabilities `π_θ(a|s) / π_θ_old(a|s)`
                ratio = torch.exp(current_log_prob - old_log_probs[:, j])
                
                # Detach advantages so no gradients flow into the reward calculation
                advantages = advantages_flat.detach()

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - config.train.clip_epsilon, 1.0 + config.train.clip_epsilon) * advantages
                
                # The policy loss is the negative of the minimum of the two surrogate objectives
                step_policy_loss = -torch.min(surr1, surr2).mean()
                total_policy_loss += step_policy_loss
                
                # --- KL Divergence Penalty (Equation 3 & 4) ---
                # The paper's reference model `π_ref` is the SFT model. Here, we use `π_θ_old`
                # (the policy at the start of the batch) as the reference for regularization.
                log_ratio = old_log_probs[:, j] - current_log_prob
                # Unbiased estimator for KL divergence from the paper (Equation 4)
                # D_KL(π_ref || π_θ) ≈ (π_ref / π_θ) - log(π_ref / π_θ) - 1
                kl_penalty_step = (torch.exp(log_ratio) - log_ratio - 1).mean()
                total_kl_loss += kl_penalty_step

        # Average the losses over the number of timesteps
        avg_policy_loss = total_policy_loss / config.student.num_steps
        avg_kl_loss = total_kl_loss / config.student.num_steps

        # Combine the losses
        total_loss = avg_policy_loss + config.train.kl_beta * avg_kl_loss

        # Perform a single backward pass for the entire trajectory
        with accelerator.accumulate(unet):
            accelerator.backward(total_loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(unet.parameters(), config.train.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
        # <--- END OF CORRECTION BLOCK

        total_loss_val = total_loss.item()
        all_losses.append(total_loss_val)
        all_rewards.append(rewards.mean().item())
        all_rewards_std.append(rewards.std().item())

        with open(stats_file, "a") as f:
            f.write(f"Epoch {epoch}: Loss={total_loss_val:.6f}, Reward_mean={rewards.mean().item():.6f}, Reward_std={rewards.std().item():.6f}")

        print(f"Epoch {epoch}: Reward mean = {rewards.mean().item():.4f}, Reward std = {rewards.std().item():.4f}")

        if epoch % config.save_freq == 0 and accelerator.is_main_process:
            accelerator.save_state(config.student_output_dir)

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
                            num_inference_steps=config.student.num_steps,
                            guidance_scale=config.sample.guidance_scale,
                        ).images[0]
                        for prompt in eval_prompts
                    ]

        save_side_by_side(student_eval_images, teacher_eval_images, epoch, outdir)

        student_pipeline.save_pretrained(stats_dir + "/student_model/")

    epochs = range(len(all_losses))

    plt.figure()
    plt.plot(epochs, all_losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.savefig(os.path.join(stats_dir, "loss_curve.png"))
    plt.close()

    all_rewards = np.array(all_rewards)
    all_rewards_std = np.array(all_rewards_std)
    plt.figure()
    plt.plot(epochs, all_rewards, label="Reward mean")
    plt.fill_between(epochs, all_rewards - all_rewards_std, all_rewards + all_rewards_std, alpha=0.3, label="Reward std")
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.title("Reward Curve")
    plt.legend()
    plt.savefig(os.path.join(stats_dir, "reward_curve.png"))
    plt.close()


if __name__ == "__main__":
    app.run(main)
