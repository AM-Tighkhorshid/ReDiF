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
import json
import random


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

# --- New flags for COCO captions ---
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


def main(_):
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    config.run_name = config.run_name or unique_id
    config.run_name += f"_distill_{unique_id}"

    config.train.clip_epsilon = getattr(config.train, "clip_epsilon", 0.2) # Epsilon for PPO clipping
    config.train.kl_beta = getattr(config.train, "kl_beta", 0.04) # Beta for KL penalty

    dr_grpo_flag = getattr(config.train, "dr_grpo_flag", False)

    #"dino", "text_image", "aesthetic"
    reward_types = ["clip"]

    if dr_grpo_flag:
        outdir = "dr_"
    else:
        outdir = ""
    outdir = outdir + "GRPO_" + "_".join(reward_types)
    kl_lambda = getattr(config.train, "kl_lambda", 1.0) 
    if kl_lambda != 0:
        outdir = outdir + "_kl_" + str(kl_lambda)
    if FLAGS.prompt_source == "coco":
        outdir = outdir + "_coco_prompts2"
    else:
        outdir = outdir + "_ddpo_prompts"
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

    reward_fn = get_reward_fn(reward_types, teacher_pipeline, student_pipeline)

    group_size = getattr(config.train, "group_size", 2)  # default group size

    for epoch in range(config.num_epochs):
        logger.info(f"Epoch {epoch}: Sampling and Training")

        # --- Pick prompts ---
        if FLAGS.prompt_source == "coco" and coco_captions is not None:
            if len(coco_captions) >= config.sample.batch_size:
                prompts = random.sample(coco_captions, k=config.sample.batch_size)
            else:
                prompts = [random.choice(coco_captions) for _ in range(config.sample.batch_size)]
            prompt_pairs = [(p, None) for p in prompts]
        else:
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

        # Teacher sampling
        with torch.no_grad():
            teacher_images = teacher_pipeline(
                prompt_embeds=prompt_embeds,
                num_inference_steps=config.sample.num_steps,
                guidance_scale=config.sample.guidance_scale,
                eta=config.sample.eta,
                output_type="pil",
            ).images

        # Student sampling
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

        # Rewards
        rewards = torch.tensor(
            reward_fn(student_images, teacher_images, prompts * group_size)[0],
            device=accelerator.device,
        )

        batch_size = config.sample.batch_size

        # <--- Implement advantage normalization as per Section 4.1.2
        rewards_grouped = rewards.view(batch_size, group_size)
        rewards_mean = rewards_grouped.mean(dim=1, keepdim=True)
        rewards_std = rewards_grouped.std(dim=1, keepdim=True) + 1e-8 # Add epsilon for numerical stability
        if dr_grpo_flag == False:
            advantages_grouped = (rewards_grouped - rewards_mean)/rewards_std
        else:
            advantages_grouped = (rewards_grouped - rewards_mean)
        advantages_flat = advantages_grouped.view(-1) # Flatten back to per-sample shape

        latents = torch.stack(student_latents_all, dim=1)
        # <--- Store the log probabilities from the sampling step (`π_θ_old`)
        old_log_probs = torch.stack(student_log_probs_all, dim=1)
        
        timesteps = student_pipeline.scheduler.timesteps.repeat(batch_size * group_size, 1)


        logger.info(f"Epoch {epoch}: Training")
        student_pipeline.unet.train()
        
        # --- These will be python floats for logging ---
        total_policy_loss_epoch = 0.0
        total_kl_loss_epoch = 0.0

        old_log_probs = old_log_probs.detach()

        # Training loop over timesteps
        student_pipeline.unet.train()
        for j in range(config.student.num_steps):
            # The accumulate context manager wraps EACH step
            with accelerator.accumulate(unet):
                with accelerator.autocast():
                    # unet expects latents for current step; latents shape [batch_all, seq_len, ...]
                    noise_pred = unet(latents[:, j], timesteps[:, j], prompt_embeds).sample

                    # compute the log_prob for this step
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
                ratio = torch.exp(current_log_prob - old_log_probs[:, j])
                advantages = advantages_flat.detach()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - config.train.clip_epsilon, 1.0 + config.train.clip_epsilon) * advantages
                
                step_policy_loss = -torch.min(surr1, surr2).mean()
                
                # --- KL Divergence Penalty (Equation 3 & 4) ---
                log_ratio = old_log_probs[:, j] - current_log_prob
                kl_penalty_step = (torch.exp(log_ratio) - log_ratio - 1).mean()

                # --- Combine loss for this step ---
                total_step_loss = step_policy_loss + config.train.kl_beta * kl_penalty_step

                # --- Average the loss over timesteps ---
                # We divide by num_steps here so the gradient is scaled correctly
                # as we are backpropping on each step.
                avg_step_loss = total_step_loss / config.student.num_steps

                # --- Perform backward pass INSIDE the loop ---
                # accelerator will handle accumulation and stepping
                accelerator.backward(avg_step_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), config.train.max_grad_norm)
                
                optimizer.step()
                optimizer.zero_grad()

                # --- Accumulate python floats for logging ---
                total_policy_loss_epoch += step_policy_loss.item()
                total_kl_loss_epoch += kl_penalty_step.item()
        
        # --- End of j loop ---

        # Now, calculate final loss for logging
        avg_policy_loss = total_policy_loss_epoch / config.student.num_steps
        avg_kl_loss = total_kl_loss_epoch / config.student.num_steps
        total_loss_val = avg_policy_loss + config.train.kl_beta * avg_kl_loss

        all_losses.append(total_loss_val)
        all_rewards.append(rewards.mean().item())
        all_rewards_std.append(rewards.std().item())

        with open(stats_file, "a") as f:
            f.write(f"Epoch {epoch}: Loss={total_loss_val:.6f}, Reward_mean={rewards.mean().item():.6f}, Reward_std={rewards.std().item():.6f}\n") # Added newline

        print(f"Epoch {epoch}: Reward mean = {rewards.mean().item():.4f}, Reward std = {rewards.std().item():.4f}")
        
        if epoch % config.save_freq == 0 and accelerator.is_main_process:
            accelerator.save_state(config.student_output_dir)

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
