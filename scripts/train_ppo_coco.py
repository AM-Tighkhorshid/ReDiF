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

def js_divergence(p, q):
    p = F.softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    m = 0.5 * (p + q)
    js = 0.5 * (F.kl_div(p.log(), m, reduction="batchmean", log_target=False) +
                F.kl_div(q.log(), m, reduction="batchmean", log_target=False))
    return js

def chi2_divergence(p, q):
    p = F.softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    div = torch.mean(torch.sum(((p - q) ** 2) / (q + 1e-8), dim=-1))
    return div

def power_divergence(p, q, power=1.2):
    p = F.softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    div = torch.mean(torch.sum((p ** power * q ** (1 - power) - 1) / (power * (power - 1)), dim=-1))
    return div

def renyi_divergence(p, q, alpha=0.6):
    p = F.softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    div = (1.0 / (alpha - 1.0)) * torch.log(torch.sum(p ** alpha * q ** (1 - alpha), dim=-1) + 1e-8)
    return torch.mean(div)

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
flags.DEFINE_enum(
    "divergence_type",
    default="kl",
    enum_values=["kl", "js", "chi2", "power", "renyi"],
    help="Divergence function to use for PPO regularization."
)

flags.DEFINE_float(
    "divergence_param",
    default=0.2,
    help="Extra parameter (p or α) for power or Rényi divergences."
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

    div_type = FLAGS.divergence_type.lower()
    reward_types = ["clip"]
    outdir = "PPO_" + "_".join(reward_types)
    kl_lambda = getattr(config.train, "kl_lambda", 1.0)
    if kl_lambda != 0:
        outdir = outdir + "_" + div_type + "_" + str(kl_lambda)
    if FLAGS.prompt_source == "coco":
        outdir = outdir + "_coco_prompts"
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

    set_seed(config.seed, device_specific=True)

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

    # Reward function
    reward_fn = get_reward_fn(reward_types, teacher_pipeline, student_pipeline)

    for epoch in range(config.num_epochs):
        logger.info(f"Epoch {epoch}: Sampling and Training")

        # --- Pick prompts ---
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

        # --- Encode prompts ---
        prompt_ids = student_pipeline.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=student_pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
        prompt_embeds = student_pipeline.text_encoder(prompt_ids)[0]

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
            log_probs = student_log_probs_all[-1]

        latents = torch.stack(student_latents_all, dim=1)
        timesteps = student_pipeline.scheduler.timesteps.repeat(config.sample.batch_size, 1)

        rewards = torch.tensor(
            reward_fn(student_images, teacher_images, prompts)[0], device=accelerator.device
        )

        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        aligned_teacher_log_probs = torch.stack(teacher_log_probs_all, dim=1)
        aligned_student_log_probs = torch.stack(student_log_probs_all, dim=1)

        if aligned_teacher_log_probs.shape[1] != aligned_student_log_probs.shape[1]:
            aligned_teacher_log_probs = torch.nn.functional.interpolate(
                aligned_teacher_log_probs.unsqueeze(1),
                size=aligned_student_log_probs.shape[1],
                mode="linear",
                align_corners=True,
            ).squeeze(1)


        div_type = FLAGS.divergence_type.lower()
        if div_type == "kl":
            kl_loss_total = kl_divergence(aligned_student_log_probs, aligned_teacher_log_probs)
        elif div_type == "js":
            kl_loss_total = js_divergence(aligned_student_log_probs, aligned_teacher_log_probs)
        elif div_type == "chi2":
            kl_loss_total = chi2_divergence(aligned_student_log_probs, aligned_teacher_log_probs)
        elif div_type == "power":
            kl_loss_total = power_divergence(aligned_student_log_probs, aligned_teacher_log_probs, FLAGS.divergence_param)
        elif div_type == "renyi":
            kl_loss_total = renyi_divergence(aligned_student_log_probs, aligned_teacher_log_probs, FLAGS.divergence_param)
        else:
            raise ValueError(f"Unknown divergence type: {div_type}")

        # kl_loss_total = torch.nn.functional.kl_div(
        #     aligned_student_log_probs,
        #     aligned_teacher_log_probs,
        #     reduction="batchmean",
        #     log_target=True
        # )

        student_pipeline.unet.train()
        for j in range(config.student.num_steps):
            with accelerator.accumulate(unet):
                with accelerator.autocast():
                    noise_pred = unet(latents[:, j], timesteps[:, j], prompt_embeds).sample
                    _, log_prob = ddim_step_with_logprob(
                        student_pipeline.scheduler,
                        noise_pred,
                        timesteps[:, j],
                        latents[:, j],
                        eta=config.sample.eta,
                        prev_sample=latents[:, j+1] if j + 1 < latents.shape[1] else latents[:, j],
                    )
                adv = torch.clamp(advantages, -config.train.adv_clip_max, config.train.adv_clip_max)
                print(advantages)
                ratio = torch.exp(log_prob - log_probs)
                unclipped = -adv * ratio
                print(ratio)
                clipped = -adv * torch.clamp(ratio, 1.0 - config.train.clip_range, 1.0 + config.train.clip_range)
                loss = torch.mean(torch.maximum(unclipped, clipped))

                total_loss = loss + getattr(config.train, "kl_lambda", 1.0) * kl_loss_total

                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), config.train.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

        total_loss_val = total_loss.item() if 'total_loss' in locals() else 0.0
        all_losses.append(total_loss_val)
        all_rewards.append(rewards.mean().item())
        all_rewards_std.append(rewards.std().item())

        with open(stats_file, "a") as f:
            f.write(f"Epoch {epoch}: Loss={total_loss_val:.6f}, Reward_mean={rewards.mean().item():.6f}, Reward_std={rewards.std().item():.6f}\n")

        print(f"Epoch {epoch}: Reward mean = {rewards.mean().item():.4f}, Reward std = {rewards.std().item():.4f}")

        if accelerator.is_main_process:
            eval_prompts = [
                "A glass bowl filled with oranges on a table",
                "A cat pausing as it's picture is taken",
                "A futuristic city skyline at night",
                "A Marine that is looking at his cell phone",
                "A cozy cabin in a snowy forest",
                "A motorcycle is parked near a puddle and a van"
            ]

            # generator = torch.Generator(device=accelerator.device).manual_seed(config.seed)

            with torch.no_grad():
                with accelerator.autocast():
                    teacher_eval_images = [
                        teacher_pipeline(
                            prompt,
                            num_inference_steps=config.sample.num_steps,
                            guidance_scale=config.sample.guidance_scale,
                            # generator=generator
                        ).images[0]
                        for prompt in eval_prompts
                    ]

                    student_eval_images = [
                        student_pipeline(
                            prompt,
                            num_inference_steps=config.student.num_steps,
                            guidance_scale=config.sample.guidance_scale,
                            # generator=generator
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
