"""
Teacher (50-step DDIM) → Student (5-step DDIM) distillation via PPO.

Reward   : composite reward from rewards.py
           (clip | dino | perception | text_image | aesthetic, or any combo)
KL reg   : analytic KL on final latent  N(μ_student, I) || N(μ_teacher, I)
           = ½ ||μ_s − μ_t||²   (no step-alignment needed)

PPO loop : old log-probs stored during no-grad rollout;
           new log-probs recomputed per gradient step  → correct importance ratio
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
import torch.distributed
import torch.nn.functional as F
from functools import partial
import tqdm
import matplotlib.pyplot as plt
from PIL import Image

import ddpo_pytorch.prompts
# rewards.py must be on the Python path (e.g. same directory or installed package)
from ddpo_pytorch.rewards import get_reward_fn
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

# ---------------------------------------------------------------------------
# Flags
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
    "kl_lambda", 0,
    "Weight λ for the latent KL regularisation term in the total loss.",
)
flags.DEFINE_list(
    "reward_types", ["clip"],
    "Comma-separated reward(s): clip, dino, perception, text_image, aesthetic. "
    "Example: --reward_types=clip,dino",
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Utilities
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


def rewards_to_tensor(rewards_raw, device) -> torch.Tensor:
    """Convert list[float] or np.ndarray reward output to a (B,) float32 tensor."""
    if isinstance(rewards_raw, torch.Tensor):
        return rewards_raw.to(device=device, dtype=torch.float32)
    return torch.tensor(np.array(rewards_raw, dtype=np.float32), device=device)


def latent_kl(student_latent: torch.Tensor,
              teacher_latent: torch.Tensor) -> torch.Tensor:
    """
    Analytic KL(N(μ_s, I) || N(μ_t, I)) = ½ ||μ_s − μ_t||²  averaged over batch.
    Both tensors: (B, C, H, W).
    """
    s = student_latent.float().view(student_latent.shape[0], -1)  # (B, D)
    t = teacher_latent.float().view(teacher_latent.shape[0], -1)  # (B, D)
    return 0.5 * ((s - t) ** 2).sum(dim=-1).mean()               # scalar


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(_):
    config       = FLAGS.config
    reward_types = list(FLAGS.reward_types)   # e.g. ["clip"] or ["clip","dino"]

    unique_id        = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    config.run_name  = (config.run_name or unique_id) + f"_distill_{unique_id}"

    reward_tag = "_".join(reward_types)
    outdir  = f"PPO_{reward_tag}_kl{FLAGS.kl_lambda}"
    outdir += "_coco" if FLAGS.prompt_source == "coco" else "_ddpo"
    outdir += "_batch_size_" + str(config.sample.batch_size)

    outdir = outdir + "_____third_try"

    stats_dir  = outdir
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
        # Accumulate over ALL student steps in one virtual batch
        gradient_accumulation_steps=(
            config.train.gradient_accumulation_steps * config.student.num_steps
        ),
    )
    set_seed(config.seed, device_specific=True)

    # Create output dir only on main process
    if accelerator.is_main_process:
        os.makedirs(stats_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        accelerator.init_trackers(
            "ddpo-distill",
            config={**config.to_dict(), "reward_types": reward_types,
                    "kl_lambda": FLAGS.kl_lambda},
        )

    logger.info(f"\n{config}")
    logger.info(f"Reward types : {reward_types}")
    logger.info(f"KL λ         : {FLAGS.kl_lambda}")

    # ------------------------------------------------------------------ #
    # Teacher pipeline  (all frozen)
    # ------------------------------------------------------------------ #
    # Load to CPU first; each rank will move to its own device after prepare().
    teacher_pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained.model, revision=config.pretrained.revision
    )
    teacher_pipeline.scheduler = DDIMScheduler.from_config(
        teacher_pipeline.scheduler.config
    )
    teacher_pipeline.safety_checker = None
    teacher_pipeline.set_progress_bar_config(disable=True)
    teacher_pipeline.vae.requires_grad_(False)
    teacher_pipeline.text_encoder.requires_grad_(False)
    teacher_pipeline.unet.requires_grad_(False)
    if accelerator.is_main_process:
        teacher_pipeline.save_pretrained(config.teacher_output_dir)

    # ------------------------------------------------------------------ #
    # Student pipeline  (UNet or LoRA trainable)
    # ------------------------------------------------------------------ #
    student_pipeline = StableDiffusionPipeline.from_pretrained(
        config.student.model, revision=config.student.revision
    )
    student_pipeline.scheduler = DDIMScheduler.from_config(
        student_pipeline.scheduler.config
    )
    student_pipeline.safety_checker = None
    student_pipeline.set_progress_bar_config(disable=True)
    # NOTE: do NOT call .to(device) here — accelerator.prepare() handles placement
    # so each rank lands on its own GPU. Calling .to(accelerator.device) before
    # prepare() causes both ranks to target GPU 0, which causes CUDA crashes.
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

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )
    # prepare() must come FIRST — it assigns each rank to its own GPU device.
    # Only after this call is accelerator.device reliable per-rank.
    unet, optimizer = accelerator.prepare(unet, optimizer)

    # ------------------------------------------------------------------ #
    # Move everything to the correct per-rank device AFTER prepare()
    # ------------------------------------------------------------------ #
    # accelerator.prepare() already placed `unet` (and wrapped optimizer) on the
    # right device.  We now move the frozen components of both pipelines.
    teacher_pipeline.to(accelerator.device)

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(
        accelerator.mixed_precision, torch.float32
    )
    student_pipeline.vae.to(accelerator.device, dtype=dtype)
    student_pipeline.text_encoder.to(accelerator.device, dtype=dtype)
    if config.use_lora:
        student_pipeline.unet.to(accelerator.device, dtype=dtype)

    if accelerator.is_main_process:
        lora_params = sum(
            p.numel()
            for m in student_pipeline.unet.attn_processors.values()
            for p in m.parameters()
        )
        logger.info(f"LoRA parameters: {lora_params:,}")

    # ------------------------------------------------------------------ #
    # Reward function  (built AFTER accelerator.prepare so device is
    # correctly set per rank — avoids both processes loading onto GPU 0)
    # ------------------------------------------------------------------ #
    accelerator.wait_for_everyone()
    reward_fn = get_reward_fn(reward_types, teacher_pipeline, student_pipeline)

    # ------------------------------------------------------------------ #
    # Prompts
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
        logger.info(f"Epoch {epoch}: Sampling + Training")

        # ---- Prompt batch ------------------------------------------------
        # IMPORTANT: sample prompts only on rank 0, then broadcast to all ranks.
        # Every GPU must run the same prompts so that accelerator.gather(rewards)
        # aggregates comparable values and the advantage slice-back is correct.
        if accelerator.is_main_process:
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
        else:
            prompts = [None] * config.sample.batch_size

        # Broadcast the list of strings from rank 0 to all other ranks.
        # broadcast_object_list requires torch.distributed to be initialised,
        # which Accelerate does automatically in multi-GPU mode.
        if accelerator.num_processes > 1:
            prompt_container = [prompts]
            torch.distributed.broadcast_object_list(prompt_container, src=0)
            prompts = prompt_container[0]

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
        # Rollout  (no gradients)
        # ---------------------------------------------------------------- #
        with torch.no_grad():

            # --- Teacher: 50 DDIM steps ----------------------------------
            with accelerator.autocast():
                teacher_images, _, teacher_latents_all, _ = pipeline_with_logprob(
                    teacher_pipeline,
                    prompt_embeds=prompt_embeds,
                    num_inference_steps=config.sample.num_steps,   # 50
                    guidance_scale=config.sample.guidance_scale,
                    eta=config.sample.eta,
                    output_type="pt",
                    return_all_latents=True,
                )
            # teacher_latents_all: list[Tensor(B,C,h,w)], length = 51
            teacher_final_latent = teacher_latents_all[-1]   # (B, C, h, w)

            # --- Student: 5 DDIM steps  (OLD policy) ---------------------
            student_pipeline.unet.eval()
            with accelerator.autocast():
                student_images, _, student_latents_all, student_log_probs_all = \
                    pipeline_with_logprob(
                        student_pipeline,
                        prompt_embeds=prompt_embeds,
                        num_inference_steps=config.student.num_steps,  # 5
                        guidance_scale=config.sample.guidance_scale,
                        eta=config.sample.eta,
                        output_type="pt",
                        return_all_latents=True,
                    )
            # student_latents_all   : list[Tensor(B,C,h,w)], length = 6
            # student_log_probs_all : list[Tensor(B,)],      length = 5

            # Stack for gradient phase
            latents      = torch.stack(student_latents_all, dim=1)       # (B, T+1, C, h, w)
            old_log_probs = torch.stack(student_log_probs_all, dim=1)    # (B, T)

            student_final_latent = student_latents_all[-1]               # (B, C, h, w)

        # (B, T) timestep schedule for student
        timesteps = student_pipeline.scheduler.timesteps.repeat(
            config.sample.batch_size, 1
        )

        # ---------------------------------------------------------------- #
        # Reward  — calls composite_reward_fn from rewards.py
        # teacher_images and student_images are (B,C,H,W) tensors in [0,1]
        # ---------------------------------------------------------------- #
        with torch.no_grad():
            rewards_raw, reward_details = reward_fn(
                student_images, teacher_images, prompts
            )

        rewards = rewards_to_tensor(rewards_raw, accelerator.device)  # (B,)

        # Gather rewards from all GPUs before normalising advantages
        rewards = accelerator.gather(rewards)                          # (B * num_gpus,)

        # Normalised, clamped advantages
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        advantages = torch.clamp(
            advantages,
            -config.train.adv_clip_max,
             config.train.adv_clip_max,
        )  # (B * num_gpus,)

        # Slice back to this rank's local batch
        local_bs   = config.sample.batch_size
        rank       = accelerator.process_index
        advantages = advantages[rank * local_bs : (rank + 1) * local_bs]  # (B,)

        # ---------------------------------------------------------------- #
        # Latent KL regularisation
        # ---------------------------------------------------------------- #
        kl_reg = latent_kl(student_final_latent, teacher_final_latent)  # scalar

        # ---------------------------------------------------------------- #
        # PPO gradient steps over student denoising steps
        # ---------------------------------------------------------------- #
        student_pipeline.unet.train()
        total_loss = torch.tensor(0.0, device=accelerator.device)

        for j in range(config.student.num_steps):
            with accelerator.accumulate(unet):
                with accelerator.autocast():
                    noise_pred = unet(
                        latents[:, j],
                        timesteps[:, j],
                        prompt_embeds,
                    ).sample   # (B, C, h, w)

                    # New log-prob under current (updated) policy
                    _, new_log_prob = ddim_step_with_logprob(
                        student_pipeline.scheduler,
                        noise_pred,
                        timesteps[:, j],
                        latents[:, j],
                        eta=config.sample.eta,
                        prev_sample=(
                            latents[:, j + 1]
                            if j + 1 < latents.shape[1]
                            else latents[:, j]
                        ),
                    )
                    # new_log_prob : (B,)

                # Importance ratio  π_new / π_old
                ratio  = torch.exp(new_log_prob - old_log_probs[:, j])   # (B,)

                # PPO clipped surrogate  (we MINIMISE, so negate the reward)
                surr1  = -advantages * ratio
                surr2  = -advantages * torch.clamp(
                    ratio,
                    1.0 - config.train.clip_range,
                    1.0 + config.train.clip_range,
                )
                policy_loss = torch.mean(torch.maximum(surr1, surr2))

                step_loss   = policy_loss + FLAGS.kl_lambda * kl_reg
                total_loss  = total_loss + step_loss

                accelerator.backward(step_loss)
                if accelerator.sync_gradients:
                    torch.nn.utils.clip_grad_norm_(
                        unet.parameters(), config.train.max_grad_norm
                    )
                optimizer.step()
                optimizer.zero_grad()

        # ---------------------------------------------------------------- #
        # Logging
        # ---------------------------------------------------------------- #
        total_loss_val = total_loss.item()
        reward_mean    = rewards.mean().item()
        reward_std     = rewards.std().item()

        all_losses.append(total_loss_val)
        all_rewards.append(reward_mean)
        all_rewards_std.append(reward_std)

        if accelerator.is_main_process:
            detail_str = "  ".join(
                f"{k}={float(np.mean(v)):.4f}" for k, v in reward_details.items()
            )
            print(
                f"Epoch {epoch:4d} | reward {reward_mean:+.4f} ± {reward_std:.4f} "
                f"| kl {kl_reg.item():.4f} | loss {total_loss_val:.6f}"
                + (f"  [{detail_str}]" if detail_str else "")
            )

            with open(stats_file, "a") as f:
                f.write(
                    f"Epoch {epoch}: loss={total_loss_val:.6f}, "
                    f"reward={reward_mean:.6f}±{reward_std:.6f}, "
                    f"kl={kl_reg.item():.6f}"
                )
                for k, v in reward_details.items():
                    f.write(f", {k}={float(np.mean(v)):.6f}")
                f.write("\n")

        if accelerator.is_main_process:
            log_dict = {
                "reward_mean": reward_mean,
                "reward_std":  reward_std,
                "kl_latent":   kl_reg.item(),
                "loss":        total_loss_val,
            }
            for k, v in reward_details.items():
                log_dict[f"reward/{k}"] = float(np.mean(v))
            accelerator.log(log_dict, step=epoch)

        # ---------------------------------------------------------------- #
        # Visualisation
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
    # Save model & training curves  (main process only)
    # ------------------------------------------------------------------ #
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        student_pipeline.save_pretrained(os.path.join(stats_dir, "student_model"))

        epochs = range(len(all_losses))

        plt.figure()
        plt.plot(epochs, all_losses, label="Total loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training Loss")
        plt.legend()
        plt.savefig(os.path.join(stats_dir, "loss_curve.png"))
        plt.close()

        all_rewards    = np.array(all_rewards)
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