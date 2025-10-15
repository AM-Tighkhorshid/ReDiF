# train_consistency.py
# Consistency Distillation training implementation (based on Consistency Models paper)
# Converted from your train_ppo_coco.py while keeping model loading, LoRA, evaluation, and output structure intact.

from collections import defaultdict
import contextlib
import os
import datetime
import json
import random
import copy
import time
from functools import partial

from absl import app, flags
from ml_collections import config_flags
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from PIL import Image
import tqdm

from ddpo_pytorch.rewards import get_reward_fn
import ddpo_pytorch.prompts
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
logger = get_logger(__name__)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/distill_clip.py", "Training configuration.")

flags.DEFINE_enum("prompt_source", "coco", ["default", "coco"], "Prompt source")
flags.DEFINE_enum("coco_split", "train", ["train", "val", "both"], "COCO caption split")
flags.DEFINE_string("coco_annotations_dir", "coco_dataset/annotations", "COCO annotations directory")


# ------------------------------
# Utility functions
# ------------------------------
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


def compute_pred_original_sample(scheduler, model_output, timestep, sample, detach=False):
    """
    Compute x0 from model output and sample.
    detach=False keeps gradient for student network; True disables grad for teacher/EMA.
    """
    ctx = torch.no_grad() if detach else contextlib.nullcontext()
    with ctx:
        alpha_prod_t = scheduler.alphas_cumprod.gather(0, timestep.cpu()).to(timestep.device)
        alpha_prod_t = alpha_prod_t.reshape(alpha_prod_t.shape + (1,) * (sample.ndim - alpha_prod_t.ndim)).to(sample.device)
        beta_prod_t = 1 - alpha_prod_t

        if scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / (alpha_prod_t ** 0.5)
        elif scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output
        else:
            raise ValueError("Unknown prediction_type")

        if scheduler.config.thresholding:
            pred_original_sample = scheduler._threshold_sample(pred_original_sample)
        elif scheduler.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -scheduler.config.clip_sample_range, scheduler.config.clip_sample_range
            )

    return pred_original_sample


def align_teacher_student_steps(teacher_latents, student_latents, teacher_steps, student_steps):
    if isinstance(teacher_latents, list):
        teacher_latents = torch.stack(teacher_latents, dim=1)
    align_indices = torch.linspace(max(1, teacher_steps // student_steps), teacher_steps, steps=student_steps).long() - 1
    align_indices = torch.clamp(align_indices, 0, teacher_latents.shape[1] - 1)
    aligned_teacher_latents = teacher_latents[:, align_indices]
    return aligned_teacher_latents, student_latents[:, 1:], align_indices


def update_ema(ema_model, model, decay):
    with torch.no_grad():
        ms = dict(model.named_parameters())
        emas = dict(ema_model.named_parameters())
        for name, param in ms.items():
            if name in emas:
                emas[name].data.mul_(decay).add_(param.data, alpha=1.0 - decay)


# ------------------------------
# Main training function
# ------------------------------
def main(_):
    config = FLAGS.config
    config.student.num_steps = 5  # enforce 5 student steps

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    config.run_name = config.run_name or unique_id
    config.run_name += f"_cd_{unique_id}"

    outdir = "CD_run"
    if FLAGS.prompt_source == "coco":
        outdir += "_coco_prompts"

    os.makedirs(outdir, exist_ok=True)
    stats_file = os.path.join(outdir, "training_stats.txt")

    all_losses, all_rewards, all_rewards_std = [], [], []

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
        accelerator.init_trackers("cd-distill", config=config.to_dict())

    # ---------- Load teacher ----------
    teacher_pipeline = StableDiffusionPipeline.from_pretrained(config.pretrained.model, revision=config.pretrained.revision)
    teacher_pipeline.scheduler = DDIMScheduler.from_config(teacher_pipeline.scheduler.config)
    teacher_pipeline.safety_checker = None
    teacher_pipeline.set_progress_bar_config(disable=False)
    teacher_pipeline.vae.to(accelerator.device)
    teacher_pipeline.text_encoder.to(accelerator.device)
    teacher_pipeline.unet.to(accelerator.device)
    teacher_pipeline.vae.requires_grad_(False)
    teacher_pipeline.text_encoder.requires_grad_(False)
    teacher_pipeline.unet.requires_grad_(False)

    # ---------- Load student ----------
    student_pipeline = StableDiffusionPipeline.from_pretrained(config.student.model, revision=config.student.revision)
    student_pipeline.scheduler = DDIMScheduler.from_config(student_pipeline.scheduler.config)
    student_pipeline.safety_checker = None
    student_pipeline.set_progress_bar_config(disable=False)
    student_pipeline.vae.to(accelerator.device)
    student_pipeline.text_encoder.to(accelerator.device)
    student_pipeline.unet.to(accelerator.device)

    if config.use_lora:
        lora_attn_procs = {}
        for name in student_pipeline.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else student_pipeline.unet.config.cross_attention_dim
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

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )
    unet, optimizer = accelerator.prepare(unet, optimizer)

    # EMA
    ema_unet = copy.deepcopy(unet)
    for p in ema_unet.parameters():
        p.requires_grad = False
    ema_decay = getattr(config.train, "ema_decay", 0.9999)

    prompt_fn = getattr(ddpo_pytorch.prompts, config.prompt_fn)

    # COCO captions (optional)
    coco_captions = None
    if FLAGS.prompt_source == "coco":
        ann_dir = FLAGS.coco_annotations_dir
        files = []
        if FLAGS.coco_split in ("train", "both"):
            files.append(os.path.join(ann_dir, "captions_train2017.json"))
        if FLAGS.coco_split in ("val", "both"):
            files.append(os.path.join(ann_dir, "captions_val2017.json"))
        coco_captions = []
        for fp in files:
            if not os.path.exists(fp):
                continue
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            coco_captions += [a["caption"] for a in data["annotations"] if "caption" in a]

    reward_fn = get_reward_fn(["clip", "text_image", "aesthetic"], teacher_pipeline, student_pipeline)

    # ---------------- TRAINING LOOP ----------------
    for epoch in range(config.num_epochs):
        logger.info(f"Epoch {epoch}: Consistency Distillation training...")

        # ----- sample prompts -----
        if FLAGS.prompt_source == "coco" and coco_captions:
            prompts = random.sample(coco_captions, k=config.sample.batch_size)
        else:
            prompts = [p[0] for p in [prompt_fn(**config.prompt_fn_kwargs) for _ in range(config.sample.batch_size)]]

        # encode text
        prompt_ids = student_pipeline.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=student_pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
        prompt_embeds = student_pipeline.text_encoder(prompt_ids)[0]

        # teacher latents
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
            teacher_images, _, teacher_latents_all, _ = teacher_out

        # student latents
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
            student_images, _, student_latents_all, _ = student_out

        student_latents = torch.stack(student_latents_all, dim=1)
        teacher_latents = torch.stack(teacher_latents_all, dim=1)

        teacher_steps = teacher_pipeline.scheduler.num_inference_steps
        student_steps = config.student.num_steps
        aligned_teacher_latents, student_latents_trim, align_indices = align_teacher_student_steps(
            teacher_latents, student_latents, teacher_steps, student_steps
        )

        rewards = torch.tensor(reward_fn(student_images, teacher_images, prompts)[0], device=accelerator.device)
        all_rewards.append(rewards.mean().item())
        if rewards.numel() > 1:
            all_rewards_std.append(rewards.std().item())
        else:
            all_rewards_std.append(0.0)

        # ----- CD training -----
        unet.train()
        total_loss = 0.0
        for j in range(student_steps):
            # Detach per-step tensors to prevent reusing freed graphs
            xtn_plus1 = student_latents_trim[:, j].detach().to(accelerator.device)
            tn1 = student_pipeline.scheduler.timesteps[j].expand(len(prompts)).to(accelerator.device)
            prompt_embeds_detached = prompt_embeds.detach()

            with accelerator.autocast():
                student_out = unet(xtn_plus1, tn1, prompt_embeds_detached).sample
                pred_x0_student = compute_pred_original_sample(
                    student_pipeline.scheduler, student_out, tn1, xtn_plus1, detach=False
                )

            # Teacher backward one-step
            with torch.no_grad():
                t_latent = aligned_teacher_latents[:, j].to(accelerator.device)
                t_timestep = teacher_pipeline.scheduler.timesteps[align_indices[j]].expand(len(prompts)).to(accelerator.device)
                t_out = teacher_pipeline.unet(t_latent, t_timestep, prompt_embeds_detached).sample
                x_hat_phi_tn, _ = ddim_step_with_logprob(
                    teacher_pipeline.scheduler, t_out, t_timestep, t_latent, eta=0.0, prev_sample=None
                )
                ema_unet.eval()
                ema_out = ema_unet(x_hat_phi_tn, tn1, prompt_embeds_detached).sample
                pred_x0_ema = compute_pred_original_sample(
                    student_pipeline.scheduler, ema_out, tn1, x_hat_phi_tn, detach=True
                )

            cd_loss = F.mse_loss(pred_x0_student, pred_x0_ema, reduction="mean")

            with accelerator.accumulate(unet):
                accelerator.backward(cd_loss)
                if accelerator.sync_gradients:
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), config.train.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                update_ema(ema_unet, unet, ema_decay)

            total_loss += cd_loss.detach().cpu().item()


        total_loss /= student_steps
        all_losses.append(total_loss)

        with open(stats_file, "a") as f:
            f.write(f"Epoch {epoch}: Loss={total_loss:.6f}, Reward={rewards.mean().item():.4f}\n")

        print(f"Epoch {epoch}: CD Loss={total_loss:.6f}, Reward mean={rewards.mean().item():.4f}")

        # ----- evaluation -----
        if accelerator.is_main_process:
            eval_prompts = [
                "A cat on a chair",
                "A boy in a forest",
                "A futuristic city skyline at night",
                "A dragon flying over mountains",
                "A cozy cabin in a snowy forest",
            ]
            with torch.no_grad(), accelerator.autocast():
                teacher_eval = [
                    teacher_pipeline(p, num_inference_steps=config.sample.num_steps, guidance_scale=config.sample.guidance_scale).images[0]
                    for p in eval_prompts
                ]
                student_eval = [
                    student_pipeline(p, num_inference_steps=config.student.num_steps, guidance_scale=config.sample.guidance_scale).images[0]
                    for p in eval_prompts
                ]
            save_side_by_side(student_eval, teacher_eval, epoch, outdir)

    # ----- save -----
    student_pipeline.save_pretrained(outdir + "/student_model/")
    epochs = range(len(all_losses))

    plt.figure()
    plt.plot(epochs, all_losses, label="CD Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Consistency Distillation Loss")
    plt.legend()
    plt.savefig(os.path.join(outdir, "loss_curve.png"))
    plt.close()

    all_rewards = np.array(all_rewards)
    all_rewards_std = np.array(all_rewards_std)
    plt.figure()
    plt.plot(epochs, all_rewards, label="Reward mean")
    plt.fill_between(epochs, all_rewards - all_rewards_std, all_rewards + all_rewards_std, alpha=0.3)
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig(os.path.join(outdir, "reward_curve.png"))
    plt.close()


if __name__ == "__main__":
    app.run(main)
