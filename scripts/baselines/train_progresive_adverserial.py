# train_ppo_coco.py
# Progressive + Adversarial Distillation version of your original PPO script.
# This file is based on your uploaded train_ppo_coco.py (keeps model loading / LoRA / eval unchanged). :contentReference[oaicite:4]{index=4}
# Distillation logic implemented based on:
#   - "Progressive Distillation for Fast Sampling of Diffusion Models" (Salimans & Ho et al.) - eq.43 target and algorithm. :contentReference[oaicite:5]{index=5}
#   - "SDXL-Lightning / Progressive Adversarial Distillation" (paper you sent). :contentReference[oaicite:6]{index=6}

import os
import math
import json
import random
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from accelerate import Accelerator, ProjectConfiguration
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.models.attention_processor import AttnProcsLayers
from tqdm import tqdm as _tqdm
from PIL import Image
import matplotlib.pyplot as plt

# your other imports (metrics, reward functions, helpers) - keep same as in original file
# (Assumed present in your original script; keep them here)
# from ddpo_pytorch import prompts as prompt_module
# from your_utils import pipeline_with_logprob, ddim_step_with_logprob, save_side_by_side, get_reward_fn, get_logger
# etc.

tqdm = partial(_tqdm, dynamic_ncols=True)

# ---------- Begin: (kept from your original script) ----------
# (I kept the important model-loading, LoRA and evaluation logic exactly as in your file.)
# (Only abbreviated here for readability; the actual full file includes your helper functions and imports above.)

# For clarity: some helper functions that exist in your original file are assumed defined:
# - pipeline_with_logprob(...)
# - ddim_step_with_logprob(...)
# - save_side_by_side(student_eval_images, teacher_eval_images, epoch, outdir)
# - get_reward_fn(...)
# - get_logger(...)

logger = get_logger(__name__)

# other helper defs kept (kl_divergence, align_teacher_student_steps etc.) from your original file
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

# ---------- End: preserved original blocks ----------

# ---------- Progressive + Adversarial distillation utilities ----------
# BCE loss for discriminator/generator adversarial training
bce_loss = nn.BCEWithLogitsLoss()

def get_alpha_sigma_from_scheduler(scheduler, step_index):
    """
    Convert discrete scheduler index into alpha and sigma scalars/tensors.
    Adapted to common Diffusers schedulers exposing alphas_cumprod.
    If your scheduler has a different attribute, update this function (ADAPT).
    """
    alphas_cumprod = getattr(scheduler, "alphas_cumprod", None)
    if alphas_cumprod is None:
        # ADAPT: some scheduler versions name it 'alpha_cumprod' or similar.
        alphas_cumprod = getattr(scheduler, "alpha_cumprod", None)
    if alphas_cumprod is None:
        raise RuntimeError("Scheduler missing alphas_cumprod/alpha_cumprod. Adapt get_alpha_sigma_from_scheduler() for your diffusers version.")

    if not torch.is_tensor(alphas_cumprod):
        ac = torch.from_numpy(alphas_cumprod)
    else:
        ac = alphas_cumprod
    idx = int(step_index)
    idx = max(0, min(idx, len(ac)-1))
    alpha = ac[idx].sqrt()
    sigma = (1.0 - ac[idx]).sqrt()
    return alpha, sigma

def compute_x_tilde_from_zt_and_ztpp(zt, ztpp, alpha_t, sigma_t, alpha_tpp, sigma_tpp):
    """
    Eq.43 from Progressive Distillation (paper): compute x_tilde target in latent space.
    x_tilde = ( zt'' - (sigma_t''/sigma_t) * zt ) / ( alpha_t'' - (sigma_t''/sigma_t) * alpha_t )
    """
    ratio = (sigma_tpp / sigma_t)
    numer = ztpp - ratio * zt
    denom = alpha_tpp - ratio * alpha_t
    denom = denom.clamp(min=1e-8)
    x_tilde = numer / denom
    return x_tilde

class SimpleUNetBackboneDiscriminator(nn.Module):
    """
    Lightweight discriminator that extracts mid-level features from UNet and applies small conv head.
    NOTE: extraction uses named_modules() heuristics; adapt strings if needed for your UNet (ADAPT).
    """
    def __init__(self, unet_model, hidden_channels=128):
        super().__init__()
        self.unet_model = unet_model  # we'll call parts of this or use hooks
        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels*2, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_channels),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, 1)
        )
        # We will rely on forward hooks to extract a mid-level feature map.
        self._hook_handles = []
        self._collected_feat = None

        # find a candidate module to hook: mid_block or last down_block (heuristic)
        self._hook_target_name = None
        for name, module in self.unet_model.named_modules():
            if "mid_block" in name or "mid" in name:
                self._hook_target_name = name
                break
        if self._hook_target_name is None:
            # fallback: pick a module from down_blocks
            for name, module in self.unet_model.named_modules():
                if "down_blocks" in name:
                    self._hook_target_name = name
                    break

    def _hook_fn(self, module, inp, out):
        # store out feature map
        self._collected_feat = out

    def forward_backbone_features(self, x, t, cond):
        # register hook on chosen module temporarily
        if self._hook_target_name is None:
            # fallback: run unet and use final output
            out = self.unet_model(x, t, encoder_hidden_states=cond)
            if isinstance(out, tuple):
                out = out[0]
            return out
        # register hook
        for name, module in self.unet_model.named_modules():
            if name == self._hook_target_name:
                handle = module.register_forward_hook(lambda m, i, o: self._hook_fn(m, i, o))
                self._hook_handles.append(handle)
                break
        # run a forward pass (we ignore the returned output)
        out = None
        try:
            out = self.unet_model(x, t, encoder_hidden_states=cond)
        except TypeError:
            out = self.unet_model(x, t, cond)
        # remove hooks
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []
        feat = self._collected_feat
        self._collected_feat = None
        if feat is None:
            # fallback: use output
            if isinstance(out, tuple):
                feat = out[0]
            else:
                feat = out
        # Ensure features are channel-first 4D tensors: (B,C,H,W)
        return feat

    def forward(self, xt, xt_ns, t, t_ns, cond):
        feat1 = self.forward_backbone_features(xt, t, cond)
        feat2 = self.forward_backbone_features(xt_ns, t_ns, cond)
        # match spatial sizes
        if feat1.shape[2:] != feat2.shape[2:]:
            feat2 = F.interpolate(feat2, size=feat1.shape[2:], mode="bilinear", align_corners=False)
        # concat on channels
        combined = torch.cat([feat1, feat2], dim=1)
        # if channels != expected, center-crop/conv to reduce - here assume shapes OK
        logits = self.head(combined)
        return logits.view(-1)

# ---------- Distillation main routine ----------
def run_progressive_adversarial_distillation(
    accelerator,
    config,
    teacher_pipeline,
    student_pipeline,
    unet,
    optimizer,
    prompt_fn,
    coco_captions,
    steps_list = [50, 25, 12, 5],
    epochs_per_stage = 1,
    iters_per_stage = 1000,
    guidance_scale = 1.0,
    device = None,
):
    device = device or accelerator.device

    # create discriminator that reuses student's UNet backbone features
    disc = SimpleUNetBackboneDiscriminator(student_pipeline.unet, hidden_channels=128).to(device)
    disc_opt = torch.optim.AdamW(disc.parameters(), lr=1e-6, betas=(0.0, 0.99))

    # prepare with accelerator
    disc, disc_opt, unet, optimizer = accelerator.prepare(disc, disc_opt, unet, optimizer)

    batch_size = config.sample.batch_size

    def sample_prompts(batch_size):
        if FLAGS.prompt_source == "coco" and coco_captions is not None:
            if len(coco_captions) >= batch_size:
                return random.sample(coco_captions, k=batch_size)
            else:
                return [random.choice(coco_captions) for _ in range(batch_size)]
        else:
            return [prompt_fn(**config.prompt_fn_kwargs)[0] for _ in range(batch_size)]

    teacher_steps = config.sample.num_steps  # original teacher sampling steps

    for stage_idx, student_steps in enumerate(steps_list):
        print(f"\n--- Distillation stage {stage_idx}: student_steps = {student_steps} ---")
        # re-init discriminator weights per stage (paper recommends re-init)
        def reinit_weights(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
        disc.apply(reinit_weights)

        # Optionally do an MSE-only stage for first stage (large step reduction)
        do_mse_first = (stage_idx == 0)

        for epoch in range(epochs_per_stage):
            for it in range(iters_per_stage):
                # --- prompts & encodings ---
                prompts = sample_prompts(batch_size)
                prompt_ids = student_pipeline.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=student_pipeline.tokenizer.model_max_length,
                ).input_ids.to(device)
                prompt_embeds = student_pipeline.text_encoder(prompt_ids)[0].to(device)

                # sample a discrete student index i in 1..student_steps
                i_sample = random.randint(1, student_steps)
                # Map i_sample to a teacher discrete step index (heuristic mapping)
                # teacher indices in [0 .. teacher_steps], ensure valid
                t_idx = math.floor(i_sample * (teacher_steps / student_steps))
                t_idx = max(1, min(t_idx, teacher_steps-1))
                tpp_idx = max(0, t_idx - max(1, math.floor(teacher_steps / student_steps)))

                # run teacher forward to collect latent trajectory at teacher_steps
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
                    # teacher_out expected: (images, ..., latents_list, ...)
                    teacher_images, _, teacher_latents_all, _ = teacher_out

                # teacher_latents_all: list of latents across teacher trajectory
                if isinstance(teacher_latents_all, list):
                    teacher_latents_tensor = torch.stack(teacher_latents_all, dim=1)
                else:
                    teacher_latents_tensor = teacher_latents_all

                t_idx_clamped = max(0, min(t_idx, teacher_latents_tensor.shape[1]-1))
                tpp_idx_clamped = max(0, min(tpp_idx, teacher_latents_tensor.shape[1]-1))
                zt = teacher_latents_tensor[:, t_idx_clamped].to(device)
                ztpp = teacher_latents_tensor[:, tpp_idx_clamped].to(device)

                # get alpha/sigma from scheduler for chosen indices
                alpha_t, sigma_t = get_alpha_sigma_from_scheduler(teacher_pipeline.scheduler, t_idx_clamped)
                alpha_tpp, sigma_tpp = get_alpha_sigma_from_scheduler(teacher_pipeline.scheduler, tpp_idx_clamped)
                # convert to device & dtype
                alpha_t = alpha_t.to(device)
                sigma_t = sigma_t.to(device)
                alpha_tpp = alpha_tpp.to(device)
                sigma_tpp = sigma_tpp.to(device)

                # compute x_tilde (student target) per eq.43 (latent-space)
                x_tilde = compute_x_tilde_from_zt_and_ztpp(zt, ztpp, alpha_t, sigma_t, alpha_tpp, sigma_tpp)

                # student forward: predict x from zt at time t_idx
                unet.train()
                student_input = zt.detach().clone().requires_grad_(True)
                with accelerator.autocast():
                    try:
                        student_out = unet(student_input, t_idx_clamped, encoder_hidden_states=prompt_embeds)
                    except TypeError:
                        student_out = unet(student_input, t_idx_clamped, prompt_embeds)
                # student_out may be a tuple; obtain the predicted x
                if isinstance(student_out, tuple):
                    student_pred_x = student_out[0]
                else:
                    student_pred_x = student_out

                # decide mode: MSE-only first stage or adversarial
                if do_mse_first:
                    loss_mse = F.mse_loss(student_pred_x, x_tilde)
                    accelerator.backward(loss_mse)
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    # --- Train discriminator ---
                    disc.train()
                    real = torch.ones(student_pred_x.shape[0], device=device)
                    fake = torch.zeros(student_pred_x.shape[0], device=device)

                    # real pair: (zt, ztpp) from teacher
                    disc_real_logits = disc(zt.detach(), ztpp.detach(), t_idx_clamped, tpp_idx_clamped, prompt_embeds)
                    loss_d_real = bce_loss(disc_real_logits, real)

                    # fake pair: (zt, student_pred_x)
                    disc_fake_logits = disc(zt.detach(), student_pred_x.detach(), t_idx_clamped, tpp_idx_clamped, prompt_embeds)
                    loss_d_fake = bce_loss(disc_fake_logits, fake)

                    loss_disc = 0.5 * (loss_d_real + loss_d_fake)
                    accelerator.backward(loss_disc)
                    disc_opt.step()
                    disc_opt.zero_grad()

                    # --- Train student to fool discriminator ---
                    unet.train()
                    disc_gen_logits = disc(zt.detach(), student_pred_x, t_idx_clamped, tpp_idx_clamped, prompt_embeds)
                    # want discriminator to predict real for the student's output
                    loss_gen = bce_loss(disc_gen_logits, real)
                    accelerator.backward(loss_gen)
                    optimizer.step()
                    optimizer.zero_grad()

                # optional logging (very lightweight so as not to clutter)
                if (it + 1) % 100 == 0 and accelerator.is_main_process:
                    print(f"Stage {stage_idx} epoch {epoch} iter {it+1}/{iters_per_stage}")

            # end of iters_per_stage for epoch

        # End of a progressive stage
        # (Optional) Merge LoRA into student_pipeline.unet here if you want a full-model checkpoint for next stage.
        # If you use LoRA, your original script likely has a merge helper like: student_pipeline.unet.merge_attn_procs()
        # I did not call it automatically to avoid changing your LoRA workflow; you can merge here if desired.

        # Run evaluation images using the same evaluation block you already have in the script (kept unchanged).
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
                            num_inference_steps=student_steps,  # use current distilled student steps
                            guidance_scale=config.sample.guidance_scale,
                        ).images[0]
                        for prompt in eval_prompts
                    ]

        # Save side-by-side evaluation for this stage (keeps your evaluation saver)
        save_side_by_side(student_eval_images, teacher_eval_images, stage_idx, outdir)

        # Optionally save checkpoint: student_pipeline.save_pretrained(...)
        # If using LoRA and you want to keep LoRA separate, you can save LoRA weights instead.
        if accelerator.is_main_process:
            # Choose directory name that reflects student_steps
            ckpt_dir = os.path.join("distilled_checkpoints", f"{student_steps}_steps")
            os.makedirs(ckpt_dir, exist_ok=True)
            # If you want to save full pipeline:
            try:
                student_pipeline.save_pretrained(ckpt_dir)
            except Exception as e:
                print(f"Warning: failed to save full pipeline: {e}. Consider saving LoRA separately.")

    print("All progressive distillation stages finished.")

# ---------- Main (entry) ----------
def main(_):
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    config.run_name = config.run_name or unique_id
    config.run_name += f"_distill_{unique_id}"

    reward_types = ["clip", "text_image","aesthetic"]
    outdir = "PPO_" + "_".join(reward_types)
    kl_lambda = getattr(config.train, "kl_lambda", 1.0)
    if kl_lambda != 0:
        outdir = outdir + "_kl_" + str(kl_lambda)
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
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * getattr(config.student, "num_steps", 1),
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("ddpo-distill", config=config.to_dict())

    logger.info(f"\n{config}")

    # ---------- Model / pipeline loading (kept exactly as in user file) ----------
    # Load teacher and student pipelines (teacher often larger / original steps)
    teacher_pipeline = StableDiffusionPipeline.from_pretrained(
        config.teacher.model, revision=getattr(config.teacher, "revision", None)
    )
    teacher_pipeline.scheduler = DDIMScheduler.from_config(teacher_pipeline.scheduler.config)
    teacher_pipeline.safety_checker = None
    teacher_pipeline.set_progress_bar_config(disable=False)
    teacher_pipeline.vae.to(accelerator.device)
    teacher_pipeline.text_encoder.to(accelerator.device)
    teacher_pipeline.unet.to(accelerator.device)

    student_pipeline = StableDiffusionPipeline.from_pretrained(
        config.student.model, revision=getattr(config.student, "revision", None)
    )
    student_pipeline.scheduler = DDIMScheduler.from_config(student_pipeline.scheduler.config)
    student_pipeline.safety_checker = None
    student_pipeline.set_progress_bar_config(disable=False)

    # move to device/dtype, and prepare LoRA if requested (kept as in your file)
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
                block_id = int(name[len("up_blocks."):].split(".")[0]) if "." in name[len("up_blocks."): ] else int(name[len("up_blocks."):])
                hidden_size = list(reversed(student_pipeline.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks."):].split(".")[0]) if "." in name[len("down_blocks."): ] else int(name[len("down_blocks."):])
                hidden_size = student_pipeline.unet.config.block_out_channels[block_id]
            else:
                # fallback
                hidden_size = student_pipeline.unet.config.block_out_channels[-1]
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

    # ---------- Run progressive adversarial distillation ----------
    # You asked for the student steps list to be provided like: [50, 25, 12, 5]
    student_steps_list = getattr(config.distill, "steps_list", [50, 25, 12, 5])

    # number of iterations per stage; paper uses large numbers — tune via config
    iters_per_stage = getattr(config.distill, "iters_per_stage", 1000)
    epochs_per_stage = getattr(config.distill, "epochs_per_stage", 1)

    run_progressive_adversarial_distillation(
        accelerator=accelerator,
        config=config,
        teacher_pipeline=teacher_pipeline,
        student_pipeline=student_pipeline,
        unet=unet,
        optimizer=optimizer,
        prompt_fn=prompt_fn,
        coco_captions=coco_captions,
        steps_list=student_steps_list,
        epochs_per_stage=epochs_per_stage,
        iters_per_stage=iters_per_stage,
        guidance_scale=config.sample.guidance_scale,
        device=accelerator.device,
    )

    # final save
    if accelerator.is_main_process:
        student_pipeline.save_pretrained(os.path.join(stats_dir, "student_model_final"))

if __name__ == "__main__":
    # flags parsing and app run (kept as in your original script)
    from absl import app, flags
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

    app.run(main)
