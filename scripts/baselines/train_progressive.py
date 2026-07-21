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
import gc
import ddpo_pytorch.flop_budget


def save_side_by_side(student_images, teacher_images, epoch, outdir):
    os.makedirs(outdir, exist_ok=True)

    # Separate directories for clarity
    student_dir = os.path.join(outdir, "student_only")
    teacher_dir = os.path.join(outdir, "teacher_only")
    combined_dir = os.path.join(outdir, "side_by_side")

    os.makedirs(student_dir, exist_ok=True)
    os.makedirs(teacher_dir, exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)

    for idx in range(min(len(student_images), len(teacher_images))):
        s_img = student_images[idx].convert("RGB")
        t_img = teacher_images[idx].convert("RGB")

        # Resize both to the same square size (keep consistent size)
        h = min(s_img.height, t_img.height)
        s_img = s_img.resize((h, h), Image.Resampling.LANCZOS)
        t_img = t_img.resize((h, h), Image.Resampling.LANCZOS)

        # --- Save teacher-only image ---
        teacher_path = os.path.join(teacher_dir, f"epoch{epoch}_teacher{idx}.png")
        t_img.save(teacher_path)
        print(f"Saved teacher image: {teacher_path}")

        # --- Save student-only image ---
        student_path = os.path.join(student_dir, f"epoch{epoch}_student{idx}.png")
        s_img.save(student_path)
        print(f"Saved student image: {student_path}")

        # --- Create side-by-side combined image ---
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
flags.DEFINE_string(
    "reference_config", "config/distill_clip.py",
    "Config file for scripts/train_ppo_coco.py's run - used ONLY to compute "
    "the FLOP budget this baseline should match, not to change this "
    "baseline's own hyperparameters."
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
    config.run_name += f"_progressive_{unique_id}"

    outdir = "ProgressiveDistill"
    if FLAGS.prompt_source == "coco":
        outdir = outdir + "_coco_prompts"
    else:
        outdir = outdir + "_ddpo_prompts"

    stats_dir = outdir
    os.makedirs(stats_dir, exist_ok=True)
    stats_file = os.path.join(stats_dir, "distill_stats.txt")

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        total_limit=config.num_checkpoint_limit,
    )
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=getattr(config.train, "gradient_accumulation_steps", 1),
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("progressive-distill", config=config.to_dict())

    logger.info(f"\n{config}")

    # --- Load teacher pipeline (will remain frozen) ---
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

    # --- Load student pipeline (we will train its UNet / LoRA) ---
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
    # UNet will be trainable unless using LoRA
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
                block_id = int(name[len("up_blocks."):].split(".")[0])
                hidden_size = list(reversed(student_pipeline.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks."):].split(".")[0])
                hidden_size = student_pipeline.unet.config.block_out_channels[block_id]
            lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
        student_pipeline.unet.set_attn_processor(lora_attn_procs)

        class _Wrapper(AttnProcsLayers):
            def forward(self, *args, **kwargs):
                return student_pipeline.unet(*args, **kwargs)
        # 'unet' is the trainable wrapper containing ONLY LoRA params
        unet = _Wrapper(student_pipeline.unet.attn_processors)
    else:
        # 'unet' is the full student unet
        unet = student_pipeline.unet

    dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16

    student_pipeline.vae.to(accelerator.device, dtype=dtype)
    student_pipeline.text_encoder.to(accelerator.device, dtype=dtype)
    if config.use_lora:
        # Must move base unet to device/dtype, even if frozen
        student_pipeline.unet.to(accelerator.device, dtype=dtype)

    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        unet.parameters(),  # This correctly targets LoRA params if use_lora=True
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


    # Distillation hyperparameters
    steps_list = getattr(config, "distill", None) and getattr(config.distill, "steps_list", None)
    if steps_list is None:
        steps_list = [25, 12, 5]  # default

    batch_size = config.sample.batch_size

    # ------------------------------------------------------------
    # FLOP-matching: derive updates_per_stage from train_ppo_coco.py's
    # compute budget instead of a fixed, hand-picked constant. Per update
    # this loop does: one teacher rollout (config.sample.num_steps forward
    # calls, no_grad), two extra single-step teacher calls (no_grad, used to
    # build the two-step DDIM target x_tilde), and one student
    # forward+backward call.
    # ------------------------------------------------------------
    calib_dtype = next(teacher_pipeline.unet.parameters()).dtype
    cross_attn_dim = teacher_pipeline.unet.config.cross_attention_dim
    teacher_flops = flop_budget.calibrate_unet_call(
        teacher_pipeline.unet, batch_size, cross_attn_dim,
        device=accelerator.device, dtype=calib_dtype, backward=False,
    )
    student_flops = flop_budget.calibrate_unet_call(
        accelerator.unwrap_model(student_unet), batch_size, cross_attn_dim,
        device=accelerator.device, dtype=calib_dtype, backward=True,
    )
    ref_config = flop_budget.load_reference_config(FLAGS.reference_config)
    target_flops = flop_budget.reference_budget_flops(ref_config, teacher_flops, student_flops)
    per_stage_budget = target_flops / len(steps_list)

    updates_per_stage_list = []
    for _student_steps in steps_list:
        flops_per_update = (config.sample.num_steps + 2) * teacher_flops + student_flops
        updates_per_stage_list.append(flop_budget.units_needed(per_stage_budget, flops_per_update))
    accelerator.print(
        f"[flop-match] target = {target_flops:.4e} FLOPs (train_ppo_coco.py's configured run); "
        f"updates per stage: {updates_per_stage_list}"
    )

    # Main progressive distillation loop
    teacher_unet = teacher_pipeline.unet
    # 'student_unet' is the base unet (student_pipeline.unet)
    student_unet = student_pipeline.unet
    teacher_unet.eval()
  
    for stage_idx, student_steps in enumerate(steps_list):
        print(f"\n==== Distillation stage {stage_idx+1}/{len(steps_list)}: student_steps = {student_steps} ====\n")
        
        ### --- FIX BLOCK 1: Student Initialization & LoRA Reset --- ###
        # Initialize student weights from teacher (copy)
        # We copy the teacher's (potentially merged) weights to the student's base UNet
        teacher_state = teacher_unet.state_dict()
        student_unet_state = student_unet.state_dict()
        
        copied = []
        for k in teacher_state:
            if k in student_unet_state and teacher_state[k].size() == student_unet_state[k].size():
                student_unet_state[k].copy_(teacher_state[k].to(student_unet_state[k].device))
                copied.append(k)
        print(f"Copied {len(copied)} matching parameters from teacher -> student base")

        # *** NEW FIX: Reset LoRA weights for the new stage ***
        if config.use_lora:
            # 'unet' is the LoRA wrapper (or the full unet if LoRA is off)
            # We must reset the LoRA parameters at the start of each new stage.
            # We access the parameters via the 'unet' variable prepared by accelerator
            for param in unet.parameters():
                # Re-initialize LoRA weights (zero init is common)
                param.data.zero_()
            print("Reset student LoRA (unet wrapper) parameters for new stage.")
        ### --- END FIX BLOCK 1 --- ###

        # Put student in train mode
        student_unet.train()
        if config.use_lora:
            unet.train() # The wrapper also needs to be in train mode

        # Distillation training loop for this stage
        updates_done = 0
        updates_per_stage = updates_per_stage_list[stage_idx]
        pbar = tqdm(range(updates_per_stage), desc=f"Stage {stage_idx+1} updates", disable=not accelerator.is_main_process)
        while updates_done < updates_per_stage:
            # --- Sample prompts ---
            if FLAGS.prompt_source == "coco" and coco_captions is not None:
                if len(coco_captions) >= batch_size:
                    prompts = random.sample(coco_captions, k=batch_size)
                else:
                    prompts = [random.choice(coco_captions) for _ in range(batch_size)]
                prompt_pairs = [(p, None) for p in prompts]
                prompt_metadata = [None] * len(prompts)
            else:
                prompt_pairs = [prompt_fn(**config.prompt_fn_kwargs) for _ in range(batch_size)]
                prompts = [p[0] for p in prompt_pairs]
                prompt_metadata = [p[1] for p in prompt_pairs]

            # Encode prompts
            prompt_ids = student_pipeline.tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=student_pipeline.tokenizer.model_max_length,
            ).input_ids.to(accelerator.device)
            prompt_embeds = student_pipeline.text_encoder(prompt_ids)[0]

            # --- Generate teacher samples for these prompts (we'll use the final latents as 'clean x') ---
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
                x_latents = teacher_latents_all[-1] 

            # Prepare timesteps: sample discrete i in [1..N] uniformly per sample
            N = int(student_steps)
            i_vals = torch.randint(low=1, high=N + 1, size=(batch_size,), device=accelerator.device)
            t_vals = i_vals.float() / float(N)  # in [1/N, ..., 1]
            pseudo_timestep_scale = 1000 
            timesteps_for_unet = (t_vals * (pseudo_timestep_scale - 1)).long()

            # Vectorize alpha_t and sigma_t for each element
            alpha_t, sigma_t = cosine_alpha_sigma(t_vals)
            alpha_t = alpha_t.view(batch_size, *([1] * (x_latents.dim() - 1)))
            sigma_t = sigma_t.view(batch_size, *([1] * (x_latents.dim() - 1)))

            # Sample noise in latent space
            eps_latent = torch.randn_like(x_latents, device=accelerator.device)
            zt = alpha_t * x_latents + sigma_t * eps_latent

            # --- Get Teacher's two-step target (x_tilde) ---
            with torch.no_grad():
                # Teacher eps_pred at zt
                teacher_eps_pred = teacher_unet(zt, timesteps_for_unet, prompt_embeds).sample
                teacher_x_hat = compute_x_hat_from_eps_pred(zt, teacher_eps_pred, alpha_t, sigma_t)

                # Step 1: compute z_{t'} where t' = t - 0.5 / N
                t_prime_vals = (i_vals.float() - 0.5) / float(N)
                t_prime_vals = torch.clamp(t_prime_vals, min=0.0)
                alpha_t_prime, sigma_t_prime = cosine_alpha_sigma(t_prime_vals)
                alpha_t_prime = alpha_t_prime.view(batch_size, *([1] * (x_latents.dim() - 1)))
                sigma_t_prime = sigma_t_prime.view(batch_size, *([1] * (x_latents.dim() - 1)))

                z_tprime = ddim_one_step_from_x_hat(zt, teacher_x_hat, alpha_t_prime, sigma_t_prime, alpha_t, sigma_t)

                # Teacher eps_pred at z_tprime
                teacher_eps_pred_tprime = teacher_unet(z_tprime, (t_prime_vals * (pseudo_timestep_scale - 1)).long(), prompt_embeds).sample
                teacher_x_hat_tprime = compute_x_hat_from_eps_pred(z_tprime, teacher_eps_pred_tprime, alpha_t_prime, sigma_t_prime)

                # Step 2: compute z_{t''} where t'' = t - 1/N
                t_dprime_vals = (i_vals.float() - 1.0) / float(N)
                t_dprime_vals = torch.clamp(t_dprime_vals, min=0.0)
                alpha_t_dprime, sigma_t_dprime = cosine_alpha_sigma(t_dprime_vals)
                alpha_t_dprime = alpha_t_dprime.view(batch_size, *([1] * (x_latents.dim() - 1)))
                sigma_t_dprime = sigma_t_dprime.view(batch_size, *([1] * (x_latents.dim() - 1)))

                z_tdprime = ddim_one_step_from_x_hat(z_tprime, teacher_x_hat_tprime, alpha_t_dprime, sigma_t_dprime, alpha_t_prime, sigma_t_prime)

                # Compute x_tilde target
                ratio = sigma_t_dprime / (sigma_t + 1e-8)
                numerator = z_tdprime - ratio * zt
                denom = alpha_t_dprime - ratio * alpha_t
                denom_safe = denom.clone()
                denom_safe = torch.where(denom_safe.abs() < 1e-8, torch.sign(denom_safe) * 1e-8 + 1e-8, denom_safe)
                x_tilde = numerator / denom_safe

            # --- Get Student's one-step prediction and compute loss ---
            with accelerator.accumulate(unet):
                # If using LoRA, 'unet' is the wrapper. If not, 'unet' is student_unet.
                # This call works in both cases because the wrapper's forward pass
                # calls student_unet internally.
                noise_pred_student = unet(zt, timesteps_for_unet, prompt_embeds).sample
                x_hat_student = compute_x_hat_from_eps_pred(zt, noise_pred_student, alpha_t, sigma_t)

                # Compute loss in x-space weighted by SNR+1
                weight = snr_plus_one_weight(alpha_t.view(batch_size, *([1] * (x_latents.dim() - 1))),
                                            sigma_t.view(batch_size, *([1] * (x_latents.dim() - 1))))
                weight = weight.to(x_tilde.dtype)

                l2 = (x_tilde - x_hat_student).pow(2)
                loss_per_sample = l2.view(batch_size, -1).mean(dim=1)
                weighted_loss_per_sample = (weight.view(batch_size, -1).mean(dim=1)) * loss_per_sample
                loss = weighted_loss_per_sample.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), config.train.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            updates_done += 1
            pbar.update(1)

        pbar.close()
        print(f"Finished distillation stage for student_steps={student_steps}. Updates done = {updates_done}")

        ### --- FIX BLOCK 2: Student-to-Teacher Promotion with LoRA merge --- ###
        # After finishing training, promote student -> teacher (θ ← η)
        # If using LoRA, we must merge weights *before* copying to the teacher.
        
        # Ensure we're using the unwrapped models for state_dict operations
        # The 'unet' variable from accelerator.prepare might be a wrapper
        unwrapped_student_unet = accelerator.unwrap_model(student_unet)
        unwrapped_teacher_unet = accelerator.unwrap_model(teacher_unet)

        if config.use_lora:
            if accelerator.is_main_process:
                print("Merging LoRA weights into student base for promotion...")
            # We need the pipeline object to access fuse/unfuse
            unwrapped_student_unet.eval()  # Must be in eval mode to fuse
            student_pipeline.fuse_lora()
            # After fuse_lora(), unwrapped_student_unet.state_dict() contains the merged weights

        new_teacher_state = unwrapped_teacher_unet.state_dict()
        student_state = unwrapped_student_unet.state_dict() # This is now the merged state if LoRA
        copied2 = []
        for k in student_state:
            if k in new_teacher_state and student_state[k].size() == new_teacher_state[k].size():
                new_teacher_state[k].copy_(student_state[k].to(new_teacher_state[k].device))
                copied2.append(k)
        
        if config.use_lora:
            if accelerator.is_main_process:
                print("Un-fusing LoRA weights from student...")
            student_pipeline.unfuse_lora() # Revert student to base + LoRA
            unwrapped_student_unet.train() # Put back in train mode

        print(f"Promoted student -> teacher by copying {len(copied2)} params")
        ### --- END FIX BLOCK 2 --- ###

        # --- Save student model checkpoint for this stage ---
        stage_save_dir = os.path.join(stats_dir, f"student_steps_{student_steps}")
        os.makedirs(stage_save_dir, exist_ok=True)
        if accelerator.is_main_process:
            try:
                # Save the pipeline, which includes LoRA weights if used
                student_pipeline.save_pretrained(stage_save_dir)
                print(f"Saved student pipeline at {stage_save_dir}")
            except Exception as e:
                print(f"Warning: failed to save student pipeline: {e}")

        # --- Evaluation imaging ---
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
                    # Teacher uses its full steps
                    teacher_eval_images = [
                        teacher_pipeline(
                            prompt,
                            num_inference_steps=config.sample.num_steps,
                            guidance_scale=config.sample.guidance_scale,
                        ).images[0]
                        for prompt in eval_prompts
                    ]

                    # Student uses its new, smaller step count
                    student_eval_images = [
                        student_pipeline(
                            prompt,
                            num_inference_steps=student_steps,
                            guidance_scale=config.sample.guidance_scale,
                        ).images[0]
                        for prompt in eval_prompts
                    ]

            # Save side-by-side
            save_side_by_side(student_eval_images, teacher_eval_images, student_steps, outdir)

    print("Progressive distillation completed for all stages.")
    # Final save
    final_save_dir = os.path.join(stats_dir, "final_student")
    if accelerator.is_main_process:
        try:
            student_pipeline.save_pretrained(final_save_dir)
            print(f"Saved final student pipeline at {final_save_dir}")
        except Exception as e:
            print(f"Warning: failed to save final student pipeline: {e}")

if __name__ == "__main__":
    app.run(main)