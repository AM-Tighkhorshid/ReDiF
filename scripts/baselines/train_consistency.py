# train_consistency.py
# Consistency Distillation training implementation (based on Consistency Models paper)
# Corrected from the DDPO-hybrid version.

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
import ddpo_pytorch.flop_budget

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
logger = get_logger(__name__)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/distill_clip.py", "Training configuration.")

flags.DEFINE_enum("prompt_source", "coco", ["default", "coco"], "Prompt source")
flags.DEFINE_enum("coco_split", "train", ["train", "val", "both"], "COCO caption split")
flags.DEFINE_string("coco_annotations_dir", "coco_dataset/annotations", "COCO annotations directory")
flags.DEFINE_string(
    "reference_config", "config/distill_clip.py",
    "Config file for scripts/train_ppo_coco.py's run - used ONLY to compute "
    "the FLOP budget this baseline should match, not to change this "
    "baseline's own hyperparameters."
)


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


# -----------------------------------------------
# NEW HELPER: Manual DDIM step
# -----------------------------------------------
def ddim_step_manual(scheduler, model_output, t, t_prev, sample, eta=0.0):
    """
    Performs a manual DDIM step from t to t_prev.
    This is needed for the CD loss, to ensure the teacher's one-step
    jump (t -> t_prev) is calculated correctly.
    """
    with torch.no_grad():
        # Get alpha/beta values for t
        alpha_prod_t = scheduler.alphas_cumprod.gather(0, t.cpu()).to(t.device)
        alpha_prod_t = alpha_prod_t.reshape(alpha_prod_t.shape + (1,) * (sample.ndim - alpha_prod_t.ndim))
        beta_prod_t = 1 - alpha_prod_t

        # Get alpha/beta values for t_prev
        alpha_prod_t_prev = scheduler.alphas_cumprod.gather(0, t_prev.cpu()).to(t.device)
        alpha_prod_t_prev = alpha_prod_t_prev.reshape(alpha_prod_t_prev.shape + (1,) * (sample.ndim - alpha_prod_t_prev.ndim))
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 1. Compute "predicted x_0"
        #    (Same as compute_pred_original_sample)
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / (alpha_prod_t ** 0.5)

        # 2. Compute "direction pointing to x_t" (epsilon)
        pred_epsilon = (sample - alpha_prod_t ** 0.5 * pred_original_sample) / (beta_prod_t ** 0.5)

        # 3. Compute variance (sigma_t)
        #    We use eta=0 for deterministic distillation
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        std_dev_t = eta * variance ** 0.5

        # 4. Compute x_t_prev (x_{t_n})
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** 0.5 * pred_epsilon
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction

        # Note: eta=0, so we don't add noise
        # if eta > 0:
        #     noise = torch.randn_like(model_output)
        #     prev_sample = prev_sample + std_dev_t * noise

        return prev_sample


def align_teacher_student_steps(teacher_latents, student_latents, teacher_steps, student_steps):
    # This function is no longer used by the CD loss, but kept for compatibility
    # if other parts of the original code (e.g. logging) still use it.
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
    # NOTE: config.student.num_steps is now only used for EVALUATION
    # The training process distills the 1-step model
    
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
        # Gradient accumulation is handled per-step in the new loop
        gradient_accumulation_steps=1,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers("cd-distill", config=config.to_dict())

    # ---------- Load teacher ----------
    teacher_pipeline = StableDiffusionPipeline.from_pretrained(config.pretrained.model, revision=config.pretrained.revision)
    teacher_pipeline.scheduler = DDIMScheduler.from_config(teacher_pipeline.scheduler.config)
    teacher_pipeline.safety_checker = None
    teacher_pipeline.set_progress_bar_config(disable=True) # Disable for training
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

    # Freeze VAE and Text Encoder for student (we only train UNet)
    student_pipeline.vae.requires_grad_(False)
    student_pipeline.text_encoder.requires_grad_(False)

    if config.use_lora:
        lora_attn_procs = {}
        for name in student_pipeline.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else student_pipeline.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = student_pipeline.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                # --- FIX: name[len("up_blocks.")] indexes a single character
                # at a fixed offset, which only happened to equal the block
                # digit because SD1.5 has < 10 blocks per side. Parse the
                # actual "up_blocks.N...." segment instead, matching the
                # robust version used in the other baselines. ---
                block_id = int(name[len("up_blocks."):].split(".")[0])
                hidden_size = list(reversed(student_pipeline.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks."):].split(".")[0])
                hidden_size = student_pipeline.unet.config.block_out_channels[block_id]
            lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
        student_pipeline.unet.set_attn_processor(lora_attn_procs)
        
        # This wrapper and unet preparation is from the original DDPO code
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

    # ------------------------------------------------------------
    # EMA target network
    # ------------------------------------------------------------
    # --- FIX (critical): `copy.deepcopy(unet)` when use_lora=True deep-copies
    # the `_Wrapper(AttnProcsLayers)` object, which DOES get its own
    # independent LoRA parameter tensors - but `_Wrapper.forward()` is
    # hard-coded to call `student_pipeline.unet(*args, **kwargs)` via closure,
    # completely ignoring `self`. That means calling the deep-copied
    # `ema_unet(...)` executes the SAME live, currently-training student
    # weights as calling `unet(...)` - `update_ema()` was faithfully
    # maintaining an EMA buffer, but that buffer was never actually consulted
    # to compute `ema_out`, silently defeating the entire point of using an
    # EMA target for consistency-distillation stability. Fixed by cloning the
    # actual underlying UNet2DConditionModel (base weights + LoRA attention
    # processors together, since that is what the forward pass really
    # computes) into a standalone module whose OWN forward is called. ---
    ema_full_unet = copy.deepcopy(accelerator.unwrap_model(student_pipeline.unet))
    for p in ema_full_unet.parameters():
        p.requires_grad_(False)
    ema_full_unet.to(accelerator.device)
    ema_decay = getattr(config.train, "ema_decay", 0.9999)

    def update_ema_full(ema_model, source_unet, decay):
        with torch.no_grad():
            src = dict(source_unet.named_parameters())
            for name, ema_p in ema_model.named_parameters():
                if name in src:
                    ema_p.data.mul_(decay).add_(src[name].data.to(ema_p.device), alpha=1.0 - decay)

    # ------------------------------------------------------------
    # FLOP-matching: run as many epochs as it takes to spend the same total
    # FLOPs as train_ppo_coco.py's configured run, instead of the fixed
    # config.num_epochs. Per epoch this loop does: one teacher rollout
    # (sample.num_steps forward calls, no_grad), one student "eval" rollout
    # for logging only (student.num_steps forward calls, no_grad), and
    # train.gradient_accumulation_steps inner iterations each doing one
    # student forward+backward call plus two forward-only teacher-shaped
    # calls (the single-step teacher call and the EMA-network call).
    # ------------------------------------------------------------
    calib_dtype = next(teacher_pipeline.unet.parameters()).dtype
    cross_attn_dim = teacher_pipeline.unet.config.cross_attention_dim
    teacher_flops = flop_budget.calibrate_unet_call(
        teacher_pipeline.unet, config.sample.batch_size, cross_attn_dim,
        device=accelerator.device, dtype=calib_dtype, backward=False,
    )
    student_flops = flop_budget.calibrate_unet_call(
        accelerator.unwrap_model(student_pipeline.unet), config.sample.batch_size, cross_attn_dim,
        device=accelerator.device, dtype=calib_dtype, backward=True,
    )
    ref_config = flop_budget.load_reference_config(FLAGS.reference_config)
    target_flops = flop_budget.reference_budget_flops(ref_config, teacher_flops, student_flops)
    flops_per_epoch = (
        config.sample.num_steps * teacher_flops
        + config.student.num_steps * teacher_flops
        + config.train.gradient_accumulation_steps * (student_flops + 2 * teacher_flops)
    )
    matched_num_epochs = flop_budget.units_needed(target_flops, flops_per_epoch)
    flop_budget.report(accelerator.print, target_flops, flops_per_epoch, "epoch", matched_num_epochs)
    config.num_epochs = matched_num_epochs

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

    # Reward function (for logging only)
    reward_fn = get_reward_fn(["clip", "text_image", "aesthetic"], teacher_pipeline, student_pipeline)

    # Get the 1000-step timesteps from the scheduler
    # This is crucial for sampling t_n and t_{n+1}
    student_pipeline.scheduler.set_timesteps(student_pipeline.scheduler.config.num_train_timesteps)
    all_timesteps = student_pipeline.scheduler.timesteps

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
        prompt_embeds = prompt_embeds.detach() # Detach, we don't train text encoder

        # ----- Generate "Ground Truth" $x_0$ with Teacher -----
        # We run the full teacher pipeline to get its $x_0$ latent prediction.
        # This serves as the "clean" data to which we'll add noise for training.
        with torch.no_grad():
            teacher_out = pipeline_with_logprob(
                teacher_pipeline,
                prompt_embeds=prompt_embeds,
                num_inference_steps=config.sample.num_steps,
                guidance_scale=config.sample.guidance_scale,
                eta=config.sample.eta,
                output_type="pt",
                return_all_latents=True, # We need the latents
            )
            teacher_images, _, teacher_latents_all, _ = teacher_out

            # The final latent in the list is the predicted x0
            # Stack them: [B, N_steps+1, C, H, W]
            teacher_latents_stack = torch.stack(teacher_latents_all, dim=1)
            # Get the last latent (x0) from each batch item
            x0_latents = teacher_latents_stack[:, -1].detach() # [B, C, H, W]

        # ----- Student Evaluation Run (for logging/rewards ONLY) -----
        # This is kept from your original code to maintain the "fair setting"
        # for logging rewards and std. It is NOT used in the CD loss.
        student_pipeline.unet.eval()
        with torch.no_grad(), accelerator.autocast():
            student_out = pipeline_with_logprob(
                student_pipeline,
                prompt_embeds=prompt_embeds,
                num_inference_steps=config.student.num_steps, # Use 5-step for eval
                guidance_scale=config.sample.guidance_scale,
                eta=config.sample.eta,
                output_type="pt",
                return_all_latents=False, # Don't need student latents
            )
            student_images, _, _, _ = student_out

        # Log rewards (as in your original code)
        rewards = torch.tensor(reward_fn(student_images, teacher_images, prompts)[0], device=accelerator.device)
        all_rewards.append(rewards.mean().item())
        if rewards.numel() > 1:
            all_rewards_std.append(rewards.std().item())
        else:
            all_rewards_std.append(0.0)

        # -----------------------------------------------
        # ----- CD TRAINING LOOP REWRITE (START) -----
        # -----------------------------------------------
        # This is the correct Consistency Distillation training logic.
        # We replace the `for j in range(student_steps)` loop.
        unet.train()
        total_loss = 0.0
        
        # We run `gradient_accumulation_steps` batches per epoch
        for _ in tqdm(range(config.train.gradient_accumulation_steps), desc="CD Steps"):
            with accelerator.accumulate(unet):
                # 1. Sample random timesteps t_n+1 and t_n
                #    We sample from the *full* 1000-step schedule
                N = len(all_timesteps)

                # Sample indices from [0, N-2]
                # --- FIX: Create indices on CPU, because all_timesteps is on CPU ---
                indices = torch.randint(
                    0, N - 1, (config.sample.batch_size,),
                    device="cpu"  # Was accelerator.device
                )

                # t_{n+1} (e.g., timestep 500)
                # Now we index (cpu tensor by cpu tensor) and move the result to the device
                t_n_plus_1 = all_timesteps[indices].to(accelerator.device)
                
                # t_n (e.g., timestep 499)
                t_n = all_timesteps[indices + 1].to(accelerator.device)

                # 2. Get x_{t_{n+1}} by noising our ground-truth x0_latents
                noise = torch.randn_like(x0_latents)
                xt_n_plus_1 = student_pipeline.scheduler.add_noise(
                    x0_latents, noise, t_n_plus_1
                )
                xt_n_plus_1 = xt_n_plus_1.to(prompt_embeds.dtype)

                # 3. Get Student ("online") prediction of x0
                with accelerator.autocast():
                    student_out = unet(xt_n_plus_1, t_n_plus_1, prompt_embeds).sample
                    pred_x0_student = compute_pred_original_sample(
                        student_pipeline.scheduler, student_out, t_n_plus_1, xt_n_plus_1, 
                        detach=False # Keep gradient
                    )

                # 4. Get Target ("EMA") prediction of x0
                with torch.no_grad():
                    # 4a. Get teacher's model_output at t_{n+1}
                    teacher_model_out = teacher_pipeline.unet(
                        xt_n_plus_1, t_n_plus_1, prompt_embeds
                    ).sample

                    # 4b. Perform one DDIM step (t_{n+1} -> t_n) using teacher
                    #     This is the "distillation" step.
                    x_t_n = ddim_step_manual(
                        teacher_pipeline.scheduler,
                        teacher_model_out,
                        t_n_plus_1, # current time
                        t_n,        # previous time
                        xt_n_plus_1, # current sample
                        eta=0.0     # Must be deterministic (eta=0)
                    )
                    
                    # 4c. Get EMA model's prediction from x_{t_n} at time t_n
                    ema_out = ema_full_unet(x_t_n.to(prompt_embeds.dtype), t_n, encoder_hidden_states=prompt_embeds).sample
                    pred_x0_ema = compute_pred_original_sample(
                        student_pipeline.scheduler, ema_out, t_n, x_t_n, 
                        detach=True # No gradient for target
                    )

                # 5. Calculate CD Loss
                #    (Using MSE as in your original code, though L1 is also common)
                cd_loss = F.mse_loss(pred_x0_student, pred_x0_ema, reduction="mean")

                # 6. Backward pass
                accelerator.backward(cd_loss)
                
                if accelerator.sync_gradients:
                    torch.nn.utils.clip_grad_norm_(
                        unet.parameters(), config.train.max_grad_norm
                    )
                
                optimizer.step()
                optimizer.zero_grad()
                
                # 7. Update EMA model
                update_ema_full(ema_full_unet, accelerator.unwrap_model(student_pipeline.unet), ema_decay)

                total_loss += cd_loss.detach().cpu().item()

        # -----------------------------------------------
        # ----- CD TRAINING LOOP REWRITE (END) -----
        # -----------------------------------------------

        avg_loss = total_loss / config.train.gradient_accumulation_steps
        all_losses.append(avg_loss)

        with open(stats_file, "a") as f:
            f.write(f"Epoch {epoch}: Loss={avg_loss:.6f}, Reward={rewards.mean().item():.4f}\n")

        print(f"Epoch {epoch}: CD Loss={avg_loss:.6f}, Reward mean={rewards.mean().item():.4f}")

        # ----- evaluation -----
        # This part is unchanged and correctly evaluates the student
        # using the 5-step sampler.
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
            with torch.no_grad(), accelerator.autocast():
                # Teacher (long steps)
                teacher_eval = [
                    teacher_pipeline(p, num_inference_steps=config.sample.num_steps, guidance_scale=config.sample.guidance_scale).images[0]
                    for p in eval_prompts
                ]
                # Student (short steps)
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