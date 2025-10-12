# train_dmd2_coco.py
# DMD2 distillation trainer adapted from train_ppo_coco.py
# Uses your repo helpers (ddim_step_with_logprob, pipeline_with_logprob, save_side_by_side, etc.)
# Preserves LoRA, Accelerator, COCO prompts, and validation plotting.
# References: uses imports and helpers from train_ppo_coco.py. :contentReference[oaicite:1]{index=1}

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
# NOTE: we removed PPO reward imports per request; keep other repo helpers
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
import copy

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
logger = get_logger(__name__)

# ---------------------------
# Helper: save_side_by_side (copied from your train_ppo_coco.py)
# ---------------------------
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

# ---------------------------
# Flags / config (same as your original file)
# ---------------------------
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

# ---------------------------
# Small discriminator head (latent-level)
# ---------------------------
class DiscriminatorHead(torch.nn.Module):
    def __init__(self, in_channels, hidden=512):
        super().__init__()
        layers = []
        c = in_channels
        for _ in range(3):
            layers.append(torch.nn.Conv2d(c, min(c * 2, hidden), 4, 2, 1))
            layers.append(torch.nn.GroupNorm(32, min(c * 2, hidden)))
            layers.append(torch.nn.SiLU())
            c = min(c * 2, hidden)
        layers.append(torch.nn.Conv2d(c, c, 4, 1))
        self.net = torch.nn.Sequential(*layers)
        self.fc = torch.nn.Linear(c, 1)

    def forward(self, x):
        out = self.net(x)
        out = out.view(out.shape[0], -1)
        return torch.sigmoid(self.fc(out))

# ---------------------------
# Utilities: cosine alpha/sigma & convert predicted noise -> xhat
# ---------------------------
def cosine_alpha_sigma(t):
    # t: tensor in [0,1]
    alpha = torch.cos(0.5 * torch.pi * t)
    sigma = torch.sqrt(torch.clamp(1 - alpha ** 2, min=0.0))
    return alpha, sigma

def noise_pred_to_xhat(x_t, eps, t):
    # x_t: (B,C,H,W), eps predicted noise, t: (B,) in [0,1]
    alpha, sigma = cosine_alpha_sigma(t)
    # expand dims for broadcasting
    a = alpha.view(-1, 1, 1, 1)
    s = sigma.view(-1, 1, 1, 1)
    return (x_t - s * eps) / a

def gan_generator_loss(d_out):
    return -torch.log(d_out.clamp(min=1e-7)).mean()

def gan_discriminator_loss(d_real, d_fake):
    real_loss = -torch.log(d_real.clamp(min=1e-7)).mean()
    fake_loss = -torch.log((1.0 - d_fake).clamp(min=1e-7)).mean()
    return real_loss + fake_loss

# ---------------------------
# run_student_steps: simulate N-step student inference using your ddim_with_logprob helper
# ---------------------------
def run_student_steps(unet_obj, scheduler, x_start, timestep_list, prompt_embeds, eta=0.0):
    x = x_start
    for t_int in timestep_list:
        t_tensor = torch.full((x.size(0),), int(t_int), dtype=torch.long, device=x.device)
        out = unet_obj(x, t_tensor, encoder_hidden_states=prompt_embeds)
        if hasattr(out, "sample"):
            noise_pred = out.sample
        elif isinstance(out, tuple):
            noise_pred = out[0]
        else:
            noise_pred = out
        # Use your helper to keep the exact semantics
        x, _ = ddim_step_with_logprob(scheduler, noise_pred, t_tensor, x, eta=eta, prev_sample=None)
    return x

# ---------------------------
# Main training function
# ---------------------------
def main(_):
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    config.run_name = config.run_name or unique_id
    config.run_name += f"_dmd2_{unique_id}"

    # build outdir similar to original script's convention
    stats_dir = "DMD2_" + config.run_name
    os.makedirs(stats_dir, exist_ok=True)
    stats_file = os.path.join(stats_dir, "training_stats.txt")

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        total_limit=config.num_checkpoint_limit,
    )
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("dmd2-distill", config=config.to_dict())

    logger.info(f"\n{config}")

    # ---------------------------
    # Load teacher & student pipelines using same config fields as train_ppo_coco.py
    # NOTE: teacher loaded from config.pretrained.* to match your original file. :contentReference[oaicite:2]{index=2}
    # ---------------------------
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
    # Keep previous behavior: save teacher snapshot if requested
    if getattr(config, "teacher_output_dir", None):
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

    # ---------------------------
    # LoRA handling (copy of your original code)
    # ---------------------------
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

    # dtype management (same as original)
    dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16

    student_pipeline.vae.to(accelerator.device, dtype=dtype)
    student_pipeline.text_encoder.to(accelerator.device, dtype=dtype)
    if config.use_lora:
        student_pipeline.unet.to(accelerator.device, dtype=dtype)

    # ---------------------------
    # Optimizers (student UNet optimizer)
    # ---------------------------
    optimizer_g = torch.optim.AdamW(
        unet.parameters(),
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # ---------------------------
    # Create mu_fake (fake-score model) & discriminator head
    # ---------------------------
    # Use a deepcopy of the student_pipeline.unet architecture for mu_fake.
    mu_fake_unet = copy.deepcopy(student_pipeline.unet)
    # Discriminator operating on latents; in your repo latents channel = vae latent channels after encode
    latent_channels = student_pipeline.unet.config.in_channels
    disc_head = DiscriminatorHead(in_channels=latent_channels, hidden=512)

    optimizer_fake = torch.optim.AdamW(mu_fake_unet.parameters(), lr=getattr(config.train, "lr_fake", 1e-4), betas=(0.9, 0.95))
    optimizer_disc = torch.optim.AdamW(disc_head.parameters(), lr=getattr(config.train, "lr_disc", 1e-4), betas=(0.5, 0.9))

    # Prepare with accelerator (student unet wrapper, mu_fake, disc, and their optimizers)
    unet, mu_fake_unet, disc_head, optimizer_g, optimizer_fake, optimizer_disc = accelerator.prepare(
        unet, mu_fake_unet, disc_head, optimizer_g, optimizer_fake, optimizer_disc
    )

    # Ensure teacher is on device and in eval
    teacher_pipeline.unet.to(accelerator.device)
    teacher_pipeline.unet.eval()
    for p in teacher_pipeline.unet.parameters():
        p.requires_grad_(False)

    # prompt function
    prompt_fn = getattr(ddpo_pytorch.prompts, config.prompt_fn)

    # load COCO captions if requested (same as original)
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

    # ---------------------------
    # DMD2 hyperparams (paper defaults / your request)
    # ---------------------------
    TTUR_FAKE_UPDATES = getattr(config.train, "ttur_fake_updates", 5)
    DMD_WEIGHT = getattr(config.train, "dmd_weight", 1.0)
    GAN_WEIGHT = getattr(config.train, "gan_weight", 0.1)
    STUDENT_STEPS = getattr(config.student, "num_steps", 5)  # fixed 5 steps as requested
    BATCH_SIZE = config.sample.batch_size

    stats = []

    # Training loop
    for epoch in range(config.num_epochs):
        logger.info(f"Epoch {epoch}: DMD2 distillation")

        # --- sample prompts ---
        if FLAGS.prompt_source == "coco" and coco_captions is not None:
            if len(coco_captions) >= BATCH_SIZE:
                prompts = random.sample(coco_captions, k=BATCH_SIZE)
            else:
                prompts = [random.choice(coco_captions) for _ in range(BATCH_SIZE)]
        else:
            prompt_pairs = [prompt_fn(**config.prompt_fn_kwargs) for _ in range(BATCH_SIZE)]
            prompts = [p[0] for p in prompt_pairs]

        # --- encode text ---
        prompt_ids = student_pipeline.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=student_pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
        prompt_embeds = student_pipeline.text_encoder(prompt_ids)[0]

        # --- use teacher pipeline to produce real images & latents (treat as real) ---
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
            # teacher_latents_all is a list of latents per step; take final decode or use last latent as "real image latents"
            # We will use the final-latents as the 'real' latents for discriminator and forward-diffusion.
            real_latents = teacher_latents_all[-1].to(accelerator.device)

        # --- forward diffusion: sample random continuous t in [0,1) per sample and create x_t ---
        t_cont = torch.rand((real_latents.size(0),), device=real_latents.device)
        alpha, sigma = cosine_alpha_sigma(t_cont)
        eps_noise = torch.randn_like(real_latents)
        x_t = alpha.view(-1, 1, 1, 1) * real_latents + sigma.view(-1, 1, 1, 1) * eps_noise

        # --- teacher score: run teacher_unet on x_t to get predicted noise -> convert to score s_real ---
        with torch.no_grad():
            # convert continuous t to integer timesteps from teacher scheduler
            timesteps_len = len(teacher_pipeline.scheduler.timesteps)
            t_indices = (t_cont * (timesteps_len - 1)).long().tolist()
            t_ints = [int(teacher_pipeline.scheduler.timesteps[i]) for i in t_indices]
            t_tensor = torch.tensor(t_ints, device=x_t.device, dtype=torch.long)
            teacher_out_unet = teacher_pipeline.unet(x_t, t_tensor, encoder_hidden_states=prompt_embeds)
            teacher_noise_pred = teacher_out_unet.sample if hasattr(teacher_out_unet, "sample") else (teacher_out_unet[0] if isinstance(teacher_out_unet, tuple) else teacher_out_unet)
            teacher_xhat = noise_pred_to_xhat(x_t, teacher_noise_pred, t_cont)
            s_real = (alpha.view(-1,1,1,1) * teacher_xhat - x_t) / (sigma.view(-1,1,1,1) ** 2)

        # --- student inference: simulate STUDENT_STEPS from x_t using student's unet and your ddim helper ---
        # Build integer timestep list for the student scheduler (evenly spaced)
        sched_len = len(student_pipeline.scheduler.timesteps)
        idxs = np.linspace(0, sched_len - 1, STUDENT_STEPS, dtype=int).tolist()
        student_timestep_list = [int(student_pipeline.scheduler.timesteps[i]) for i in idxs]

        # run_student_steps expects an object with UNet API; use 'unet' wrapper used above (AttnProcsLayers or plain UNet)
        student_pipeline.unet.train()
        x_fake = run_student_steps(unet, student_pipeline.scheduler, x_t, student_timestep_list, prompt_embeds, eta=config.sample.eta)

        # --- mu_fake: compute fake-score for x_fake (predict noise -> convert to score) ---
        mu_out = mu_fake_unet(x_fake, t_tensor, encoder_hidden_states=prompt_embeds)
        mu_noise_pred = mu_out.sample if hasattr(mu_out, "sample") else (mu_out[0] if isinstance(mu_out, tuple) else mu_out)
        mu_xhat = noise_pred_to_xhat(x_fake, mu_noise_pred, t_cont)
        mu_score = (alpha.view(-1,1,1,1) * mu_xhat - x_fake) / (sigma.view(-1,1,1,1) ** 2)

        # DMD loss: MSE between mu_score and teacher s_real
        dmd_loss = F.mse_loss(mu_score, s_real.detach())

        # --- discriminator: operate on latents (real_latents vs x_fake) ---
        disc_head.train()
        d_real = disc_head(real_latents.detach())
        d_fake = disc_head(x_fake.detach())
        loss_disc = gan_discriminator_loss(d_real, d_fake)

        optimizer_disc.zero_grad()
        accelerator.backward(loss_disc)
        optimizer_disc.step()

        # --- TTUR: update mu_fake several times (fake-score training) ---
        for _ in range(TTUR_FAKE_UPDATES):
            t_p = torch.rand((x_fake.size(0),), device=x_fake.device)
            a_p, s_p = cosine_alpha_sigma(t_p)
            eps_p = torch.randn_like(x_fake)
            x_tp = a_p.view(-1,1,1,1) * x_fake.detach() + s_p.view(-1,1,1,1) * eps_p
            # convert t_p to scheduler integer timesteps for UNet input (approx mapping)
            t_idx_p = (t_p * (sched_len - 1)).long().tolist()
            t_ints_p = [int(student_pipeline.scheduler.timesteps[i]) for i in t_idx_p]
            t_tensor_p = torch.tensor(t_ints_p, device=x_tp.device, dtype=torch.long)
            mu_out_p = mu_fake_unet(x_tp, t_tensor_p, encoder_hidden_states=prompt_embeds)
            eps_hat = mu_out_p.sample if hasattr(mu_out_p, "sample") else (mu_out_p[0] if isinstance(mu_out_p, tuple) else mu_out_p)
            loss_fake_score = F.mse_loss(eps_hat, eps_p.detach())
            optimizer_fake.zero_grad()
            accelerator.backward(loss_fake_score)
            optimizer_fake.step()

        # --- generator (student) update: combine DMD + GAN generator loss ---
        d_fake_for_g = disc_head(x_fake)
        loss_gan_g = gan_generator_loss(d_fake_for_g)
        loss_g = DMD_WEIGHT * dmd_loss + GAN_WEIGHT * loss_gan_g

        optimizer_g.zero_grad()
        accelerator.backward(loss_g)
        optimizer_g.step()

        # Logging & save stats
        step_log = {"epoch": epoch, "dmd_loss": dmd_loss.item(), "disc_loss": loss_disc.item(), "gen_loss": loss_g.item()}
        with open(stats_file, "a") as f:
            f.write(json.dumps(step_log) + "\n")
        print(f"Epoch {epoch}: dmd_loss={dmd_loss.item():.6f}, disc_loss={loss_disc.item():.6f}, gen_loss={loss_g.item():.6f}")

        # --- Validation plotting: generate 5 prompts and save side-by-side as in original ---
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
                            num_inference_steps=STUDENT_STEPS,
                            guidance_scale=config.sample.guidance_scale,
                        ).images[0]
                        for prompt in eval_prompts
                    ]

            save_side_by_side(student_eval_images, teacher_eval_images, epoch, stats_dir)

            # save student checkpoint every epoch (preserve original behavior)
            student_pipeline.save_pretrained(os.path.join(stats_dir, f"student_epoch_{epoch}"))

    # final save
    if accelerator.is_main_process:
        student_pipeline.save_pretrained(os.path.join(stats_dir, "student_final"))
    print("DMD2 distillation finished.")

if __name__ == "__main__":
    app.run(main)
