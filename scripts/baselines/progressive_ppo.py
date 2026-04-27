from collections import defaultdict
import os
import datetime
import json
import random
from absl import app, flags
from ml_collections import config_flags

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger

from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor

import ddpo_pytorch.prompts
from ddpo_pytorch.rewards import get_reward_fn
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob

import tqdm
from functools import partial

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", "config/distill_clip.py", "Training configuration."
)

logger = get_logger(__name__)


# ------------------------------------------------------------
# Divergence (same as PPO code)
# ------------------------------------------------------------
def kl_divergence(p, q):
    p = p.float().view(p.shape[0], -1)
    q = q.float().view(q.shape[0], -1)
    p = F.log_softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    return F.kl_div(p, q, reduction="batchmean", log_target=False)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(_):
    config = FLAGS.config

    outdir = "Progressive_ppo_Distill"
    # if FLAGS.prompt_source == "coco":
    #     outdir = outdir + "_coco_prompts"
    # else:
    #     outdir = outdir + "_ddpo_prompts"

    stats_dir = outdir
    os.makedirs(stats_dir, exist_ok=True)

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    config.run_name = (config.run_name or "ppo_progressive") + f"_{unique_id}"

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=(
            config.train.gradient_accumulation_steps
        ),
        project_config=accelerator_config,
    )

    set_seed(config.seed)

    if accelerator.is_main_process:
        accelerator.init_trackers("ppo-progressive", config=config.to_dict())

    logger.info(config)

    # ------------------------------------------------------------
    # Load initial teacher
    # ------------------------------------------------------------
    teacher_pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained.model, revision=config.pretrained.revision
    )
    teacher_pipeline.scheduler = DDIMScheduler.from_config(
        teacher_pipeline.scheduler.config
    )
    teacher_pipeline.safety_checker = None
    teacher_pipeline.to(accelerator.device)

    teacher_pipeline.unet.requires_grad_(False)
    teacher_pipeline.text_encoder.requires_grad_(False)
    teacher_pipeline.vae.requires_grad_(False)

    # ------------------------------------------------------------
    # Load student
    # ------------------------------------------------------------
    student_pipeline = StableDiffusionPipeline.from_pretrained(
        config.student.model, revision=config.student.revision
    )
    student_pipeline.scheduler = DDIMScheduler.from_config(
        student_pipeline.scheduler.config
    )
    student_pipeline.safety_checker = None
    student_pipeline.to(accelerator.device)

    student_pipeline.vae.requires_grad_(False)
    student_pipeline.text_encoder.requires_grad_(False)
    student_pipeline.unet.requires_grad_(not config.use_lora)

    # LoRA (same as your code)
    if config.use_lora:
        lora_attn_procs = {}
        for name in student_pipeline.unet.attn_processors.keys():
            cross_attention_dim = (
                None if name.endswith("attn1.processor")
                else student_pipeline.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = student_pipeline.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks."):].split(".")[0])
                hidden_size = list(
                    reversed(student_pipeline.unet.config.block_out_channels)
                )[block_id]
            else:
                block_id = int(name[len("down_blocks."):].split(".")[0])
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
    # Prompt + reward
    # ------------------------------------------------------------
    prompt_fn = getattr(ddpo_pytorch.prompts, config.prompt_fn)
    reward_fn = get_reward_fn(["clip"], teacher_pipeline, student_pipeline)

    # ------------------------------------------------------------
    # Progressive settings (JUST LIKE train_progressive)
    # ------------------------------------------------------------
    steps_list = config.distill.steps_list       # e.g. [50, 25, 12, 6]
    epochs_per_stage = config.distill.epochs_per_stage

    # ============================================================
    # Progressive PPO Loop
    # ============================================================
    for stage_idx, student_steps in enumerate(steps_list):

        accelerator.print(
            f"\n==============================\n"
            f" PPO Progressive Stage {stage_idx+1}/{len(steps_list)}\n"
            f" Student steps = {student_steps}\n"
            f"=============================="
        )

        config.student.num_steps = student_steps

        # reset schedulers
        student_pipeline.scheduler = DDIMScheduler.from_config(
            student_pipeline.scheduler.config
        )
        teacher_pipeline.scheduler = DDIMScheduler.from_config(
            teacher_pipeline.scheduler.config
        )

        for epoch in range(epochs_per_stage):

            # -----------------------------
            # sample prompts
            # -----------------------------
            prompt_pairs = [
                prompt_fn(**config.prompt_fn_kwargs)
                for _ in range(config.sample.batch_size)
            ]
            prompts = [p[0] for p in prompt_pairs]

            prompt_ids = student_pipeline.tokenizer(
                prompts,
                padding="max_length",
                truncation=True,
                max_length=student_pipeline.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)

            prompt_embeds = student_pipeline.text_encoder(prompt_ids)[0]

            # -----------------------------
            # Teacher rollout (FULL steps)
            # -----------------------------
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
                teacher_images, _, _, teacher_log_probs = teacher_out

            # -----------------------------
            # Student rollout (FEWER steps)
            # -----------------------------
            student_out = pipeline_with_logprob(
                student_pipeline,
                prompt_embeds=prompt_embeds,
                num_inference_steps=config.student.num_steps,
                guidance_scale=config.sample.guidance_scale,
                eta=config.sample.eta,
                output_type="pt",
                return_all_latents=True,
            )

            student_images, _, student_latents, student_log_probs = student_out

            rewards = torch.tensor(
                reward_fn(student_images, teacher_images, prompts)[0],
                device=accelerator.device,
            )

            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            student_pipeline.unet.train()

            # -----------------------------
            # PPO update (unchanged)
            # -----------------------------
            for j in range(config.student.num_steps):
                with accelerator.accumulate(unet):

                    latents = student_latents[j]
                    timestep = student_pipeline.scheduler.timesteps[j]

                    noise_pred = unet(latents, timestep, prompt_embeds).sample

                    _, log_prob = ddim_step_with_logprob(
                        student_pipeline.scheduler,
                        noise_pred,
                        timestep,
                        latents,
                        eta=config.sample.eta,
                    )

                    ratio = torch.exp(
                        log_prob - student_log_probs[j].detach()
                    )
                    adv = torch.clamp(
                        advantages,
                        -config.train.adv_clip_max,
                        config.train.adv_clip_max,
                    )

                    unclipped = -adv * ratio
                    clipped = -adv * torch.clamp(
                        ratio,
                        1 - config.train.clip_range,
                        1 + config.train.clip_range,
                    )

                    loss = torch.mean(torch.maximum(unclipped, clipped))

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        torch.nn.utils.clip_grad_norm_(
                            unet.parameters(), config.train.max_grad_norm
                        )

                    optimizer.step()
                    optimizer.zero_grad()

        # ========================================================
        # Promote student -> teacher (EXACTLY LIKE train_progressive)
        # ========================================================
        accelerator.wait_for_everyone()

        unwrapped_student = accelerator.unwrap_model(student_pipeline.unet)
        unwrapped_teacher = accelerator.unwrap_model(teacher_pipeline.unet)

        teacher_state = unwrapped_teacher.state_dict()
        student_state = unwrapped_student.state_dict()

        for k in student_state:
            if k in teacher_state and student_state[k].shape == teacher_state[k].shape:
                teacher_state[k].copy_(student_state[k])

        accelerator.print("Student promoted to teacher")

        accelerator.wait_for_everyone()

    accelerator.print("PPO Progressive Distillation Finished")
    final_save_dir = os.path.join(stats_dir, "final_student")
    if accelerator.is_main_process:
        try:
            student_pipeline.save_pretrained(final_save_dir)
            print(f"Saved final student pipeline at {final_save_dir}")
        except Exception as e:
            print(f"Warning: failed to save final student pipeline: {e}")



if __name__ == "__main__":
    app.run(main)