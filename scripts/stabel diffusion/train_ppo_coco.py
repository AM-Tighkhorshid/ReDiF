"""
Teacher (config.sample.num_steps-step DDIM) -> Student (config.student.num_steps-step
DDIM) distillation via PPO.

Reward : composite reward from ddpo_pytorch/rewards.py
         (clip | dino | perception | text_image | aesthetic | mse, or any combo)
KL reg : analytic KL on final latent  N(mu_student, I) || N(mu_teacher, I)
         = 1/2 ||mu_s - mu_t||^2   (no step-alignment needed - see `latent_kl`)

PPO loop: old log-probs stored during a no-grad rollout; new log-probs are
recomputed per gradient step against the SAME stored trajectory -> a
mathematically well-defined importance ratio. Multiple inner epochs are run
over each rollout so the clipped surrogate objective is actually meaningful
(see "PPO inner epochs" section below for why a single pass makes clipping a
no-op).

--------------------------------------------------------------------------
Summary of correctness bugs fixed relative to the previous implementation
--------------------------------------------------------------------------
1. Teacher/student noise desync (the bug the user said they'd already fixed,
   but hadn't): neither rollout call passed `latents=` or `generator=` to
   `pipeline_with_logprob`, so `prepare_latents` drew a fresh, independent
   random initial latent for the teacher and for the student every batch.
   Distillation was therefore trying to match images from unrelated noise
   samples. Fixed by `sample_shared_initial_latent()`, called once per
   rollout batch and passed as `latents=` to BOTH the teacher and student
   calls.

   Note on limits of synchronization: the teacher runs `sample.num_steps`
   DDIM steps and the student runs `student.num_steps` steps. Because these
   step counts differ, the two trajectories cannot use an identical
   timestep schedule or identical per-step variance noise - "identical
   latent updates" step-for-step is not achievable (nor meaningful) when
   the number of steps differs. What CAN be, and now is, made identical:
   the initial latent, the seed used to build it, the prompts, and the
   guidance scale. That is the maximal synchronization consistent with a
   50-step teacher distilling into a 5-step student.

2. Evaluation images were not reproducible across epochs: the periodic
   qualitative eval block called `teacher_pipeline(...)` / `student_pipeline(...)`
   with no seed at all, so two different epochs' eval images differed
   because of both the noise AND the model - defeating the entire purpose
   of a fixed eval set. Fixed by seeding a fresh `torch.Generator` per eval
   prompt with a constant seed from config every epoch.

3. Accelerate gradient-accumulation bookkeeping did not match the loop
   structure. The Accelerator was configured with
   `gradient_accumulation_steps = train.gradient_accumulation_steps * student.num_steps`
   (e.g. 2 * 5 = 10), but the loop body only ever executed `student.num_steps`
   (5) calls to `accelerator.accumulate(unet)` per epoch. Since Accelerate
   only calls the underlying `optimizer.step()` / `zero_grad()` once
   `sync_gradients` becomes True at the configured accumulation boundary,
   the optimizer only actually updated once every 2 epochs - silently, with
   no error, while the code logged a loss/reward every single epoch as if
   an update had happened. Fixed by collecting exactly as many rollout
   (mini-)batches as the configured accumulation window expects, and by
   asserting `accelerator.sync_gradients` at the point the loop expects it,
   mirroring upstream DDPO's own internal consistency check.

4. No PPO inner epochs: `config.train.num_inner_epochs` was defined but
   never read. The training loop performed a single forward/backward pass
   per rollout, immediately after which `old_log_probs` still exactly equal
   the just-computed log-probs (no weight update happened in between within
   that pass), so the importance ratio is identically 1 and the clipped
   surrogate term can never bind. This isn't "extra credit" PPO behavior;
   without multiple inner epochs (each reusing the same rollout with a
   shuffled order), `train.clip_range` and the clipping logic are dead code.
   Fixed by an explicit inner-epoch loop with per-inner-epoch shuffling of
   both the sample and timestep dimensions, matching the reference DDPO
   implementation.

5. No per-prompt advantage tracking: `config.per_prompt_stat_tracking` was
   defined but `ddpo_pytorch/stat_tracking.py`'s `PerPromptStatTracker` was
   never imported or used; advantages were whitened using only the current
   (size-8) batch's mean/std, which is extremely high variance. Fixed by
   wiring in `PerPromptStatTracker`, matching upstream DDPO.

6. Checkpointing exists for resume purposes (`config.save_freq`,
   `config.num_checkpoint_limit`, `config.resume_from`), via periodic
   `accelerator.save_state()` / `accelerator.load_state()`.

7. Dead/misleading config wiring: `config.train.kl_lambda` was set in the
   config file but the script actually read a *separate*, CLI-only
   `--kl_lambda` flag (defaulting to 0, i.e. off-by-default with no
   indication in the config file); `config.train.clip_epsilon` and
   `config.train.kl_beta` were defined in the config but never referenced
   anywhere. All KL/clip knobs now live under `config.train.*` and nothing
   else.

8. Reward call `torch.no_grad()` context was actually fine (rewards should
   never carry gradients - see `ddpo_pytorch/clip_distill.py`'s docstring),
   but it was redundant with the reward function's own internal
   `torch.no_grad()`. Left as an explicit outer guard for defense-in-depth,
   since a future reward type forgetting its own no_grad() would otherwise
   silently attach the reward model to the autograd graph.

9. LoRA parameter routing double-checked: `_Wrapper(AttnProcsLayers)` is a
   deliberate (not accidental) trick, kept from upstream DDPO, that lets the
   optimizer/DDP see only the LoRA parameters while the forward pass still
   calls the full UNet (whose attention processors ARE those LoRA modules).
   `unet.parameters()` therefore only yields LoRA weights, so
   `torch.optim.AdamW(unet.parameters(), ...)` only ever updates LoRA
   adapters and the base UNet weights genuinely never receive gradients.
   This part was already correct and is unchanged in spirit; what was
   missing was persisting those LoRA weights via checkpoint hooks (see #6)
   and asserting the invariant in code (see `_assert_parameter_freezing`
   below) instead of only in a comment.

10. Final "student_model" save now keeps the BEST-reward checkpoint, not
    the last epoch's weights: every epoch's mean reward is compared against
    the best mean reward seen so far (across all ranks, using the same
    `gathered_rewards`/`reward_mean` already computed for logging); on a
    new best, `student_pipeline` is saved to the exact same
    `os.path.join(stats_dir, "student_model")` path/layout that the old
    unconditional end-of-run save used - only the trigger condition
    changed, not the save mechanism, folder structure, or naming. The old
    unconditional save after the training loop is removed so it can no
    longer silently overwrite a better earlier checkpoint with a worse
    final one.
"""

import contextlib
import datetime
import json
import os
import random
from collections import defaultdict
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed
import tqdm
from absl import app, flags
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.utils import randn_tensor
from ml_collections import config_flags
from PIL import Image

import ddpo_pytorch.prompts
from ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from ddpo_pytorch.rewards import get_reward_fn
from ddpo_pytorch.stat_tracking import PerPromptStatTracker

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/distill_clip.py", "Training configuration.")
flags.DEFINE_enum(
    "prompt_source", "coco", ["default", "coco"],
    "Prompt source: 'default' = built-in fn from config.prompt_fn, 'coco' = COCO captions.",
)
flags.DEFINE_enum("coco_split", "train", ["train", "val", "both"], "COCO split to use.")
flags.DEFINE_string(
    "coco_annotations_dir", "coco_dataset/annotations", "Path to COCO annotations directory."
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def save_side_by_side(student_images, teacher_images, epoch, outdir):
    os.makedirs(outdir, exist_ok=True)
    student_dir = os.path.join(outdir, "student_only")
    combined_dir = os.path.join(outdir, "side_by_side")
    os.makedirs(student_dir, exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)

    for idx in range(min(len(student_images), len(teacher_images))):
        s = student_images[idx].convert("RGB")
        t = teacher_images[idx].convert("RGB")
        h = min(s.height, t.height)
        s = s.resize((h, h), Image.Resampling.LANCZOS)
        t = t.resize((h, h), Image.Resampling.LANCZOS)

        s.save(os.path.join(student_dir, f"epoch{epoch}_student{idx}.png"))

        combined = Image.new("RGB", (t.width + s.width, h))
        combined.paste(t, (0, 0))
        combined.paste(s, (t.width, 0))
        combined.save(os.path.join(combined_dir, f"epoch{epoch}_sample{idx}.png"))


def rewards_to_tensor(rewards_raw, device) -> torch.Tensor:
    if isinstance(rewards_raw, torch.Tensor):
        return rewards_raw.to(device=device, dtype=torch.float32)
    return torch.tensor(np.array(rewards_raw, dtype=np.float32), device=device)


def latent_kl(student_latent: torch.Tensor, teacher_latent: torch.Tensor) -> torch.Tensor:
    """Analytic KL(N(mu_s, I) || N(mu_t, I)) = 1/2 ||mu_s - mu_t||^2, averaged over batch."""
    s = student_latent.float().view(student_latent.shape[0], -1)
    t = teacher_latent.float().view(teacher_latent.shape[0], -1)
    return 0.5 * ((s - t) ** 2).sum(dim=-1).mean()


def load_pipeline_robust(model_name_or_path, revision, logger):
    """
    `StableDiffusionPipeline.from_pretrained` with a clearer failure mode.

    If `model_name_or_path` is a bare Hub repo id (e.g.
    "runwayml/stable-diffusion-v1-5") rather than a local directory,
    diffusers will try to contact the Hub first. If that fails (expired
    token, no network, repo moved/gated) it silently falls back to the
    *default* `~/.cache/huggingface` cache - which may be a different,
    incomplete snapshot than the one you intended, and the resulting error
    (a missing sub-file deep inside `from_pretrained`) gives no hint that
    the real problem was network/auth, not a corrupted download.

    This wraps that call so that a network/auth failure is retried once
    with `local_files_only=True`, and reports plainly which of "couldn't
    reach the Hub" vs "local cache is incomplete" actually happened, instead
    of surfacing a raw `OSError` about one missing tensor file.
    """
    try:
        return StableDiffusionPipeline.from_pretrained(model_name_or_path, revision=revision)
    except Exception as e:  # noqa: BLE001 - intentionally broad, we re-raise with context
        logger.warning(
            f"from_pretrained('{model_name_or_path}') failed on first attempt "
            f"({type(e).__name__}: {e}). Retrying with local_files_only=True..."
        )
        try:
            return StableDiffusionPipeline.from_pretrained(
                model_name_or_path, revision=revision, local_files_only=True
            )
        except Exception as e2:
            raise RuntimeError(
                f"Could not load '{model_name_or_path}' either from the Hub or from the "
                f"local cache with local_files_only=True. If this is meant to be a purely "
                f"local checkpoint, set config.pretrained.model (and config.student.model) "
                f"to its absolute directory path instead of a Hub repo id - that skips Hub "
                f"lookups entirely. Original errors:\n  first attempt: {e}\n  "
                f"local_files_only retry: {e2}"
            ) from e2


def sample_shared_initial_latent(student_pipeline, batch_size, dtype, device, generator):
    """
    Draw ONE initial latent tensor to be used by both the teacher and the
    student rollout for a given batch. This is the fix for bug #1 above:
    previously each rollout call independently drew its own noise.

    Shape is derived from the student pipeline's UNet/VAE config; teacher and
    student share the same base architecture (both are Stable-Diffusion-v1.5
    variants in this project), so this shape is valid for both.
    """
    unet_cfg = student_pipeline.unet.config
    height = unet_cfg.sample_size * student_pipeline.vae_scale_factor
    width = unet_cfg.sample_size * student_pipeline.vae_scale_factor
    shape = (
        batch_size,
        unet_cfg.in_channels,
        height // student_pipeline.vae_scale_factor,
        width // student_pipeline.vae_scale_factor,
    )
    return randn_tensor(shape, generator=generator, device=device, dtype=dtype)


def _assert_parameter_freezing(accelerator, teacher_pipeline, student_unet_full, unet, use_lora, logger):
    """
    Explicit runtime verification of the freezing strategy, instead of only
    a comment claiming it. Cheap to run once at startup.

    `unet` is the object actually passed to the optimizer (the
    `_Wrapper(AttnProcsLayers)` when use_lora=True, or the raw UNet
    otherwise). `student_unet_full` is always the real, full
    `student_pipeline.unet` module - needed here only to get the TRUE total
    parameter count for the "LoRA is a small fraction of the full UNet"
    check, since `AttnProcsLayers.parameters()` already enumerates ONLY the
    LoRA weights by design (that's the whole point of the wrapper - see
    docstring item #9). Comparing `unet.parameters()` against itself would
    trivially always be equal and is not a useful check.
    """
    for name, module in [
        ("teacher.unet", teacher_pipeline.unet),
        ("teacher.vae", teacher_pipeline.vae),
        ("teacher.text_encoder", teacher_pipeline.text_encoder),
    ]:
        n_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        assert n_trainable == 0, (
            f"{name} has {n_trainable} trainable parameters - the teacher must be "
            f"completely frozen."
        )

    trainable = [p for p in unet.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    total_full_unet = sum(p.numel() for p in student_unet_full.parameters())
    assert n_trainable > 0, "Student has zero trainable parameters - nothing will be optimized."
    if use_lora:
        assert n_trainable < total_full_unet, (
            "use_lora=True but the number of trainable parameters equals the full "
            "UNet parameter count - LoRA does not appear to be attached."
        )
    if accelerator.is_main_process:
        logger.info(
            f"[param-check] trainable params = {n_trainable:,} / {total_full_unet:,} "
            f"full UNet params ({100 * n_trainable / total_full_unet:.3f}%)"
        )


def _assert_gradients_flowed(unet, logger, accelerator):
    """Call once, right after the first backward(), to catch a silently
    disconnected graph (e.g. an accidental .detach() upstream) instead of
    training for hours on zero gradients."""
    n_with_grad = sum(1 for p in unet.parameters() if p.requires_grad and p.grad is not None)
    n_nonzero = sum(
        1
        for p in unet.parameters()
        if p.requires_grad and p.grad is not None and torch.any(p.grad != 0)
    )
    if accelerator.is_main_process:
        logger.info(
            f"[grad-check] {n_with_grad} trainable tensors received a grad; "
            f"{n_nonzero} have at least one nonzero element."
        )
    assert n_with_grad > 0, "No trainable parameter received a gradient - check for a detach()."


# ---------------------------------------------------------------------------
# Checkpointing hooks (mirrors upstream DDPO's approach)
# ---------------------------------------------------------------------------

def make_checkpoint_hooks(config, student_pipeline):
    def save_model_hook(models, weights, output_dir):
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            student_pipeline.unet.save_attn_procs(output_dir)
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # tell Accelerate not to also do its default save for this model

    def load_model_hook(models, input_dir):
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            tmp_unet = UNet2DConditionModel.from_pretrained(
                config.student.model, revision=config.student.revision, subfolder="unet"
            )
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(AttnProcsLayers(tmp_unet.attn_processors).state_dict())
            del tmp_unet
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()

    return save_model_hook, load_model_hook


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(_):
    reward_configs = [
        ["clip"],
        ["dino"],
        ["perception"],
        ["lpips"],
        ["clip", "dino"],
        ["clip", "lpips"],
        ["clip", "perception"],
        ["dino", "lpips"],
        ["dino", "perception"],
        ["perception", "lpips"],
        ["clip", "dino", "perception"],
        ["clip", "dino", "perception", "lpips"],
        ["mse"],
        ["kl"]]
    base_run_name = FLAGS.config.run_name or "ppo"
    for reward_types in reward_configs:
        config = FLAGS.config
        # reward_types = list(config.reward_types)

        unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        reward_tag = "_".join(reward_types)

        config.run_name = f"{base_run_name}_{reward_tag}_{unique_id}"

        if config.resume_from:
            config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
            if "checkpoint_" not in os.path.basename(config.resume_from):
                checkpoints = [d for d in os.listdir(config.resume_from) if "checkpoint_" in d]
                if not checkpoints:
                    raise ValueError(f"No checkpoints found in {config.resume_from}")
                config.resume_from = os.path.join(
                    config.resume_from, sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1]
                )

        reward_tag = "_".join(reward_types)
        outdir = f"Results_KL/PPO_{reward_tag}_kl{config.train.kl_coef}"
        outdir += "_coco" if FLAGS.prompt_source == "coco" else "_default"
        outdir += f"_batch_size_{config.sample.batch_size}"
        outdir += f"_lr_{config.train.learning_rate}"
        outdir += f"_epsilon_{config.train.clip_range}"
        stats_dir = outdir
        stats_file = os.path.join(stats_dir, "training_stats.txt")
        all_losses, all_rewards, all_rewards_std = [], [], []

        # Number of student timesteps trained on per inner epoch.
        num_train_timesteps = int(config.student.num_steps * config.train.timestep_fraction)

        # ------------------------------------------------------------------ #
        # Accelerator
        #
        # Bug fix #3: the number of `accelerator.accumulate(unet)` calls the
        # training loop makes per epoch is EXACTLY
        #     num_inner_epochs * num_minibatches_per_epoch * num_train_timesteps
        # so gradient_accumulation_steps below must equal
        #     train.gradient_accumulation_steps * num_train_timesteps
        # where `train.gradient_accumulation_steps` is set (via the assertion
        # below) to num_minibatches_per_epoch, i.e. one full optimizer step per
        # inner epoch - never a fractional one silently deferred to a future
        # epoch.
        # ------------------------------------------------------------------ #
        accelerator = Accelerator(
            log_with= None,
            mixed_precision=config.mixed_precision,
            project_config=ProjectConfiguration(
                project_dir=os.path.join(config.logdir, config.run_name),
                automatic_checkpoint_naming=True,
                total_limit=config.num_checkpoint_limit,
            ),
            gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
        )
        set_seed(config.seed, device_specific=True)

        if accelerator.is_main_process:
            os.makedirs(stats_dir, exist_ok=True)
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            accelerator.init_trackers(
                "ddpo-distill",
                config={**config.to_dict(), "reward_types": reward_types},
                init_kwargs={"wandb": {"name": config.run_name}},
            )

        logger.info(f"\n{config}")
        logger.info(f"Reward types : {reward_types}")

        # ------------------------------------------------------------------ #
        # Teacher pipeline (frozen)
        # ------------------------------------------------------------------ #
        teacher_pipeline = load_pipeline_robust(config.pretrained.model, config.pretrained.revision, logger)
        teacher_pipeline.scheduler = DDIMScheduler.from_config(teacher_pipeline.scheduler.config)
        teacher_pipeline.safety_checker = None
        teacher_pipeline.set_progress_bar_config(disable=True)
        teacher_pipeline.vae.requires_grad_(False)
        teacher_pipeline.text_encoder.requires_grad_(False)
        teacher_pipeline.unet.requires_grad_(False)
        if accelerator.is_main_process:
            teacher_pipeline.save_pretrained(config.teacher_output_dir)

        # ------------------------------------------------------------------ #
        # Student pipeline (LoRA-trainable UNet, or full UNet if use_lora=False)
        # ------------------------------------------------------------------ #
        student_pipeline = load_pipeline_robust(config.student.model, config.student.revision, logger)
        student_pipeline.scheduler = DDIMScheduler.from_config(student_pipeline.scheduler.config)
        student_pipeline.safety_checker = None
        student_pipeline.set_progress_bar_config(disable=True)
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
                    block_id = int(name[len("up_blocks."):len("up_blocks.") + 1])
                    hidden_size = list(reversed(student_pipeline.unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks."):len("down_blocks.") + 1])
                    hidden_size = student_pipeline.unet.config.block_out_channels[block_id]
                lora_attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
                )
            student_pipeline.unet.set_attn_processor(lora_attn_procs)

            # See docstring item #9: this wrapper is intentional, not accidental.
            # AttnProcsLayers.parameters() enumerates only the LoRA weights, so
            # the optimizer/DDP see only the LoRA parameters while the forward
            # pass still calls the full UNet, since that's what actually has
            # the LoRA-patched attention modules wired into its forward graph.
            class _Wrapper(AttnProcsLayers):
                def forward(self, *args, **kwargs):
                    return student_pipeline.unet(*args, **kwargs)

            unet = _Wrapper(student_pipeline.unet.attn_processors)
        else:
            student_pipeline.unet.requires_grad_(True)
            unet = student_pipeline.unet

        save_model_hook, load_model_hook = make_checkpoint_hooks(config, student_pipeline)
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

        if config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        optimizer = torch.optim.AdamW(
            unet.parameters(),
            lr=config.train.learning_rate,
            betas=(config.train.adam_beta1, config.train.adam_beta2),
            weight_decay=config.train.adam_weight_decay,
            eps=config.train.adam_epsilon,
        )

        # prepare() must come first - it assigns each rank to its own GPU device;
        # only afterwards is accelerator.device reliable per-rank.
        unet, optimizer = accelerator.prepare(unet, optimizer)

        teacher_pipeline.to(accelerator.device)
        inference_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(
            accelerator.mixed_precision, torch.float32
        )
        student_pipeline.vae.to(accelerator.device, dtype=inference_dtype)
        student_pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
        if config.use_lora:
            student_pipeline.unet.to(accelerator.device, dtype=inference_dtype)

        _assert_parameter_freezing(
            accelerator, teacher_pipeline, student_pipeline.unet, unet, config.use_lora, logger
        )

        autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

        if config.resume_from:
            logger.info(f"Resuming from {config.resume_from}")
            accelerator.load_state(config.resume_from)
            first_epoch = int(config.resume_from.split("_")[-1]) + 1
        else:
            first_epoch = 0

        # ------------------------------------------------------------------ #
        # Reward + per-prompt stat tracker
        # ------------------------------------------------------------------ #
        accelerator.wait_for_everyone()
        reward_fn = get_reward_fn(reward_types, teacher_pipeline, student_pipeline)
        stat_tracker = PerPromptStatTracker(
            config.per_prompt_stat_tracking.buffer_size,
            config.per_prompt_stat_tracking.min_count,
        )

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
                logger.warning("No COCO captions loaded - falling back to config.prompt_fn.")
                coco_captions = None

        def sample_prompts(batch_size):
            """Sample on rank 0 only, then broadcast, so every rank trains on the
            same prompts - required for `accelerator.gather(rewards)` and the
            subsequent slice-back-per-rank to be meaningful."""
            if accelerator.is_main_process:
                if FLAGS.prompt_source == "coco" and coco_captions:
                    if len(coco_captions) >= batch_size:
                        prompts = random.sample(coco_captions, k=batch_size)
                    else:
                        prompts = [random.choice(coco_captions) for _ in range(batch_size)]
                else:
                    prompts = [prompt_fn(**config.prompt_fn_kwargs)[0] for _ in range(batch_size)]
            else:
                prompts = [None] * batch_size

            if accelerator.num_processes > 1:
                container = [prompts]
                torch.distributed.broadcast_object_list(container, src=0)
                prompts = container[0]
            return prompts

        # Number of rollout (mini-)batches accumulated before one optimizer step,
        # per inner epoch. Kept equal to train.gradient_accumulation_steps so the
        # Accelerator config above and the loop below agree (bug #3).
        num_minibatches_per_epoch = config.train.gradient_accumulation_steps
        total_batch_size = config.sample.batch_size * num_minibatches_per_epoch
        assert total_batch_size % config.train.batch_size == 0, (
            "sample.batch_size * train.gradient_accumulation_steps must be divisible "
            "by train.batch_size."
        )

        global_step = 0

        # Best-reward tracking for checkpoint #10: this is the ONLY new piece
        # of state driving the "save best, not last" behavior. Everything
        # else about the save (path, format, when-in-the-run it can happen)
        # is unchanged from before.
        best_reward = float("-inf")
        best_reward_epoch = None
        student_model_dir = os.path.join(stats_dir, "student_model")

        # ================================================================== #
        # Training loop
        # ================================================================== #
        for epoch in range(first_epoch, config.num_epochs):
            logger.info(f"Epoch {epoch}: sampling rollouts")

            # ---------------------------------------------------------------- #
            # Rollout phase: collect `num_minibatches_per_epoch` batches, each
            # with its OWN prompts and its OWN shared initial latent (bug #1).
            # ---------------------------------------------------------------- #
            rollout_batches = []
            reward_details_accum = defaultdict(list)

            with torch.no_grad():
                for b in range(num_minibatches_per_epoch):
                    prompts = sample_prompts(config.sample.batch_size)

                    prompt_ids = student_pipeline.tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=student_pipeline.tokenizer.model_max_length,
                    ).input_ids.to(accelerator.device)
                    prompt_embeds = student_pipeline.text_encoder(prompt_ids)[0]

                    # One generator per (epoch, minibatch, rank): deterministic
                    # given the run seed, but different across ranks/epochs so
                    # different ranks/epochs don't all sample identical noise.
                    gen_seed = config.seed + epoch * 100_003 + b * 97 + accelerator.process_index
                    generator = torch.Generator(device=accelerator.device).manual_seed(gen_seed)

                    shared_latent = sample_shared_initial_latent(
                        student_pipeline,
                        batch_size=config.sample.batch_size,
                        dtype=prompt_embeds.dtype,
                        device=accelerator.device,
                        generator=generator,
                    )

                    with accelerator.autocast():
                        teacher_images, _, teacher_latents_all, _ = pipeline_with_logprob(
                            teacher_pipeline,
                            prompt_embeds=prompt_embeds,
                            num_inference_steps=config.sample.num_steps,
                            guidance_scale=config.sample.guidance_scale,
                            eta=config.sample.eta,
                            latents=shared_latent.clone(),
                            output_type="pt",
                            return_all_latents=True,
                        )
                        teacher_final_latent = teacher_latents_all[-1]

                        student_pipeline.unet.eval()
                        student_images, _, student_latents_all, student_log_probs_all = (
                            pipeline_with_logprob(
                                student_pipeline,
                                prompt_embeds=prompt_embeds,
                                num_inference_steps=config.student.num_steps,
                                guidance_scale=config.sample.guidance_scale,
                                eta=config.sample.eta,
                                latents=shared_latent.clone(),
                                output_type="pt",
                                return_all_latents=True,
                            )
                        )

                    latents = torch.stack(student_latents_all, dim=1)          # (B, T+1, C, h, w)
                    old_log_probs = torch.stack(student_log_probs_all, dim=1)  # (B, T)
                    timesteps = student_pipeline.scheduler.timesteps.repeat(
                        config.sample.batch_size, 1
                    )
                    student_final_latent = student_latents_all[-1]

                    rewards_raw, reward_details = reward_fn(student_images, teacher_images, prompts)
                    for k, v in reward_details.items():
                        reward_details_accum[k].extend(v)
                    rewards = rewards_to_tensor(rewards_raw, accelerator.device)

                    rollout_batches.append(
                        {
                            "prompts": prompts,
                            "prompt_embeds": prompt_embeds,
                            "latents": latents,
                            "old_log_probs": old_log_probs,
                            "timesteps": timesteps,
                            "rewards": rewards,
                            "kl": latent_kl(student_final_latent, teacher_final_latent),
                        }
                    )

            # ---------------------------------------------------------------- #
            # Advantages: per-prompt whitening (bug #5), computed on globally
            # gathered rewards/prompts, then sliced back to this rank.
            # ---------------------------------------------------------------- #
            all_rewards_local = torch.cat([b["rewards"] for b in rollout_batches])  # (total_batch_size,)
            all_prompts_local = sum((b["prompts"] for b in rollout_batches), [])

            gathered_rewards = accelerator.gather(all_rewards_local).cpu().numpy()
            if accelerator.num_processes > 1:
                prompt_container = [all_prompts_local]
                gathered_prompts_container = [None] * accelerator.num_processes
                torch.distributed.all_gather_object(gathered_prompts_container, all_prompts_local)
                gathered_prompts = sum(gathered_prompts_container, [])
            else:
                gathered_prompts = all_prompts_local

            advantages_np = stat_tracker.update(gathered_prompts, gathered_rewards)
            advantages_np = np.clip(advantages_np, -config.train.adv_clip_max, config.train.adv_clip_max)
            advantages_all = (
                torch.as_tensor(advantages_np, dtype=torch.float32)
                .reshape(accelerator.num_processes, -1)[accelerator.process_index]
                .to(accelerator.device)
            )
            # split per-rank advantages back out across this rank's rollout batches
            for i, batch in enumerate(rollout_batches):
                start = i * config.sample.batch_size
                end = start + config.sample.batch_size
                batch["advantages"] = advantages_all[start:end]

            kl_reg = torch.stack([b["kl"] for b in rollout_batches]).mean()

            # ---------------------------------------------------------------- #
            # PPO training phase: `num_inner_epochs` passes over the SAME
            # rollout (bug #4), each with a fresh shuffle of sample & timestep
            # order.
            # ---------------------------------------------------------------- #
            student_pipeline.unet.train()
            epoch_info = defaultdict(list)

            for inner_epoch in range(config.train.num_inner_epochs):
                perm = torch.randperm(len(rollout_batches))
                batches_this_pass = [rollout_batches[i] for i in perm]

                for i, batch in enumerate(batches_this_pass):
                    # shuffle timestep order independently per sample, as in
                    # upstream DDPO, so consecutive inner epochs don't always
                    # train timesteps in the same order.
                    bsz = batch["latents"].shape[0]
                    t_perm = torch.stack(
                        [torch.randperm(num_train_timesteps, device=accelerator.device) for _ in range(bsz)]
                    )
                    gather_idx = torch.arange(bsz, device=accelerator.device)[:, None]

                    for j_idx in range(num_train_timesteps):
                        j = t_perm[:, j_idx]  # (B,) - per-sample shuffled timestep index
                        # gather per-sample latents/timesteps/log-probs at index j
                        lat_j = batch["latents"][gather_idx[:, 0], j]
                        next_lat_j = batch["latents"][
                            gather_idx[:, 0], torch.clamp(j + 1, max=batch["latents"].shape[1] - 1)
                        ]
                        t_j = batch["timesteps"][gather_idx[:, 0], j]
                        old_lp_j = batch["old_log_probs"][gather_idx[:, 0], j]

                        with accelerator.accumulate(unet):
                            with autocast():
                                noise_pred = unet(lat_j, t_j, batch["prompt_embeds"]).sample
                                _, new_log_prob = ddim_step_with_logprob(
                                    student_pipeline.scheduler,
                                    noise_pred,
                                    t_j,
                                    lat_j,
                                    eta=config.sample.eta,
                                    prev_sample=next_lat_j,
                                )

                            advantages = batch["advantages"]
                            ratio = torch.exp(new_log_prob - old_lp_j)
                            surr1 = -advantages * ratio
                            surr2 = -advantages * torch.clamp(
                                ratio, 1.0 - config.train.clip_range, 1.0 + config.train.clip_range
                            )
                            policy_loss = torch.mean(torch.maximum(surr1, surr2))
                            step_loss = policy_loss + config.train.kl_coef * kl_reg

                            accelerator.backward(step_loss)
                            if accelerator.sync_gradients:
                                accelerator.clip_grad_norm_(unet.parameters(), config.train.max_grad_norm)
                            optimizer.step()
                            optimizer.zero_grad()

                            if global_step == 0:
                                _assert_gradients_flowed(unet, logger, accelerator)

                        epoch_info["policy_loss"].append(policy_loss.detach())
                        epoch_info["approx_kl"].append(0.5 * torch.mean((new_log_prob - old_lp_j) ** 2).detach())
                        epoch_info["clipfrac"].append(
                            torch.mean((torch.abs(ratio - 1.0) > config.train.clip_range).float()).detach()
                        )

                        if accelerator.sync_gradients:
                            # sanity-check: sync should land exactly at the last
                            # timestep of the last minibatch in this inner epoch.
                            assert (j_idx == num_train_timesteps - 1) and (
                                (i + 1) % num_minibatches_per_epoch == 0
                            ), "gradient sync landed at an unexpected point - accumulation math is off."
                            global_step += 1

                assert accelerator.sync_gradients, "inner epoch ended without an optimizer step."

            # ---------------------------------------------------------------- #
            # Logging
            # ---------------------------------------------------------------- #
            reward_mean = float(np.mean(gathered_rewards))
            reward_std = float(np.std(gathered_rewards))
            total_loss_val = float(torch.stack(epoch_info["policy_loss"]).mean().item())
            all_losses.append(total_loss_val)
            all_rewards.append(reward_mean)
            all_rewards_std.append(reward_std)

            if accelerator.is_main_process:
                detail_str = "  ".join(f"{k}={float(np.mean(v)):.4f}" for k, v in reward_details_accum.items())
                print(
                    f"Epoch {epoch:4d} | reward {reward_mean:+.4f} +/- {reward_std:.4f} "
                    f"| kl {kl_reg.item():.4f} | loss {total_loss_val:.6f}"
                    + (f"  [{detail_str}]" if detail_str else "")
                )
                with open(stats_file, "a") as f:
                    f.write(
                        f"Epoch {epoch}: loss={total_loss_val:.6f}, "
                        f"reward={reward_mean:.6f}+/-{reward_std:.6f}, kl={kl_reg.item():.6f}"
                    )
                    for k, v in reward_details_accum.items():
                        f.write(f", {k}={float(np.mean(v)):.6f}")
                    f.write("\n")

                log_dict = {
                    "reward_mean": reward_mean,
                    "reward_std": reward_std,
                    "kl_latent": kl_reg.item(),
                    "loss": total_loss_val,
                    "approx_kl": torch.stack(epoch_info["approx_kl"]).mean().item(),
                    "clipfrac": torch.stack(epoch_info["clipfrac"]).mean().item(),
                    "epoch": epoch,
                }
                for k, v in reward_details_accum.items():
                    log_dict[f"reward/{k}"] = float(np.mean(v))
                accelerator.log(log_dict, step=epoch)

            # ---------------------------------------------------------------- #
            # Bug fix #10 - save the BEST-reward student checkpoint, not the
            # last one. `reward_mean` above is derived from `gathered_rewards`
            # (already an all-rank collective), so every rank computes the
            # identical `is_new_best` decision - only the main process
            # actually writes to disk, into the exact same path/layout
            # (`stats_dir/student_model`) the old end-of-run save used.
            # `accelerator.wait_for_everyone()` is called unconditionally (by
            # every rank) so this can never deadlock a multi-GPU run, unlike
            # calling it from inside an `is_main_process` block would.
            # ---------------------------------------------------------------- #
            is_new_best = reward_mean > best_reward
            accelerator.wait_for_everyone()
            if is_new_best:
                best_reward = reward_mean
                best_reward_epoch = epoch
                if accelerator.is_main_process:
                    student_pipeline.save_pretrained(student_model_dir)
                    logger.info(
                        f"[best-ckpt] New best reward {best_reward:+.4f} at epoch {epoch} "
                        f"- saved student to {student_model_dir}"
                    )

            # ---------------------------------------------------------------- #
            # Reproducible qualitative evaluation (bug #2): fixed seed + fixed
            # prompts every epoch, so the only thing that can change between
            # epochs' images is the student's weights.
            # ---------------------------------------------------------------- #
            if accelerator.is_main_process and (epoch % config.eval.freq == 0):
                student_pipeline.unet.eval()
                eval_prompts = config.eval.prompts
                with torch.no_grad(), accelerator.autocast():
                    teacher_eval, student_eval = [], []
                    for i, p in enumerate(eval_prompts):
                        t_gen = torch.Generator(device=accelerator.device).manual_seed(config.eval.seed + i)
                        s_gen = torch.Generator(device=accelerator.device).manual_seed(config.eval.seed + i)
                        teacher_eval.append(
                            teacher_pipeline(
                                p,
                                num_inference_steps=config.sample.num_steps,
                                guidance_scale=config.sample.guidance_scale,
                                generator=t_gen,
                            ).images[0]
                        )
                        student_eval.append(
                            student_pipeline(
                                p,
                                num_inference_steps=config.student.num_steps,
                                guidance_scale=config.sample.guidance_scale,
                                generator=s_gen,
                            ).images[0]
                        )
                save_side_by_side(student_eval, teacher_eval, epoch, outdir)

            # ---------------------------------------------------------------- #
            # Resume-style checkpointing (unrelated to best/last student save
            # above - this is Accelerate's own state, for crash recovery)
            # ---------------------------------------------------------------- #
            if epoch != 0 and epoch % config.save_freq == 0:
                accelerator.wait_for_everyone()
                accelerator.save_state()

        # ------------------------------------------------------------------ #
        # End of run: the best-reward student checkpoint was already written
        # to `student_model_dir` during the loop above (bug fix #10) -
        # nothing left to save here. We only render the training curves.
        # ------------------------------------------------------------------ #
        
        

        # ------------------------------------------------------------------ #
        # End of run: the best-reward student checkpoint was already written
        # to `student_model_dir` during the loop above (bug fix #10) -
        # nothing left to save here. We only render the training curves.
        # ------------------------------------------------------------------ #
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            logger.info(
                f"[best-ckpt] Training finished. Best reward = {best_reward:+.4f} "
                f"(epoch {best_reward_epoch}). Student weights saved at {student_model_dir}"
            )

            # ---------------------------------------------------------------- #
            # Publication-style plotting (paper-ready figures)
            # ---------------------------------------------------------------- #
            plt.rcParams.update({
                "font.family": "serif",
                "font.serif": ["Times New Roman", "DejaVu Serif"],
                "font.size": 11,
                "axes.titlesize": 12,
                "axes.labelsize": 11,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 9,
                "axes.linewidth": 0.8,
                "lines.linewidth": 1.6,
                "figure.dpi": 150,
                "savefig.dpi": 300,
                "savefig.bbox": "tight",
                "pdf.fonttype": 42,   # editable text in Illustrator, required by many venues
                "ps.fonttype": 42,
            })

            reward_label = "+".join(r.upper() for r in reward_types)
            epochs = range(len(all_losses))

            # -- Loss curve ---------------------------------------------------
            fig, ax = plt.subplots(figsize=(4.0, 3.0))
            ax.plot(epochs, all_losses, color="#1f77b4", label="Policy loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True, linewidth=0.4, alpha=0.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(frameon=False, loc="best")
            fig.tight_layout()
            fig.savefig(os.path.join(stats_dir, "loss_curve.png"))
            fig.savefig(os.path.join(stats_dir, "loss_curve.pdf"))
            plt.close(fig)

            # -- Reward curve ---------------------------------------------------
            all_rewards_arr = np.array(all_rewards)
            all_rewards_std_arr = np.array(all_rewards_std)

            fig, ax = plt.subplots(figsize=(4.0, 3.0))
            ax.plot(
                epochs, all_rewards_arr,
                color="#1f3fbf",
                label=f"Reward ({reward_label})",
            )
            ax.fill_between(
                epochs,
                all_rewards_arr - all_rewards_std_arr,
                all_rewards_arr + all_rewards_std_arr,
                color="#a9bdf2",
                alpha=0.4,
                linewidth=0,
                label=r"$\pm 1$ std",
            )
            ax.axvline(
                best_reward_epoch,
                color="#B94F4F",
                linestyle="--",
                linewidth=1.0,
                label=f"Best epoch ({best_reward_epoch})",
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Reward")
            ax.grid(True, linewidth=0.4, alpha=0.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(frameon=False, loc="best")
            fig.tight_layout()
            fig.savefig(os.path.join(stats_dir, "reward_curve.png"))
            fig.savefig(os.path.join(stats_dir, "reward_curve.pdf"))
            plt.close(fig)


if __name__ == "__main__":
    app.run(main)