"""
Teacher -> Student distillation on Stable Diffusion / COCO by DIRECT REWARD BACKPROPAGATION
(DRaFT / AlignProp style), as the no-RL control arm for `train_ppo_coco.py`.

This is the SD counterpart of `train_draft_imagenet_edm.py`. Everything that is not the gradient
estimator is imported from `train_ppo_coco.py` rather than copied -- the config file, the absl
flags, the pipeline loaders, the prompt sampling, the shared-latent logic, the reward registry,
the checkpoint hooks, the compute accounting, the run-directory versioning and the plotting are
all literally the same objects. The two scripts therefore cannot drift apart, which is the whole
point of the comparison.

    PPO    (train_ppo_coco.py)   the reward is a black-box scalar; the update is
                                 (importance ratio) x (advantage), a score-function estimator.
                                 One scalar of information per sampled trajectory.

    DRaFT  (this file)           the reward encoders are differentiable, so the reward is
                                 backpropagated through the VAE decoder and the sampling
                                 trajectory into the student. A full gradient per trajectory.

Both optimise the same objective -- alignment between the student's image and the teacher's image
for the same prompt and the same initial latent -- so a difference in outcome is attributable to
the estimator.

Why this is the same MDP, not a different objective
---------------------------------------------------
The policy is unchanged: x_{t-1} = mu_theta(x_t) + sigma_t * z, with sigma_t set by
`config.sample.eta`. During a rollout z is drawn and then held fixed. PPO differentiates
log pi(x_{t-1} | x_t) and multiplies by the advantage; DRaFT differentiates the reward through
mu_theta with z treated as a constant -- the pathwise derivative of the same quantity. Keeping
`config.sample.eta` identical across the two arms makes the sampling distributions identical too.
Both use `ddim_step_with_logprob`, so the scheduler arithmetic is shared down to the line.

Memory
------
Backpropagating through K sampling steps stores activations for K UNet forwards plus one VAE
decode at full resolution -- the binding constraint on SD, far more than compute. Three
mitigations, all on by default:
  --backprop_steps K   gradient flows only through the last K steps (DRaFT-K; K=1 is the variant
                       the DRaFT paper mainly uses). K < num_steps makes the gradient a biased
                       estimate of the true objective gradient, so report K in the paper.
  --grad_checkpoint    recompute activations in the backward pass instead of storing them.
  --draft_cfg          whether the differentiable rollout uses classifier-free guidance. Off
                       halves the UNet cost; see the note at `sample_with_grad`.

Usage (matched against a PPO run)
---------------------------------
  python train_draft_coco.py --config config/distill_clip.py \\
      --prompt_source coco --backprop_steps 1
"""

import contextlib
import copy
import datetime
import json
import os
import random
import sys
import importlib.util
from collections import defaultdict
from functools import partial

import numpy as np
import torch
import torch.distributed
import torch.nn.functional as Fn
import tqdm
from absl import app, flags
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor

HERE = os.path.dirname(os.path.abspath(__file__))


def load_ppo_module():
    """Import the PPO trainer as a library.

    This also registers its absl flags (--config, --prompt_source, --coco_split,
    --coco_annotations_dir), so this script inherits an identical command-line interface instead
    of redefining it. Only the DRaFT-specific flags below are new.
    """
    path = os.path.join(HERE, "train_ppo_coco.py")
    spec = importlib.util.spec_from_file_location("redif_ppo_coco", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["redif_ppo_coco"] = mod
    spec.loader.exec_module(mod)
    return mod


P = load_ppo_module()

from ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob  # noqa: E402
from ddpo_pytorch.rewards import get_reward_fn  # noqa: E402
import ddpo_pytorch.prompts  # noqa: E402

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
logger = get_logger(__name__)

FLAGS = P.FLAGS
flags.DEFINE_integer(
    "backprop_steps", 1,
    "DRaFT-K: gradient flows only through the last K sampling steps; earlier steps run under "
    "no_grad. 0 = all steps (most faithful to what PPO trains, since PPO updates every timestep, "
    "but K times the activation memory).")
flags.DEFINE_boolean(
    "grad_checkpoint", True,
    "Recompute UNet/VAE activations during the backward pass instead of storing them.")
flags.DEFINE_boolean(
    "draft_cfg", True,
    "Run classifier-free guidance inside the differentiable rollout. Matches the PPO arm's "
    "rollout, at 2x the UNet cost.")
flags.DEFINE_boolean(
    "reward_check", True,
    "At startup, verify the differentiable reward reproduces the RL arm's reward_fn on the same "
    "batch. Disable only if a term legitimately has no differentiable form.")
flags.DEFINE_float(
    "reward_check_tol", 2e-3,
    "Absolute tolerance for that verification.")
flags.DEFINE_boolean(
    "match_pipeline_decode", True,
    "Reproduce the RL arm's exact VAE-decode-to-pixels chain, including its double range "
    "transformation (see decode_to_images). Required for a like-for-like comparison; turn off "
    "only if the RL arm is changed to match.")


# ---------------------------------------------------------------------------
# Differentiable reward
# ---------------------------------------------------------------------------

class DifferentiableReward:
    """The reward terms of `ddpo_pytorch/rewards.py`, with the graph kept.

    Those functions are correct for the RL arm and deliberately unusable here: every encoder runs
    under `torch.no_grad()` and every reward returns `.tolist()`, so nothing downstream of the
    reward can receive a gradient. This class mirrors them term for term, using the SAME encoder
    checkpoints, the SAME input resolutions and the SAME normalisation constants.

    Because "mirrors" is a claim and not a guarantee, `verify()` runs both implementations on the
    same batch at startup and refuses to train if they disagree. That is what makes the arm-to-arm
    comparison defensible: the reward is not merely intended to be identical, it is checked.

    The teacher side is a constant and is encoded under no_grad; only the student side carries
    gradient. Note the clamp in preprocessing: pixels pushed outside [0,1] receive zero gradient,
    which is standard DRaFT behaviour and keeps the reward values comparable across arms.
    """

    IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    SUPPORTED = ("clip", "dino", "perception", "lpips", "mse")

    def __init__(self, reward_types, device, dino_size=512):
        self.device = device
        self.terms = list(reward_types)
        self.dino_size = dino_size
        unsupported = [t for t in self.terms if t not in self.SUPPORTED]
        if unsupported:
            raise ValueError(
                f"reward term(s) {unsupported} have no differentiable implementation in the DRaFT "
                f"arm (the aesthetic scorer takes uint8 input; text_image and kl are either "
                f"non-differentiable or trivially so). Choose from {list(self.SUPPORTED)} so the "
                f"two arms optimise the same objective.")

        self.clip = self.dino = self.pe = self.lpips = None
        self._dino_prefix = 1

        if "clip" in self.terms:
            from transformers import CLIPModel
            self.clip = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32").to(device).eval().requires_grad_(False)
        if "dino" in self.terms:
            import timm
            self.dino = timm.create_model("vit_base_patch16_dinov3.lvd1689m", pretrained=True,
                                          num_classes=0, dynamic_img_size=True)
            self.dino = self.dino.to(device).eval().requires_grad_(False)
            self._dino_prefix = getattr(self.dino, "num_prefix_tokens", 1)
        if "perception" in self.terms:
            import timm
            self.pe = timm.create_model("vit_pe_core_base_patch16_224.fb", pretrained=True,
                                        num_classes=0).to(device).eval().requires_grad_(False)
            try:
                cfg = timm.data.resolve_data_config({}, model=self.pe)
                self.pe_mean = list(cfg.get("mean", (0.5,) * 3))
                self.pe_std = list(cfg.get("std", (0.5,) * 3))
                self.pe_size = cfg.get("input_size", (3, 224, 224))[-1]
            except Exception:
                self.pe_mean, self.pe_std, self.pe_size = [0.5] * 3, [0.5] * 3, 224
        if "lpips" in self.terms:
            import lpips as lpips_lib
            self.lpips = lpips_lib.LPIPS(net="vgg").to(device).eval().requires_grad_(False)

    # -- preprocessing ------------------------------------------------------

    def _prep(self, x01, mean, std, size):
        x = Fn.interpolate(x01, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)
        m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        s = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x - m) / s

    # -- per-term embeddings -------------------------------------------------

    def _emb_clip(self, x01):
        e = self.clip.get_image_features(
            pixel_values=self._prep(x01, self.CLIP_MEAN, self.CLIP_STD, 224))
        return Fn.normalize(e.float(), dim=-1)

    def _emb_dino(self, x01):
        f = self.dino.forward_features(
            self._prep(x01, self.IMNET_MEAN, self.IMNET_STD, self.dino_size))
        return Fn.normalize(f[:, self._dino_prefix:, :].float(), dim=-1)   # patch tokens

    def _emb_pe(self, x01):
        e = self.pe(self._prep(x01, self.pe_mean, self.pe_std, self.pe_size))
        return Fn.normalize(e.float(), dim=-1)

    # -- public API ----------------------------------------------------------

    def __call__(self, student01, teacher01):
        """student01/teacher01: [B,3,H,W] float in [0,1]. Returns (total [B], per-term dict)."""
        parts = {}
        with torch.no_grad():
            t_emb = {}
            if self.clip is not None:
                t_emb["clip"] = self._emb_clip(teacher01)
            if self.dino is not None:
                t_emb["dino"] = self._emb_dino(teacher01)
            if self.pe is not None:
                t_emb["perception"] = self._emb_pe(teacher01)

        for term in self.terms:
            if term == "clip":
                parts[term] = (self._emb_clip(student01) * t_emb["clip"]).sum(-1)
            elif term == "dino":
                # per-token cosine, averaged over tokens -- the aggregation used by
                # dinov3_reward_fn in rewards.py
                parts[term] = Fn.cosine_similarity(
                    self._emb_dino(student01), t_emb["dino"], dim=2).mean(dim=1)
            elif term == "perception":
                parts[term] = (self._emb_pe(student01) * t_emb["perception"]).sum(-1)
            elif term == "lpips":
                s = self._prep(student01, (0.5,) * 3, (0.5,) * 3, 224) * 1.0
                t = self._prep(teacher01, (0.5,) * 3, (0.5,) * 3, 224) * 1.0
                parts[term] = -self.lpips(s, t).view(-1)
            elif term == "mse":
                parts[term] = -((student01 - teacher01) ** 2).mean(dim=(1, 2, 3))

        total = sum(parts.values())
        return total, parts

    @torch.no_grad()
    def verify(self, rl_reward_fn, student01, teacher01, tol, logger_=None):
        """Assert this reward matches the RL arm's reward_fn on the same images."""
        mine, mine_parts = self(student01, teacher01)
        theirs, theirs_parts = rl_reward_fn(student01, teacher01, None)
        theirs = np.asarray(theirs, dtype=np.float64)
        mine_np = mine.detach().float().cpu().numpy().astype(np.float64)
        msg = [f"[reward-check] total: max|diff| = {np.abs(mine_np - theirs).max():.2e}"]
        for k in mine_parts:
            if theirs_parts and k in theirs_parts:
                d = np.abs(mine_parts[k].detach().float().cpu().numpy()
                           - np.asarray(theirs_parts[k], dtype=np.float64)).max()
                msg.append(f"[reward-check]   {k}: max|diff| = {d:.2e}")
        text = "\n".join(msg)
        print(text)
        if logger_ is not None:
            logger_.info(text)
        worst = np.abs(mine_np - theirs).max()
        if worst > tol:
            raise RuntimeError(
                f"The DRaFT arm's differentiable reward disagrees with the RL arm's reward_fn by "
                f"{worst:.3e} > tol {tol:.1e}. The two arms would not be optimising the same "
                f"objective, so the comparison would be meaningless. Fix the mismatch (check the "
                f"encoder checkpoints, the input resolutions and the normalisation constants) or "
                f"pass --noreward_check if you know why they differ.")
        return worst


# ---------------------------------------------------------------------------
# Differentiable sampling
# ---------------------------------------------------------------------------

def sample_with_grad(pipeline, unet_callable, prompt_embeds, uncond_embeds, latents,
                     num_steps, guidance_scale, eta, backprop_steps, grad_checkpoint,
                     budget=None, n_images=0, count_key="unet"):
    """DDIM sampling that keeps the graph for the last `backprop_steps` transitions.

    Identical trajectory distribution to `pipeline_with_logprob` -- same scheduler, same
    `ddim_step_with_logprob` call, same eta, same CFG formula. The only differences are that the
    graph is retained and that the injected noise is treated as a constant (reparameterisation),
    which is what turns the score-function estimator into a pathwise one.

    Returns the final latent, still attached to the graph.
    """
    scheduler = pipeline.scheduler
    scheduler.set_timesteps(num_steps, device=latents.device)
    timesteps = scheduler.timesteps
    do_cfg = guidance_scale > 1.0 and uncond_embeds is not None
    k = num_steps if backprop_steps <= 0 else min(backprop_steps, num_steps)
    first_grad_step = num_steps - k

    embeds = torch.cat([uncond_embeds, prompt_embeds]) if do_cfg else prompt_embeds

    for i, t in enumerate(timesteps):
        use_grad = i >= first_grad_step

        def step(lat):
            model_in = torch.cat([lat] * 2) if do_cfg else lat
            model_in = scheduler.scale_model_input(model_in, t)
            pred = unet_callable(model_in, t, embeds).sample
            if do_cfg:
                pred_uncond, pred_text = pred.chunk(2)
                pred = pred_uncond + guidance_scale * (pred_text - pred_uncond)
            return pred

        if use_grad:
            noise_pred = (torch.utils.checkpoint.checkpoint(step, latents, use_reentrant=False)
                          if grad_checkpoint else step(latents))
            if budget is not None:
                budget.add(count_key, n_images * (2 if do_cfg else 1), backward=True,
                           is_nfe=(count_key == "unet"))
                if grad_checkpoint:
                    budget.add(count_key, n_images * (2 if do_cfg else 1))
        else:
            with torch.no_grad():
                noise_pred = step(latents)
            if budget is not None:
                budget.add(count_key, n_images * (2 if do_cfg else 1),
                           is_nfe=(count_key == "unet"))

        # ddim_step_with_logprob draws the noise internally when prev_sample is None; it is a
        # constant w.r.t. theta, so the gradient flows only through prev_sample_mean.
        ts = t.repeat(latents.shape[0]) if torch.is_tensor(t) else torch.full(
            (latents.shape[0],), t, device=latents.device, dtype=torch.long)
        latents, _ = ddim_step_with_logprob(scheduler, noise_pred, ts, latents, eta=eta)

    return latents


def decode_to_images(pipeline, latents, grad=True, match_pipeline=True):
    """VAE decode to pixels, differentiably when `grad` is set.

    `match_pipeline=True` reproduces the exact pixel chain that `pipeline_with_logprob` feeds to
    the reward in the RL arm:

        x = vae.decode(z / scaling)          # roughly [-1, 1]
        x = nan_to_num(x.clamp(0, 1))        # negatives crushed to 0  -> [0, 1]
        x = (x / 2 + 0.5).clamp(0, 1)        # postprocess denormalise -> [0.5, 1]

    The second and third lines compose into a transformation that is almost certainly not what
    was intended: `clamp(0,1)` already maps the decoder output to unit range, and the subsequent
    denormalisation -- which assumes a [-1,1] input -- then compresses everything into the upper
    half of the range. The RL arm therefore computes its reward on washed-out images.

    This is replicated rather than fixed, because the two arms must see identical pixels for the
    comparison to mean anything. Pass --nomatch_pipeline_decode to use the correct chain in BOTH
    arms once the RL side is fixed too; changing only one of them would invalidate the ablation.
    """
    ctx = contextlib.nullcontext() if grad else torch.no_grad()
    with ctx:
        img = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor,
                                  return_dict=False)[0]
        if match_pipeline:
            img = torch.nan_to_num(img.clamp(0, 1))
        img = (img / 2 + 0.5).clamp(0, 1)
    return img


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
    ]
    base_run_name = FLAGS.config.run_name or "draft"

    for reward_types in reward_configs:
        config = FLAGS.config
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
                    config.resume_from,
                    sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1])

        # Directory naming mirrors the PPO arm so the two sit side by side, with DRAFT_ and the
        # backprop depth in the name -- K is a real axis of the method, not an implementation
        # detail, and must be visible when the results are compared.
        outdir = f"Results_KL/DRAFT_{reward_tag}_kl{config.train.kl_coef}"
        outdir += "_coco" if FLAGS.prompt_source == "coco" else "_default"
        outdir += f"_batch_size_{config.sample.batch_size}"
        outdir += f"_lr_{config.train.learning_rate}"
        outdir += f"_k{FLAGS.backprop_steps}"
        outdir = P.unique_run_dir(outdir, enabled=not config.resume_from)
        stats_dir = outdir
        stats_file = os.path.join(stats_dir, "training_stats.txt")
        jsonl_file = os.path.join(stats_dir, "log.jsonl")
        all_losses, all_rewards, all_rewards_std = [], [], []

        # One optimizer step per epoch, gradients accumulated over every rollout minibatch --
        # the same cadence as the PPO arm, so "epoch" means the same thing on both x-axes.
        num_minibatches_per_epoch = config.train.gradient_accumulation_steps

        accelerator = Accelerator(
            log_with=None,
            mixed_precision=config.mixed_precision,
            project_config=ProjectConfiguration(
                project_dir=os.path.join(config.logdir, config.run_name),
                automatic_checkpoint_naming=True,
                total_limit=config.num_checkpoint_limit,
            ),
            gradient_accumulation_steps=num_minibatches_per_epoch,
        )
        set_seed(config.seed, device_specific=True)

        if accelerator.is_main_process:
            os.makedirs(stats_dir, exist_ok=True)
            print(f"[run] writing to {os.path.abspath(outdir)}")
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            accelerator.init_trackers(
                "ddpo-distill-draft",
                config={**config.to_dict(), "reward_types": reward_types,
                        "method": "draft", "backprop_steps": FLAGS.backprop_steps},
                init_kwargs={"wandb": {"name": config.run_name}})

        logger.info(f"\n{config}")
        logger.info(f"Reward types : {reward_types} (DRaFT, K={FLAGS.backprop_steps})")

        # ---------------- teacher (frozen) ---------------------------------
        teacher_pipeline = P.load_pipeline_robust(
            config.pretrained.model, config.pretrained.revision, logger)
        teacher_pipeline.scheduler = DDIMScheduler.from_config(teacher_pipeline.scheduler.config)
        teacher_pipeline.safety_checker = None
        teacher_pipeline.set_progress_bar_config(disable=True)
        teacher_pipeline.vae.requires_grad_(False)
        teacher_pipeline.text_encoder.requires_grad_(False)
        teacher_pipeline.unet.requires_grad_(False)
        if accelerator.is_main_process:
            teacher_pipeline.save_pretrained(config.teacher_output_dir)

        # ---------------- student (LoRA-trainable) -------------------------
        student_pipeline = P.load_pipeline_robust(
            config.student.model, config.student.revision, logger)
        student_pipeline.scheduler = DDIMScheduler.from_config(student_pipeline.scheduler.config)
        student_pipeline.safety_checker = None
        student_pipeline.set_progress_bar_config(disable=True)
        student_pipeline.vae.requires_grad_(False)
        student_pipeline.text_encoder.requires_grad_(False)

        if config.use_lora:
            lora_attn_procs = {}
            for name in student_pipeline.unet.attn_processors.keys():
                cross_attention_dim = (
                    None if name.endswith("attn1.processor")
                    else student_pipeline.unet.config.cross_attention_dim)
                if name.startswith("mid_block"):
                    hidden_size = student_pipeline.unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks."):len("up_blocks.") + 1])
                    hidden_size = list(reversed(
                        student_pipeline.unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks."):len("down_blocks.") + 1])
                    hidden_size = student_pipeline.unet.config.block_out_channels[block_id]
                lora_attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            student_pipeline.unet.set_attn_processor(lora_attn_procs)

            class _Wrapper(AttnProcsLayers):
                def forward(self, *args, **kwargs):
                    return student_pipeline.unet(*args, **kwargs)

            unet = _Wrapper(student_pipeline.unet.attn_processors)
        else:
            student_pipeline.unet.requires_grad_(True)
            unet = student_pipeline.unet

        save_model_hook, load_model_hook = P.make_checkpoint_hooks(config, student_pipeline)
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

        if config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        optimizer = torch.optim.AdamW(
            unet.parameters(),
            lr=config.train.learning_rate,
            betas=(config.train.adam_beta1, config.train.adam_beta2),
            weight_decay=config.train.adam_weight_decay,
            eps=config.train.adam_epsilon)

        unet, optimizer = accelerator.prepare(unet, optimizer)

        teacher_pipeline.to(accelerator.device)
        inference_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(
            accelerator.mixed_precision, torch.float32)
        student_pipeline.vae.to(accelerator.device, dtype=inference_dtype)
        student_pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
        if config.use_lora:
            student_pipeline.unet.to(accelerator.device, dtype=inference_dtype)

        P._assert_parameter_freezing(
            accelerator, teacher_pipeline, student_pipeline.unet, unet, config.use_lora, logger)

        if FLAGS.grad_checkpoint:
            # The VAE decode at 512x512 is the single largest activation block here.
            try:
                student_pipeline.vae.enable_gradient_checkpointing()
            except Exception as e:
                logger.warning(f"VAE gradient checkpointing unavailable ({e})")

        autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

        if config.resume_from:
            logger.info(f"Resuming from {config.resume_from}")
            accelerator.load_state(config.resume_from)
            first_epoch = int(config.resume_from.split("_")[-1]) + 1
        else:
            first_epoch = 0

        # ---------------- rewards ------------------------------------------
        accelerator.wait_for_everyone()
        rl_reward_fn = get_reward_fn(reward_types, teacher_pipeline, student_pipeline)
        diff_reward = DifferentiableReward(reward_types, accelerator.device)

        # ---------------- prompts (identical to the PPO arm) ---------------
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

        def encode_prompts(prompts):
            ids = student_pipeline.tokenizer(
                prompts, return_tensors="pt", padding="max_length", truncation=True,
                max_length=student_pipeline.tokenizer.model_max_length).input_ids.to(
                    accelerator.device)
            return student_pipeline.text_encoder(ids)[0]

        # ---------------- compute accounting -------------------------------
        budget = P.FlopBudget(enabled=bool(getattr(config, "count_flops", 1)))
        if budget.enabled and accelerator.is_main_process:
            try:
                u = student_pipeline.unet
                d = u.config.sample_size
                pl = torch.zeros(1, u.config.in_channels, d, d, device=accelerator.device,
                                 dtype=next(u.parameters()).dtype)
                pt = torch.zeros(1, device=accelerator.device, dtype=torch.long)
                pe = torch.zeros(1, student_pipeline.tokenizer.model_max_length,
                                 u.config.cross_attention_dim, device=accelerator.device,
                                 dtype=pl.dtype)
                budget.calibrate("unet", u, (pl, pt, pe))
                budget.calibrate("vae_decode", student_pipeline.vae.decode, (pl,))
            except Exception as e:
                print(f"[flops] calibration skipped ({e})")

        global_step = 0
        best_reward = float("-inf")
        best_reward_epoch = None
        student_model_dir = os.path.join(stats_dir, "student_model")
        checked_reward = not FLAGS.reward_check

        # =================================================================== #
        # Training loop
        # =================================================================== #
        for epoch in range(first_epoch, config.num_epochs):
            logger.info(f"Epoch {epoch}: DRaFT step")
            student_pipeline.unet.train()
            optimizer.zero_grad(set_to_none=True)

            epoch_rewards, reward_details_accum = [], defaultdict(list)
            epoch_losses = []

            for b in range(num_minibatches_per_epoch):
                prompts = sample_prompts(config.sample.batch_size)
                prompt_embeds = encode_prompts(prompts)
                uncond_embeds = encode_prompts([""] * len(prompts))

                gen_seed = (config.seed + epoch * 100_003 + b * 97
                            + accelerator.process_index)
                generator = torch.Generator(device=accelerator.device).manual_seed(gen_seed)
                shared_latent = P.sample_shared_initial_latent(
                    student_pipeline, batch_size=config.sample.batch_size,
                    dtype=prompt_embeds.dtype, device=accelerator.device, generator=generator)

                B = config.sample.batch_size
                cfg_mult = 2 if config.sample.guidance_scale > 1.0 else 1

                # ---- teacher target: no grad, same sampler, same shared latent -----------
                with torch.no_grad(), accelerator.autocast():
                    teacher_latent = sample_with_grad(
                        teacher_pipeline, teacher_pipeline.unet, prompt_embeds, uncond_embeds,
                        shared_latent.clone(), config.sample.num_steps,
                        config.sample.guidance_scale, config.sample.eta,
                        backprop_steps=0, grad_checkpoint=False)
                    teacher_images = decode_to_images(
                        teacher_pipeline, teacher_latent, grad=False,
                        match_pipeline=FLAGS.match_pipeline_decode)
                budget.add("unet_teacher", B * config.sample.num_steps * cfg_mult)
                budget.add("vae_decode", B)

                # ---- student: differentiable ---------------------------------------------
                with autocast():
                    student_latent = sample_with_grad(
                        student_pipeline, unet, prompt_embeds, uncond_embeds,
                        shared_latent.clone(), config.student.num_steps,
                        config.sample.guidance_scale if FLAGS.draft_cfg else 1.0,
                        config.sample.eta,
                        backprop_steps=FLAGS.backprop_steps,
                        grad_checkpoint=FLAGS.grad_checkpoint,
                        budget=budget, n_images=B)
                    student_images = decode_to_images(
                        student_pipeline, student_latent, grad=True,
                        match_pipeline=FLAGS.match_pipeline_decode)
                budget.add("vae_decode", B, backward=True)

                if not checked_reward:
                    diff_reward.verify(rl_reward_fn, student_images.detach(), teacher_images,
                                       FLAGS.reward_check_tol, logger)
                    checked_reward = True

                reward, parts = diff_reward(student_images, teacher_images)
                budget.add("reward", B, backward=True)
                budget.add("reward", B)

                loss = -reward.mean()
                if config.train.kl_coef > 0:
                    # Unlike the PPO arm -- where latent_kl is computed inside the no-grad rollout
                    # and therefore contributes no gradient at all -- this term is live here.
                    # Set kl_coef = 0 to match the PPO arm's EFFECTIVE behaviour.
                    loss = loss + config.train.kl_coef * P.latent_kl(student_latent, teacher_latent)

                accelerator.backward(loss / num_minibatches_per_epoch)

                if global_step == 0 and b == 0:
                    P._assert_gradients_flowed(unet, logger, accelerator)

                epoch_rewards.append(reward.detach().float().cpu())
                epoch_losses.append(float(loss.detach().item()))
                for k, v in parts.items():
                    reward_details_accum[k].extend(
                        v.detach().float().cpu().numpy().tolist())

            grad_norm = accelerator.clip_grad_norm_(unet.parameters(), config.train.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            # ------------------------------- logging -------------------------------
            local_rewards = torch.cat(epoch_rewards).to(accelerator.device)
            gathered_rewards = accelerator.gather(local_rewards).cpu().numpy()
            reward_mean = float(np.mean(gathered_rewards))
            reward_std = float(np.std(gathered_rewards))
            total_loss_val = float(np.mean(epoch_losses))
            all_losses.append(total_loss_val)
            all_rewards.append(reward_mean)
            all_rewards_std.append(reward_std)

            if accelerator.is_main_process:
                detail_str = "  ".join(f"{k}={float(np.mean(v)):.4f}"
                                       for k, v in reward_details_accum.items())
                print(f"Epoch {epoch:4d} | reward {reward_mean:+.4f} +/- {reward_std:.4f} "
                      f"| loss {total_loss_val:.6f}" + (f"  [{detail_str}]" if detail_str else ""))
                with open(stats_file, "a") as f:
                    f.write(f"Epoch {epoch}: loss={total_loss_val:.6f}, "
                            f"reward={reward_mean:.6f}+/-{reward_std:.6f}")
                    for k, v in reward_details_accum.items():
                        f.write(f", {k}={float(np.mean(v)):.6f}")
                    f.write("\n")

                log_dict = {
                    "method": "draft",
                    "epoch": epoch,
                    "reward_mean": reward_mean,
                    "reward_std": reward_std,
                    "loss": total_loss_val,
                    "grad_norm": float(grad_norm) if grad_norm is not None else None,
                    "backprop_steps": FLAGS.backprop_steps,
                    "draft_cfg": bool(FLAGS.draft_cfg),
                    "match_pipeline_decode": bool(FLAGS.match_pipeline_decode),
                }
                if budget.enabled:
                    log_dict["nfe_cum"] = budget.nfe
                    log_dict["pflops_cum"] = budget.total_flops() / 1e15
                for k, v in reward_details_accum.items():
                    log_dict[f"reward/{k}"] = float(np.mean(v))
                accelerator.log(log_dict, step=epoch)
                with open(jsonl_file, "a") as f:
                    f.write(json.dumps(log_dict) + "\n")
                if budget.enabled:
                    with open(os.path.join(stats_dir, "flops.json"), "w") as f:
                        json.dump(budget.summary(epoch), f, indent=2)

            # ------------------------- best checkpoint ------------------------------
            is_new_best = reward_mean > best_reward
            accelerator.wait_for_everyone()
            if is_new_best:
                best_reward = reward_mean
                best_reward_epoch = epoch
                if accelerator.is_main_process:
                    student_pipeline.save_pretrained(student_model_dir)
                    logger.info(f"[best-ckpt] New best reward {best_reward:+.4f} at epoch {epoch} "
                                f"- saved student to {student_model_dir}")

            # ------------------------- qualitative eval -----------------------------
            if accelerator.is_main_process and (epoch % config.eval.freq == 0):
                student_pipeline.unet.eval()
                with torch.no_grad(), accelerator.autocast():
                    teacher_eval, student_eval = [], []
                    for i, p in enumerate(config.eval.prompts):
                        t_gen = torch.Generator(device=accelerator.device).manual_seed(
                            config.eval.seed + i)
                        s_gen = torch.Generator(device=accelerator.device).manual_seed(
                            config.eval.seed + i)
                        teacher_eval.append(teacher_pipeline(
                            p, num_inference_steps=config.sample.num_steps,
                            guidance_scale=config.sample.guidance_scale,
                            generator=t_gen).images[0])
                        student_eval.append(student_pipeline(
                            p, num_inference_steps=config.student.num_steps,
                            guidance_scale=config.sample.guidance_scale,
                            generator=s_gen).images[0])
                P.save_side_by_side(student_eval, teacher_eval, epoch, outdir)

            if epoch != 0 and epoch % config.save_freq == 0:
                accelerator.wait_for_everyone()
                accelerator.save_state()

        # ---------------------------- end of run --------------------------------
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            logger.info(f"[best-ckpt] Training finished. Best reward = {best_reward:+.4f} "
                        f"(epoch {best_reward_epoch}). Student saved at {student_model_dir}")
            if budget.enabled:
                logger.info(
                    f"[flops] total {budget.total_flops() / 1e15:.3f} PFLOPs "
                    f"({budget.nfe:.0f} student NFE-images); breakdown: "
                    + ", ".join(f"{k}={v:.3g} img-fwd" for k, v in budget.calls.items()))

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.rcParams.update({
                "font.family": "serif",
                "font.serif": ["Times New Roman", "DejaVu Serif"],
                "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
                "axes.linewidth": 0.8, "lines.linewidth": 1.6,
                "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
                "pdf.fonttype": 42, "ps.fonttype": 42,
            })
            reward_label = "+".join(r.upper() for r in reward_types)
            epochs = range(len(all_losses))

            fig, ax = plt.subplots(figsize=(4.0, 3.0))
            ax.plot(epochs, all_losses, color="#1f77b4", label="DRaFT loss ($-$reward)")
            ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
            ax.grid(True, linewidth=0.4, alpha=0.4)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.legend(frameon=False, loc="best"); fig.tight_layout()
            fig.savefig(os.path.join(stats_dir, "loss_curve.png"))
            fig.savefig(os.path.join(stats_dir, "loss_curve.pdf"))
            plt.close(fig)

            arr = np.array(all_rewards); sd = np.array(all_rewards_std)
            fig, ax = plt.subplots(figsize=(4.0, 3.0))
            handles = [ax.fill_between(epochs, arr - sd, arr + sd, color="#a9bdf2", alpha=0.4,
                                       linewidth=0, label=r"$\pm 1$ std")]
            handles += ax.plot(epochs, arr, color="#1f3fbf", label=f"Reward ({reward_label})")
            if best_reward_epoch is not None:
                handles.append(ax.axvline(best_reward_epoch, color="#B94F4F", linestyle="--",
                                          linewidth=1.0,
                                          label=f"Best epoch ({best_reward_epoch})"))
            ax.set_xlabel("Epoch"); ax.set_ylabel("Reward")
            ax.grid(True, linewidth=0.4, alpha=0.4)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.legend(handles=handles, labels=[h.get_label() for h in handles],
                      frameon=False, loc="upper left", bbox_to_anchor=(0.01, 0.90))
            fig.tight_layout()
            fig.savefig(os.path.join(stats_dir, "reward_curve.png"))
            fig.savefig(os.path.join(stats_dir, "reward_curve.pdf"))
            plt.close(fig)

        del teacher_pipeline, student_pipeline, unet, optimizer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    app.run(main)
