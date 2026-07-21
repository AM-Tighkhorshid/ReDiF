"""
scripts/baselines/flop_budget.py
---------------------------------
Shared utility so that every baseline in this folder can be trained for
(approximately) the same total number of FLOPs as `scripts/train_ppo_coco.py`
(ReDiF's own method), instead of an arbitrary/fixed number of epochs that
happens to look similar on paper but is actually cheaper or more expensive
in wall-clock/compute terms because each method calls the UNet (and, for
DMD2/adversarial, extra auxiliary networks) a different number of times per
"unit" of its own training loop.

How it works
------------
1. `calibrate_call(module, make_inputs, backward=True)` runs the module
   EXACTLY ONCE on freshly-created, randomly-initialized dummy tensors of the
   correct shape and measures the FLOPs of that single forward (+ backward)
   call with `torch.utils.flop_counter.FlopCounterMode`. Only the *shape* of
   the calibration input matters for a FLOP count, not its values, so this is
   safe to run before the very first real batch, with already-loaded model
   weights - it costs one extra forward/backward pass, nothing else.

2. `reference_budget_flops(ref_config, teacher_flops, student_flops)`
   reproduces, in closed form, the total number of teacher (forward-only) and
   student (forward+backward) UNet calls that `train_ppo_coco.py` issues over
   a full run for a given config, and multiplies by the per-call FLOP costs
   measured in step 1 for the SAME model instance the baseline is about to
   train with (same architecture => calibration is exact, not an estimate).
   This is the "target" every baseline is matched against.

   NOTE: this call-count formula mirrors train_ppo_coco.py's loop structure
   as of this fix. If that script's loop structure changes, update the
   formula below to match - it is intentionally written in one place instead
   of re-derived independently in every baseline so there is exactly one
   spot to keep in sync.

3. Each baseline computes its OWN "FLOPs per outer iteration" by calibrating
   every distinct call site its training loop actually uses (student
   forward+backward, plus - where applicable - a fake-score network,
   discriminator, EMA teacher, etc.), then calls
   `units_needed(target_flops, flops_per_unit)` to get back an integer
   iteration/epoch/stage-update count, and uses THAT instead of a
   hand-picked constant to decide how long to train.

This keeps the match auditable: every number that goes into it (a call
count, a per-call FLOP cost) is printed at the start of training, so you can
sanity check it against wall-clock time instead of trusting a black box.
"""

import torch
from torch.utils.flop_counter import FlopCounterMode


def safe_get(config, section, key, default):
    """
    Read `config.<section>.<key>`, returning `default` if EITHER the
    sub-config (e.g. `config.gan`, `config.distill`) or the key inside it is
    missing.

    Plain `getattr(config.gan, "use_gan", True)` still crashes with
    AttributeError if `config.gan` itself doesn't exist, because the
    attribute access `config.gan` happens BEFORE getattr's default logic
    ever applies - ml_collections.ConfigDict raises on unknown top-level
    attributes instead of returning None, so the error surfaces at
    `config.gan`, not at the `"use_gan"` lookup. None of the configs in
    config/ define a `gan` or `distill` section, so this is the actual thing
    that makes those sections optional, not the getattr calls around it.
    """
    section_obj = getattr(config, section, None)
    if section_obj is None:
        return default
    return getattr(section_obj, key, default)


# --------------------------------------------------------------------------- #
# Step 1: calibration
# --------------------------------------------------------------------------- #

def calibrate_call(module, make_inputs, backward=True):
    """
    Measure the FLOPs of exactly one forward (+ optional backward) call
    through `module`.

    `make_inputs` is a zero-arg callable that returns either a tuple of
    positional args or a dict of keyword args for `module(...)`, built fresh
    each time (so gradient state from a previous calibration call can't leak
    in). Random values are fine - shapes are all that matter for a FLOP count.
    """
    args = make_inputs()
    kwargs = {}
    if isinstance(args, dict):
        kwargs = args
        args = ()
    elif not isinstance(args, tuple):
        args = (args,)

    with FlopCounterMode(display=False) as fc:
        out = module(*args, **kwargs)
        out = getattr(out, "sample", out)
        if backward:
            out.float().pow(2).mean().backward()
    return fc.get_total_flops()


def unet_dummy_inputs(unet, batch_size, cross_attention_dim, seq_len=77,
                       device="cuda", dtype=torch.float32, requires_grad=False):
    """Standard (latents, timesteps, encoder_hidden_states) input factory for
    any Stable-Diffusion-style `UNet2DConditionModel.forward` call."""
    def make():
        cfg = unet.config
        latents = torch.randn(
            batch_size, cfg.in_channels, cfg.sample_size, cfg.sample_size,
            device=device, dtype=dtype, requires_grad=requires_grad,
        )
        timesteps = torch.randint(0, 1000, (batch_size,), device=device).long()
        encoder_hidden_states = torch.randn(
            batch_size, seq_len, cross_attention_dim, device=device, dtype=dtype,
        )
        return (latents, timesteps, encoder_hidden_states)
    return make


def calibrate_unet_call(unet, batch_size, cross_attention_dim, seq_len=77,
                         device="cuda", dtype=torch.float32, backward=True):
    """Convenience wrapper: calibrate one `unet(latents, t, encoder_hidden_states)`
    call. Set backward=False for teacher / rollout calls that always run
    under torch.no_grad() in the real training loop, and backward=True for
    the trainable student call (LoRA or full UNet - same forward graph size
    either way; only the parameters that receive gradients differ, backward
    compute is dominated by activations, not by how many of them are
    trainable, so this is the right cost class for both cases)."""
    return calibrate_call(
        unet,
        unet_dummy_inputs(unet, batch_size, cross_attention_dim, seq_len, device, dtype,
                           requires_grad=backward),
        backward=backward,
    )


# --------------------------------------------------------------------------- #
# Step 2: the reference (train_ppo_coco.py) budget, in closed form
# --------------------------------------------------------------------------- #

def reference_budget_flops(ref_config, teacher_flops_per_call, student_flops_per_call):
    """
    Total FLOPs `train_ppo_coco.py` spends over its full configured run.

    Call-count accounting (mirrors that script's loop - see module docstring):
      per (epoch, batch-in-epoch):
        - `sample.num_steps`   teacher forward calls   (no_grad rollout)
        - `student.num_steps`  student forward calls   (no_grad rollout,
                                same shape/cost class as the teacher call)
        - `train.num_inner_epochs * num_train_timesteps` student
          forward+backward calls, where
          num_train_timesteps = int(student.num_steps * train.timestep_fraction)
      repeated for `num_epochs * sample.num_batches_per_epoch` (epoch, batch)
      pairs.
    """
    c = ref_config
    num_train_timesteps = int(c.student.num_steps * c.train.timestep_fraction)
    pairs = c.num_epochs * c.sample.num_batches_per_epoch

    teacher_calls = pairs * c.sample.num_steps
    student_rollout_calls = pairs * c.student.num_steps
    student_train_calls = pairs * c.train.num_inner_epochs * num_train_timesteps

    return (
        teacher_calls * teacher_flops_per_call
        + student_rollout_calls * teacher_flops_per_call
        + student_train_calls * student_flops_per_call
    )


# --------------------------------------------------------------------------- #
# Step 3: convert a target FLOP budget into "how many iterations do I run"
# --------------------------------------------------------------------------- #

def units_needed(target_flops: float, flops_per_unit: float) -> int:
    if flops_per_unit <= 0:
        raise ValueError("flops_per_unit must be positive - calibration failed or returned 0.")
    return max(1, int(round(target_flops / flops_per_unit)))


def load_reference_config(path_or_module="config/distill_clip.py"):
    """Load a ml_collections config file by path, independent of whichever
    `--config` this baseline itself was launched with, so the compute target
    always reflects the SAME config used for the actual ReDiF PPO run rather
    than the baseline's own (possibly different) hyperparameters."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_reference_config_module", path_or_module)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_config()


def report(logger_print, target_flops, flops_per_unit, unit_name, n_units):
    logger_print(
        f"[flop-match] target budget = {target_flops:.4e} FLOPs "
        f"(= scripts/train_ppo_coco.py's configured run)\n"
        f"[flop-match] this baseline costs ~{flops_per_unit:.4e} FLOPs per {unit_name} "
        f"-> running {n_units} {unit_name}(s) to match."
    )