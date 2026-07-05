"""
config/distill_clip.py - configuration for ReDiF teacher -> student CLIP
distillation via PPO (see scripts/train_ppo_coco.py).

Everything that the training script actually reads lives here. Anything
that used to be defined but silently ignored by the training script (a GAN
sub-config, a duplicated `learning_rate` assignment, a `clip_epsilon` /
`kl_beta` pair that the script never referenced) has been removed - dead
config is worse than no config, because it looks like a knob that does
something when it does not.
"""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    ###### General ######
    config.run_name = ""
    config.seed = 44
    config.logdir = "logs"
    config.num_epochs = 20
    config.save_freq = 5             # epochs between accelerator.save_state() checkpoints
    config.num_checkpoint_limit = 5
    config.mixed_precision = "no"
    config.allow_tf32 = True
    config.resume_from = ""          # path to a `checkpoint_N` dir, or a dir containing them
    config.use_lora = True
    config.student_output_dir = "output/distill_clip/student_model/"
    config.teacher_output_dir = "output/distill_clip/teacher_model/"

    ###### Pretrained teacher model ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    # IMPORTANT: point this at a LOCAL snapshot directory you already have
    # fully downloaded, not a bare Hub repo id like "runwayml/stable-diffusion-v1-5".
    # A bare repo id makes diffusers contact the Hub first; if that fails
    # (expired token, no network, repo gated/moved) it silently falls back to
    # the *default* ~/.cache/huggingface location, which may be a different,
    # incomplete cache than the one you actually populated. Using an absolute
    # local path skips the Hub entirely and always loads from exactly this
    # directory. Adjust the path below to wherever your snapshot actually is.
    pretrained.model = (
        "/media/external20/amirhossein_tighkhorshid/models--runwayml--stable-diffusion-v1-5/"
        "snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
    )
    pretrained.revision = "main"

    ###### Student model (inherits from teacher unless overridden) ######
    config.student = student = ml_collections.ConfigDict()
    student.model = pretrained.model  # same local snapshot by default; override if student starts elsewhere
    student.revision = pretrained.revision
    student.num_steps = 5            # denoising steps the student is trained to use

    ###### Sampling (applies to the teacher's 50-step rollout AND, for
    ###### everything except num_steps, to the student's rollout) ######
    config.sample = sample = ml_collections.ConfigDict()
    sample.num_steps = 50            # teacher denoising steps
    sample.eta = 1.0
    sample.guidance_scale = 5.0
    sample.batch_size = 8            # per-GPU rollout batch size
    # Number of rollout batches to gather before each optimizer step. This
    # directly determines Accelerate's gradient_accumulation_steps together
    # with `student.num_steps` - see scripts/train_ppo_coco.py for the exact
    # relationship. Kept >= 1; do not set independently without re-reading
    # that section.
    sample.num_batches_per_epoch = 1

    # Whether teacher and student must start each rollout from the same
    # initial latent noise. This MUST be True for ReDiF distillation to be
    # meaningful: if teacher and student denoise different noise samples,
    # the student is being asked to imitate an image that has nothing to do
    # with its own trajectory. See `sample_shared_latents` in the training
    # script.
    sample.sync_initial_latent = True

    ###### Training ######
    config.train = train = ml_collections.ConfigDict()
    train.batch_size = 8             # must divide sample.batch_size * num_batches_per_epoch
    train.use_8bit_adam = False
    train.learning_rate = 1e-5
    train.adam_beta1 = 0.9
    train.adam_beta2 = 0.999
    train.adam_weight_decay = 1e-4
    train.adam_epsilon = 1e-8
    # Number of rollout batches accumulated into one optimizer step (in
    # addition to accumulating over student.num_steps timesteps, which
    # happens unconditionally). Effective batch size per optimizer step is
    # sample.batch_size * num_batches_per_epoch * num_processes.
    train.gradient_accumulation_steps = 1
    train.max_grad_norm = 1.0
    # Number of PPO passes over the SAME collected rollout before resampling.
    # >1 is what makes the clipped surrogate objective meaningful (on the
    # very first pass the ratio is always ~1 by construction).
    train.num_inner_epochs = 1
    train.cfg = True
    train.adv_clip_max = 5.0
    train.clip_range = 1e-4          # PPO importance-ratio clip range (DDPO default)
    train.timestep_fraction = 1.0    # fraction of student.num_steps trained on per inner epoch
    # Weight of the analytic latent-space KL(student || teacher) regularizer
    # added to the PPO loss. 0 disables it.
    train.kl_coef = 0.0

    ###### Prompts ######
    config.prompt_fn = "imagenet_animals"
    config.prompt_fn_kwargs = {}

    ###### Reward ######
    # Any combination of: clip, dino, perception, text_image, aesthetic, mse.
    # "clip" (student<->teacher CLIP cosine similarity) is the reward used by
    # the ReDiF paper; the others are optional auxiliary signals.
    config.reward_types = ["clip"]

    ###### Per-prompt stat tracking ######
    # Advantages are whitened using a running per-prompt mean/std instead of
    # only the current (small) rollout batch, which is far lower-variance
    # once `min_count` prompts have been seen. Falls back to whole-batch
    # statistics until then.
    config.per_prompt_stat_tracking = ml_collections.ConfigDict()
    config.per_prompt_stat_tracking.buffer_size = 16
    config.per_prompt_stat_tracking.min_count = 16

    ###### Evaluation (fixed across the whole run for a fair before/after
    ###### comparison - see scripts/train_ppo_coco.py) ######
    config.eval = ml_collections.ConfigDict()
    config.eval.seed = 12345
    config.eval.prompts = [
        "A crystal-clear glass bowl overflowing with ripe oranges on a rustic wooden table",
        "A fluffy tabby cat mid-step, looking at the camera with curious eyes",
        "A futuristic city skyline at night with neon lights and flying cars",
        "A warm log cabin in a snowy pine forest at twilight",
        "A colorful wildflower plain under a bright sky",
    ]
    config.eval.freq = 1  # run + save eval images every N epochs

    return config