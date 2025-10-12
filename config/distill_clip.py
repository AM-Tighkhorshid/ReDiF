# File: config/distill_clip.py
import ml_collections

# from huggingface_hub import snapshot_download

# snapshot_download(
#     "stabilityai/stable-diffusion-xl-base-1.0",
#     resume_download=True,
#     max_workers=1,   # fewer parallel threads → less chance of timeout
# )

def get_config():
    config = ml_collections.ConfigDict()

    ###### General ######
    config.run_name = ""
    config.seed = 42
    config.logdir = "logs"
    config.num_epochs = 20
    config.save_freq = 20
    config.num_checkpoint_limit = 5
    config.mixed_precision = "no"
    config.allow_tf32 = True
    config.resume_from = ""
    config.use_lora = True
    config.student_output_dir = "output/distill_clip/student_model/"
    config.teacher_output_dir = "output/distill_clip/teacher_model/"


    ###### Pretrained Teacher Model ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    
    config.training_mode = "prompt"        # Stable Diffusion text-prompt
    # pretrained.model = "stabilityai/stable-diffusion-xl-base-1.0"
    pretrained.model = "/media/external20/amirhossein_tighkhorshid/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
    # pretrained.model = "stabilityai/stable-diffusion-xl-base-1.0"

    # config.training_mode = "unconditional" # CIFAR-10 unconditional
    # pretrained.model = "google/ddpm-cifar10-32"
    # config.use_lora = False
    
    pretrained.revision = "main"

    ###### Student Model (inherits from teacher if not changed) ######
    config.student = student = ml_collections.ConfigDict()
    student.model = pretrained.model
    student.revision = pretrained.revision
    student.num_steps = 5

    ###### Sampling ######
    config.sample = sample = ml_collections.ConfigDict()
    sample.num_steps = 50
    sample.eta = 1.0
    sample.guidance_scale = 5.0
    sample.batch_size = 1
    # sample.num_batches_per_epoch = 2

    ###### Training ######
    config.train = train = ml_collections.ConfigDict()
    train.batch_size = 1
    train.use_8bit_adam = False
    train.learning_rate = 3e-4
    train.adam_beta1 = 0.9
    train.adam_beta2 = 0.999
    train.adam_weight_decay = 1e-4
    train.adam_epsilon = 1e-8
    train.gradient_accumulation_steps = 1
    train.max_grad_norm = 1.0
    train.num_inner_epochs = 1
    train.cfg = True
    train.adv_clip_max = 5
    train.clip_range = 1e-4
    train.timestep_fraction = 1.0

    output_dir = "output/distill_clip"

    ###### Prompt Function ######
    config.prompt_fn = "imagenet_animals"
    config.prompt_fn_kwargs = {}

    ###### Reward Function ######
    # config.reward_fn = "clip_similarity"

    ###### Per-Prompt Stat Tracking ######
    config.per_prompt_stat_tracking = ml_collections.ConfigDict()
    config.per_prompt_stat_tracking.buffer_size = 16
    config.per_prompt_stat_tracking.min_count = 16

    # ... other config ...
    train.kl_lambda = 0.2  # or another value for KL loss strength

    config.progressive_steps = [50, 25, 12, 5]


    # ... other configurations ...

    config.train.learning_rate = 1e-5
    # ... other training configs ...

    # --- ADD THESE NEW VARIABLES ---
    # Epsilon for PPO clipping, from Equation (3)
    config.train.clip_epsilon = 0.2 
    # Beta for KL penalty, from Equation (3). The paper uses 0.04.
    config.train.kl_beta = 0.04 
    # -----------------------------


# ----- Progressive Distillation -----

    # --- existing content (keep everything you already have) ---
    # ...
    config.train.learning_rate = 1e-5
    # ...

    # ===== ADD THIS NEW SECTION =====
    config.distill = ml_collections.ConfigDict()

    # Number of sampling steps per stage of progressive distillation
    # (the paper uses this halving schedule)
    config.distill.steps_list = [50, 25, 12, 5]

    # Number of gradient updates (iterations) for each distillation stage
    # You can increase to 50000+ for high-quality results;
    # smaller values like 5000 are good for debugging.
    config.distill.updates_per_stage = 5

    # =================================

    return config
