# ReDiF: Reinforced Distillation for Few-step Diffusion

Step distillation of diffusion models framed as **terminal-reward policy optimisation**. A
few-step student is initialised from the teacher and fine-tuned with a PPO-style clipped
surrogate, where the reward is the semantic agreement between the student's image and the
teacher's image generated from the *same* initial noise and the *same* conditioning.

The method is **data-free**: no paired dataset is required, only the teacher and a pool of
(noise, condition) pairs.

Two branches are implemented and kept deliberately symmetric:

| Branch | Teacher | Data | Student | Entry point |
|---|---|---|---|---|
| SD / COCO | Stable Diffusion v1.5, 50-step DDIM | COCO-2017 captions | 5-step DDIM | `scripts/train_ppo_coco.py` |
| EDM / ImageNet | EDM ADM, 256-step Heun (511 NFE) | class-conditional ImageNet-64 | 4/8/16-step Euler | `scripts/train_ppo_imagenet_edm.py` |

Each branch ships a **direct reward-backpropagation** counterpart (DRaFT / AlignProp style, no
RL) that is identical in every respect except the gradient estimator. That is the controlled
ablation for "how much does the RL formulation actually contribute?".

---

## Repository layout

```
config/
  distill_clip.py              ml_collections config for the SD/COCO branch
ddpo_pytorch/
  rewards.py                   reward registry (clip, dino, perception, lpips, mse, kl, ...)
  clip_distill.py              CLIP image-image distillation reward
  stat_tracking.py             per-prompt advantage whitening
  diffusers_patch/
    pipeline_with_logprob.py   DDIM sampling that also returns per-step log-probabilities
    ddim_with_logprob.py       the DDIM step, with log-prob of the transition
scripts/
  train_ppo_coco.py            SD/COCO   -- RL (PPO)
  train_draft_coco.py          SD/COCO   -- direct reward backprop (control arm)
  evaluate.py                  SD/COCO   -- FID / CLIP score / PRDC
  train_ppo_imagenet_edm.py    EDM/IN64  -- RL (PPO), self-contained
  train_draft_imagenet_edm.py  EDM/IN64  -- direct reward backprop (control arm)
  eval_imagenet_edm.py         EDM/IN64  -- FID / PRDC / class consistency
  baselines/                   DMD2, progressive, progressive+adversarial, consistency
```

The EDM scripts import each other as libraries (`eval_imagenet_edm.py` and
`train_draft_imagenet_edm.py` both load `train_ppo_imagenet_edm.py`), as do the SD scripts
(`train_draft_coco.py` loads `train_ppo_coco.py`). Sampler, reward, pool, LoRA, plotting,
checkpoint format and compute accounting therefore have exactly one definition each and cannot
drift between arms. **Replace these files as a set, not individually.**

---

## Installation

```bash
git clone https://github.com/AM-Tighkhorshid/ReDiF && cd ReDiF
pip install -r requirements.txt
pip install fvcore            # exact FLOP accounting (optional but recommended)
```

The EDM branch additionally needs NVIDIA's EDM repository on the import path, because the
released checkpoint is a `torch_utils.persistence` pickle and cannot be unpickled without it:

```bash
git clone https://github.com/NVlabs/edm
# then pass --edm_repo /path/to/edm, or export PYTHONPATH=$PYTHONPATH:/path/to/edm
```

The teacher checkpoint (`edm-imagenet-64x64-cond-adm.pkl`, ~1.2 GB) is downloaded automatically
on first use into `--ckpt_dir`.

---

## Quick start

### EDM / ImageNet-64

```bash
# 1. Generate the teacher targets once. Deterministic Heun, so they depend only on
#    (z_T, class) and can be cached and reused by every later run.
python scripts/train_ppo_imagenet_edm.py --edm_repo /path/to/edm \
    --precompute_only --num_pairs 10000 --teacher_steps 256

# 2. RL distillation into an 8-step student.
python scripts/train_ppo_imagenet_edm.py --edm_repo /path/to/edm \
    --student_steps 8 --reward_fn clip+dino+perception \
    --lora_rank 4 --lr 3e-4 --kl_coeff 0.02 \
    --sample_batch_size 32 --num_batches_per_epoch 1 --num_epochs 100

# 3. Evaluate against real ImageNet-64.
python scripts/eval_imagenet_edm.py --edm_repo /path/to/edm \
    --student_ckpt edm_logs/<run>/steps8/best.pt \
    --real_dir /path/to/imagenet_val_64 --num_real 10000
```

### Stable Diffusion / COCO

```bash
python scripts/train_ppo_coco.py --config config/distill_clip.py --prompt_source coco

python scripts/evaluate.py --dataset coco --split val --num_images 5000 \
    --student_model Results_KL/<run>/student_model
```

### Control arm (no RL)

```bash
python scripts/train_draft_imagenet_edm.py --edm_repo /path/to/edm \
    --student_steps 8 --reward_fn clip+dino+perception \
    --lora_rank 4 --lr 3e-4 --kl_coeff 0 --num_epochs 100

python scripts/train_draft_coco.py --config config/distill_clip.py \
    --prompt_source coco --backprop_steps 1
```

---

## Method

### MDP

One denoising trajectory is one episode. The state is the current latent, the action is the next
latent, and the policy is the Gaussian induced by the sampler:

```
pi_theta(x_{t-1} | x_t, c) = N( mu_theta(x_t, t, c), sigma_t^2 I )
```

`sigma_t` is set by `eta` (DDIM) or by the ancestral split `sigma_to^2 = sigma_down^2 +
sigma_up^2` (EDM). `eta = 0` gives a deterministic sampler with a zero-variance policy and hence
**no policy gradient at all**, so training requires `eta > 0` while deployment uses `eta = 0`.

### Reward

Terminal only, evaluated once per trajectory on the final image against the teacher's image for
the same (noise, condition) pair:

| term | description | differentiable |
|---|---|:--:|
| `clip` | CLIP image-image cosine | yes |
| `dino` | DINOv3 patch-token cosine (dense/structural agreement) | yes |
| `perception` | Meta Perception Encoder cosine | yes |
| `lpips` | negative LPIPS perceptual distance | yes |
| `mse` | negative pixel MSE | yes |
| `kl` | negative forward KL over intensity histograms | (trivially) |
| `text_image` | CLIP alignment to the class name / caption (teacher-free) | via argmax: no |
| `aesthetic` | LAION aesthetic score (uint8 input) | **no** |

Terms are combined with `--reward_fn clip+dino` and optionally weighted with `--reward_weights`.
They live on very different scales (cosine ~0.8, `-MSE` ~-0.01, `-LPIPS` ~-0.5), so an unweighted
sum lets one term dominate; per-term values are logged separately so this is visible.

The last two rows are the point of the RL formulation: a direct-backprop method cannot use them
at all.

### Advantage baseline

`A = R - b`. Three options, selected by `--group_size` / `--stat_key`:

* **group-relative (GRPO-style, `--group_size k>1`)** — draw `n/k` distinct conditions and roll
  each out `k` times with different exploration noise. Every member of a group faces the same
  target, so the within-group spread is caused only by the policy. Lowest variance, no warm-up.
* **per-key running mean (DDPO-style, `--group_size 1`)** — a buffer per condition. Requires the
  same condition to recur often; with a large prompt or pair pool the buffer never fills and the
  code falls back to whole-batch statistics.
* **whole-batch** — the fallback.

`--adv_normalize` controls the scaling. `group_std` (the plain GRPO recipe) rescales every group
to unit variance including degenerate ones where all rollouts scored within 1e-4 of each other,
turning noise into full-strength gradient; `global_std` (default) divides by the epoch-wide std
instead so uninformative groups stay small.

---

## Diagnostics

Policy-gradient training on diffusion models fails quietly: the reward curve of a run that
converged and one that never moved look similar. Every run therefore logs the following to
`log.jsonl`, one line per epoch.

| field | what it tells you |
|---|---|
| `ratio_t0` | Importance ratio at the **first** minibatch of the **first** inner epoch, before any optimizer step. Must be `1.000000`. Anything else means the recomputed log-prob comes from a different network than the rollout used. The epoch-mean `ratio` cannot show this, since it mixes in genuine post-update drift. |
| `lora_drift` / `param_drift` | Relative weight change since init. Distinguishes "converged" from "never left the initialisation" — the reward curve cannot. |
| `within_group_std` | Spread of reward *within* a group. If it is ~0, the exploration noise produces no measurable reward difference and there is nothing to learn; raise `--eta`. |
| `eval_reward` | Deterministic (`eta=0`) reward on held-out pairs. Training reward is measured under exploration noise and is far too noisy to select checkpoints on. `best.pt` is selected on this. |
| `clipfrac` | Fraction of samples outside the clip range. Exactly 0 means the clipped surrogate is inert and the update is plain REINFORCE. |
| `opt_steps` | Optimizer steps per epoch. Easy to be off by 10x without noticing. |
| `nfe_cum`, `pflops_cum` | Cumulative compute (see below). |
| `unique_pairs` | Distinct conditions seen this epoch. |

`reward_curve.pdf`, `loss_curve.pdf` and `drift_curve.pdf` are rewritten every epoch, so a killed
run still leaves usable figures.

---

## Compute accounting

Both branches track training FLOPs. The per-forward cost is measured once at startup with fvcore;
the training loop only increments integer counters, so there is no measurable effect on
throughput. Written to `flops.json` every epoch.

Conventions, stated because they are where papers silently disagree:

* FLOPs = 2 x MACs.
* Backward is charged 2x the forward (standard approximation, not measured).
* Counts are per **image**, so they do not change with batch size.
* NFE counts diffusion-network calls only. **With classifier-free guidance every sampler step is
  two UNet calls**; that factor is included. An NFE number reported without saying whether CFG is
  included is off by 2x.
* Reward encoders and the VAE decoder are counted separately — not part of anyone's NFE budget,
  but not free either.

Two things the numbers make obvious. In the EDM branch the one-off teacher cache
(`num_pairs x (2 x teacher_steps - 1)`) dominates the total, so report the marginal RL cost
separately. In the SD branch the teacher's 50-step CFG rollout is repeated **every epoch** rather
than cached, and is the single largest term.

---

## Reproducibility mechanics

**Run directories are versioned.** Re-running the same configuration writes to `<dir>_run2`,
`_run3`, ... instead of appending into the previous run's `log.jsonl` and overwriting its
`best.pt`. Since the epoch counter restarts at 0, a merged log cannot be separated afterwards.
Disabled when resuming (`--resume_from`), where continuing in place is correct.

**Every run writes `config.json`** with the complete argument set. Two runs that behaved
differently are diagnosed by diffing those files, not by reading code:

```bash
diff <(python -m json.tool logs/<old>/config.json) <(python -m json.tool logs/<new>/config.json)
```

**Evaluation results are cached and keyed by protocol.** `metrics_<ckpt>_<protocol>.json` plus an
append-only `eval_log.jsonl` and a human-readable `eval_results.txt`, all next to the checkpoint.
Re-evaluating an already-measured (checkpoint, dataset, split, n, steps) combination loads from
disk; `--force` recomputes. Each record carries the full protocol so a number is interpretable
months later.

**The teacher cache is protected.** If an existing cache belongs to a different pool
(`pool_seed`, `num_pairs`), the new one is written to a suffixed filename rather than replacing
hours of teacher sampling. `--overwrite_cache 1` forces.

---

## Evaluation

```bash
python scripts/eval_imagenet_edm.py --student_ckpt <ckpt> --edm_repo <edm> \
    --real_dir <imagenet_val_64> --num_real 10000
```

Reports:

* **FID** for the student *and* the teacher against the same reference, plus **`FID_gap`**. The
  gap isolates what distillation cost, as opposed to what the teacher is worth.
* **PRDC** (precision / recall / density / coverage) on DINOv3 features. FID alone hides mode
  collapse, which is the failure mode RL fine-tuning is most prone to. If reward rises while
  recall falls, that is reward hacking, not progress.
* **`class_acc`** — top-1 agreement of a pretrained classifier with the conditioning label
  (EDM branch).
* **`paired_*`** — the training reward terms, measured on the evaluation set.

Two protocol details that quietly change FID: real images are shuffled with a fixed seed before
subsampling (a recursive glob over ImageNet returns class-sorted paths, so taking the first N
gives a handful of classes), and downsampling uses a **box** filter to match the standard
downsampled-ImageNet-64 protocol the teacher was trained on. Both are recorded in the metrics
file.

Use `--source fresh` for held-out (noise, class) pairs; the default `cache` reuses the training
pool, which measures fit rather than generalisation and says so in the output.

---

## Known issues and gotchas

These are documented in-code where they occur. Listed here because each cost real debugging time.

**EDM is not compatible with `torch.autocast`.** `EDMPrecond.forward` ends with
`assert F_x.dtype == dtype`, decided by the network's own `use_fp16` flag. Under autocast the
inner model returns bf16 while the precond expects fp32 and the assertion fires. EDM ships its
own mixed precision (`weight.to(x.dtype)` per layer); use `--teacher_fp16` / `--student_fp16`.

**The EDM ImageNet-64 checkpoint was trained with dropout=0.10.** The student must stay in
`eval()` mode during the PPO update. `train()` recomputes log-probs through a dropout-perturbed
network, so the importance ratio is biased rather than 1 and the update optimises a network that
neither the rollout nor the evaluation uses. SD's UNet has no dropout, which is why this only
affects the EDM branch. `requires_grad` is what controls gradient flow, not the training flag.

**Reseeding the global RNG for the sample grid silently collapses training.** The eval grid uses
a fixed seed for comparability; the RNG state must be saved and restored around it, or the next
epoch's rollout replays the same pool indices and the same exploration noise.

**Per-step gradient magnitude is wildly unequal.** The log-prob gradient at step `t` scales like
`1/sigma_up[t]`, and `sigma_up` spans ~20 down to ~0.001 over a Karras schedule — a ~10^4 ratio.
Unweighted, the last steps consume the entire gradient-norm budget and the early steps never
train. `--step_weighting sigma` equalises this.

**`eta` is not just a sharpness knob.** `eta=1` drives `sigma_down` to ~0, so each step discards
the deterministic ODE direction entirely ("denoise fully, then renoise") and the sampler
converges toward the posterior mean, which is blurry and structurally unstable at few steps.
0.2–0.5 keeps the trajectory near the ODE path while leaving enough exploration for the policy
gradient. On latent models the VAE decoder hides most of this, so inspect the latent-space effect
rather than the decoded image.

**LoRA is not only a memory optimisation here.** The alignment of a score-function gradient with
the true gradient scales like `sqrt(n_samples / n_params)`. With ~10^2 samples per epoch and
3x10^8 parameters that ratio is ~10^-3, i.e. essentially noise — and Adam takes a step of size
`lr` regardless, so the weights random-walk away from a good initialisation. Shrinking the
trainable set is the cheapest way to raise the signal-to-noise ratio, and zero-initialised `B`
makes it an implicit trust region as well.

**`latent_kl` in the SD branch contributes no gradient.** It is computed inside the no-grad
rollout, so `kl_coef` scales a constant. `kl0.0` and `kl1.0` runs are the same experiment. The
DRaFT arm computes it with a gradient, so set `kl_coef = 0` there to match.

**Guidance consistency of the importance ratio.** The SD rollout samples with classifier-free
guidance, but the PPO update recomputed log-probs from the conditional branch alone. That is not
a constant offset: the ratio then sits systematically below 1, which under the clipped surrogate
zeroes the gradient for negative-advantage samples while keeping it for positive ones — an
implicit positive-advantage filter, i.e. reward-weighted regression rather than PPO. Both
variants are selectable via `config.train.cfg` and reported as an ablation; `ratio_t0` measures
the discrepancy directly.

**The SD reward is computed on washed-out pixels.** `pipeline_with_logprob` applies
`clamp(0,1)` to the VAE output and then hands it to `image_processor.postprocess`, which
denormalises assuming a [-1,1] input. Composed, the decoder's range is mapped into **[0.5, 1]**.
The qualitative eval (standard `__call__`) is unaffected, so training and inspection disagree.
`train_draft_coco.py` replicates this by default (`--match_pipeline_decode`) so the arms stay
comparable; removing the `clamp(0,1)` line fixes both at once.

---

## Citation

```bibtex
@inproceedings{redif2026,
  title     = {ReDiF: Reinforced Distillation for Few-step Diffusion},
  author    = {Tighkhorshid, Amirhossein and others},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```

Built on [ddpo-pytorch](https://github.com/kvablack/ddpo-pytorch) (Black et al., DDPO),
[NVlabs/edm](https://github.com/NVlabs/edm) (Karras et al., 2022) and
[diffusers](https://github.com/huggingface/diffusers).
