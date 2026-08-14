"""
ReDiF ablation -- direct reward backpropagation instead of RL, on EDM / ImageNet-64.

This is the controlled counterpart of `train_ppo_imagenet_edm.py`. Same teacher, same cached
targets, same (z_T, class) pool, same reward encoders, same LoRA setup, same sampler, same
schedule, same logging, same checkpoint format. The ONLY thing that changes is how the reward
turns into a parameter update:

    PPO      reward is a black-box scalar; the update is (importance ratio) x (advantage), i.e.
             a score-function / REINFORCE estimator. One scalar of information per sample.
    DRaFT    the reward encoders are differentiable, so the reward is backpropagated straight
             through the sampling trajectory into the student. A full gradient per sample.

Both optimise the same objective -- alignment between the student's image and the teacher's image
for the same (z_T, class) -- so the difference in results is attributable to the estimator, which
is exactly the question this script exists to answer.

Everything is imported from the PPO script rather than copied, so the reward definitions, the
sampler, the pool and the compute accounting cannot drift apart between the two arms.

Why it is the "same MDP", not a different objective
--------------------------------------------------
The stochastic policy is unchanged: x_{i+1} = mu_theta(x_i) + sigma_up * z. During a rollout z is
drawn and then held fixed. REINFORCE differentiates log pi(x_{i+1} | x_i) and multiplies by the
reward; DRaFT differentiates the reward through mu_theta with z treated as a constant -- the
reparameterisation/pathwise derivative of the same quantity. Keep --eta identical across the two
arms and the sampling distributions are identical too.

Memory
------
Backpropagating through N sampling steps stores activations for N UNet forwards. With 8 steps at
batch 32 this is the binding constraint, not compute. Two standard mitigations, both available:
--backprop_steps K truncates the gradient to the last K steps (this is DRaFT-K; K=1 is what the
DRaFT paper mainly uses), and --grad_checkpoint recomputes activations instead of storing them.

Usage (matched against a PPO run)
---------------------------------
  python train_draft_imagenet_edm.py --edm_repo ../../edm \\
      --student_steps 8 --reward_fn clip+dino+perception \\
      --lora_rank 4 --lr 3e-4 --eta 1.0 --kl_coeff 0 \\
      --sample_batch_size 32 --num_batches_per_epoch 1 --num_epochs 100
"""

import argparse
import copy
import importlib.util
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def load_train_module():
    """Import the PPO trainer as a library so both arms share one definition of everything."""
    path = os.path.join(HERE, "train_ppo_imagenet_edm.py")
    spec = importlib.util.spec_from_file_location("redif_train", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["redif_train"] = mod
    spec.loader.exec_module(mod)
    return mod


T = load_train_module()


# -----------------------------------------------------------------------------------------------
# Args -- deliberately the same names and defaults as the PPO script wherever they mean the same
# thing, so two config.json files can be diffed directly.
# -----------------------------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()

    # --- models (identical) --------------------------------------------------------------------
    p.add_argument("--teacher_pkl", type=str,
                   default="https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl")
    p.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    p.add_argument("--edm_repo", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--teacher_fp16", type=int, default=1)
    p.add_argument("--student_fp16", type=int, default=0)

    # --- schedule (identical) ------------------------------------------------------------------
    p.add_argument("--student_steps", type=str, default="8")
    p.add_argument("--teacher_steps", type=int, default=256)
    p.add_argument("--sigma_min", type=float, default=0.002)
    p.add_argument("--sigma_max", type=float, default=80.0)
    p.add_argument("--rho", type=float, default=7.0)
    p.add_argument("--eta", type=float, default=1.0,
                   help="Same sampler as the PPO arm. The injected noise z is drawn and then held "
                        "constant while the gradient flows through the mean -- the pathwise "
                        "derivative of exactly the policy PPO differentiates by score function. "
                        "Set this to whatever the PPO run used, or the two arms are not comparable.")

    # --- pool + cache (identical) --------------------------------------------------------------
    p.add_argument("--num_pairs", type=int, default=1000)
    p.add_argument("--num_classes_used", type=int, default=1000)
    p.add_argument("--pool_seed", type=int, default=1234)
    p.add_argument("--active_pairs", type=int, default=0)
    p.add_argument("--teacher_cache", type=str, default="./teacher_cache_in64.pt")
    p.add_argument("--overwrite_cache", type=int, default=0)
    p.add_argument("--reuse_cache_pool", type=int, default=1)
    p.add_argument("--online_teacher", type=int, default=0)
    p.add_argument("--precompute_only", action="store_true")
    p.add_argument("--precompute_batch", type=int, default=64)

    # --- optimisation loop shape (identical) ---------------------------------------------------
    p.add_argument("--num_epochs", type=int, default=20)
    p.add_argument("--sample_batch_size", type=int, default=16)
    p.add_argument("--num_batches_per_epoch", type=int, default=4,
                   help="Gradients are accumulated across all batches of an epoch and ONE optimizer "
                        "step is taken, matching --ppo_style coco. Same samples/epoch and same "
                        "optimizer steps/epoch as the PPO arm.")
    p.add_argument("--lr", type=float, default=1e-5,
                   help="The one hyper-parameter that genuinely should be tuned per arm: a dense "
                        "gradient and a REINFORCE estimator do not share an optimal step size. "
                        "Tune each arm's lr on the held-out reward, then compare the best of each.")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--adam_eps", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--lora_rank", type=int, default=0)
    p.add_argument("--lora_alpha", type=float, default=0)
    p.add_argument("--lora_targets", type=str, default="qkv|proj|affine")
    p.add_argument("--trainable_regex", type=str, default="")

    # --- direct-backprop specific --------------------------------------------------------------
    p.add_argument("--backprop_steps", type=int, default=0,
                   help="DRaFT-K: gradient flows only through the LAST K sampling steps; the earlier "
                        "ones run under no_grad. 0 = all steps (most faithful to what PPO trains, "
                        "since PPO updates every timestep, but N times the activation memory). K=1 "
                        "is the DRaFT paper's main variant and is dramatically cheaper.")
    p.add_argument("--grad_checkpoint", type=int, default=1,
                   help="Recompute each step's activations in the backward pass instead of storing "
                        "them. Cuts activation memory from O(K) forwards to O(1) at the cost of one "
                        "extra forward per step -- which is charged to the FLOP accounting.")
    p.add_argument("--kl_coeff", type=float, default=0.0,
                   help="Weight of 0.5*||x_student - x_teacher||^2. NOTE: in the PPO script this term "
                        "is computed inside the no-grad rollout and therefore contributes NO gradient "
                        "-- it is inert there whatever its coefficient. Here it would be a real "
                        "gradient, so leaving it at 0 is what actually matches the PPO arm.")

    # --- reward (identical) --------------------------------------------------------------------
    p.add_argument("--reward_fn", type=str, default="clip+dino")
    p.add_argument("--reward_weights", type=str, default=None)
    p.add_argument("--perception_model", type=str, default="vit_pe_core_base_patch16_224.fb")
    p.add_argument("--lpips_resolution", type=int, default=64)
    p.add_argument("--kl_bins", type=int, default=32)
    p.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--dino_model", type=str, default="vit_base_patch16_dinov3.lvd1689m")
    p.add_argument("--dino_tokens", type=str, default="patch", choices=["patch", "cls"])
    p.add_argument("--reward_resolution", type=int, default=224)

    # --- logging / io (identical) --------------------------------------------------------------
    p.add_argument("--output_dir", type=str, default="./edm_logs")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--num_checkpoint_limit", type=int, default=5)
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--sample_every", type=int, default=1)
    p.add_argument("--eval_reward_n", type=int, default=64)
    p.add_argument("--eval_eta", type=float, default=0.0)
    p.add_argument("--eval_seed", type=int, default=12345)
    p.add_argument("--unique_run_dir", type=int, default=1)
    p.add_argument("--count_flops", type=int, default=1)
    p.add_argument("--plot_band", type=str, default="std", choices=["sem", "std", "none"])
    p.add_argument("--allow_tf32", type=int, default=1)
    p.add_argument("--seed", type=int, default=24)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="redif-edm-imagenet")

    return p.parse_args()


# -----------------------------------------------------------------------------------------------
# Differentiable sampling + reward
# -----------------------------------------------------------------------------------------------


def student_sample_with_grad(net, latents, labels, sigmas, eta, backprop_steps=0,
                             grad_checkpoint=True, budget=None, n_images=0):
    """Run the student's sampler, keeping the graph for the last `backprop_steps` transitions.

    Identical trajectory distribution to `student_rollout` in the PPO script -- same schedule, same
    ancestral split, same noise magnitude. The difference is only that the graph is retained, and
    that the noise is treated as a constant (reparameterisation) so the gradient reaches mu_theta.
    """
    n_steps = len(sigmas) - 1
    k = n_steps if backprop_steps <= 0 else min(backprop_steps, n_steps)
    first_grad_step = n_steps - k

    x = latents * sigmas[0]
    for i in range(n_steps):
        s_from, s_to = float(sigmas[i]), float(sigmas[i + 1])
        s_down, s_up = T.ancestral_coeffs(s_from, s_to, eta)
        use_grad = i >= first_grad_step

        def step(x_in):
            mean, _ = T.edm_step_mean(net, x_in, s_from, s_down, labels)
            return mean.float()

        if use_grad:
            if grad_checkpoint:
                mean = torch.utils.checkpoint.checkpoint(step, x, use_reentrant=False)
            else:
                mean = step(x)
            if budget is not None:
                budget.add("unet", n_images, backward=True, is_nfe=True)
                if grad_checkpoint:
                    budget.add("unet", n_images)   # the recomputation forward
        else:
            with torch.no_grad():
                mean = step(x)
            if budget is not None:
                budget.add("unet", n_images, is_nfe=True)

        # z is sampled and then held constant: this is the pathwise derivative of the same policy
        # PPO differentiates through its log-probability.
        x = mean if s_up <= 0 else mean + s_up * torch.randn_like(mean)
    return x


def differentiable_reward(bank, imgs_pm1, teacher_u8, classes=None):
    """The reward of `RewardBank.__call__`, term for term, but with the graph kept.

    `RewardBank.__call__` is wrapped in @torch.no_grad() because in PPO the reward is a black-box
    scalar. The per-term helpers it calls are not, so this reuses them directly -- the numbers are
    identical to the PPO arm's reward by construction, not by reimplementation.

    The teacher side is a constant, so it is encoded under no_grad; only the student side carries
    gradient. Note the clamp inside `_prep`: pixels driven outside [0,1] receive zero gradient,
    which is the standard DRaFT behaviour and is also what keeps the reward values comparable.
    """
    s01 = (imgs_pm1.clamp(-1, 1) + 1) / 2
    t01 = T.to_unit(teacher_u8).to(bank.device)

    with torch.no_grad():
        t_emb = {}
        if "clip" in bank.terms:
            t_emb["clip"] = bank._embed_clip(t01)
        if "dino" in bank.terms:
            t_emb["dino"] = bank._embed_dino(t01)
        if "perception" in bank.terms:
            t_emb["perception"] = bank._embed_pe(t01)

    parts = {}
    for term in bank.terms:
        if term == "clip":
            parts[term] = bank._cosine(bank._embed_clip(s01), t_emb["clip"])
        elif term == "dino":
            parts[term] = bank._cosine(bank._embed_dino(s01), t_emb["dino"])
        elif term == "perception":
            parts[term] = bank._cosine(bank._embed_pe(s01), t_emb["perception"])
        elif term == "lpips":
            parts[term] = bank._term_lpips(s01, t01)
        elif term == "mse":
            parts[term] = bank._term_mse(s01, t01)
        elif term == "kl":
            parts[term] = bank._term_kl(s01, t01)
        elif term == "text_image":
            if classes is None:
                raise ValueError("text_image reward needs the class ids of the batch.")
            img_e = bank._embed_clip(s01)
            parts[term] = (img_e * bank.text_embeds[classes.to(bank.device)]).sum(-1)
        elif term == "aesthetic":
            raise ValueError("the aesthetic scorer takes uint8 input and is not differentiable "
                             "here; drop it from --reward_fn for the direct-backprop arm.")

    total = sum(bank.weights[t] * parts[t] for t in bank.terms)
    return total, {k: v.detach().float().cpu() for k, v in parts.items()}


# -----------------------------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------------------------


def train_student(args, num_steps, base_net, pool, teacher_images, reward_fn, device):
    run_dir = T.unique_run_dir(os.path.join(args.output_dir, args.run_name, f"steps{num_steps}"),
                               enabled=bool(args.unique_run_dir) and not args.resume_from)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[run steps={num_steps}] writing to {os.path.abspath(run_dir)}")
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "num_steps": num_steps, "method": "draft_direct_backprop"},
                  f, indent=2)

    student = copy.deepcopy(base_net).to(device).eval().requires_grad_(True)
    n_ev = min(args.eval_reward_n, pool.n)
    eval_idx = torch.arange(pool.n - n_ev, pool.n) if n_ev > 0 else None

    # --- trainable parameter selection: byte-for-byte the PPO script's logic -------------------
    if args.lora_rank > 0:
        alpha = args.lora_alpha or args.lora_rank
        names = T.apply_lora(student, args.lora_rank, alpha, args.lora_targets)
        student.to(device)
        student.requires_grad_(False)
        trainable = []
        for n, prm in student.named_parameters():
            keep = n.endswith(".A") or n.endswith(".B")
            prm.requires_grad_(keep)
            if keep:
                trainable.append(prm)
        if not trainable:
            raise ValueError(f"--lora_targets '{args.lora_targets}' matched no wrappable modules.")
        print(f"[run steps={num_steps}] LoRA rank {args.lora_rank} on {len(names)} modules")
    else:
        trainable = list(student.parameters())
    if args.lora_rank == 0 and args.trainable_regex:
        import re
        pat = re.compile(args.trainable_regex)
        trainable = []
        for name, prm in student.named_parameters():
            keep = bool(pat.search(name))
            prm.requires_grad_(keep)
            if keep:
                trainable.append(prm)
        if not trainable:
            raise ValueError(f"--trainable_regex '{args.trainable_regex}' matched no parameters.")
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in student.parameters())
    print(f"[run steps={num_steps}] trainable {n_train / 1e6:.1f}M / {n_total / 1e6:.1f}M params "
          f"({100 * n_train / n_total:.1f}%)")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr,
                                  betas=(args.adam_beta1, args.adam_beta2),
                                  weight_decay=args.weight_decay, eps=args.adam_eps)

    start_epoch = 0
    if args.resume_from:
        ck = torch.load(args.resume_from, map_location="cpu")
        student.load_state_dict(ck["model"])
        if "optimizer" in ck:
            optimizer.load_state_dict(ck["optimizer"])
        start_epoch = int(ck.get("epoch", -1)) + 1
        print(f"[resume] {args.resume_from} -> starting at epoch {start_epoch}")

    best_reward = -float("inf")
    best_epoch = None
    history = defaultdict(list)
    periodic_ckpts = []

    sigmas = T.karras_sigmas(num_steps + 1, args.sigma_min, args.sigma_max, args.rho,
                             device, end_at_sigma_min=True)
    k_bp = num_steps if args.backprop_steps <= 0 else min(args.backprop_steps, num_steps)
    print(f"[run steps={num_steps}] gradient flows through the last {k_bp}/{num_steps} steps"
          f"{' (checkpointed)' if args.grad_checkpoint else ''}")

    budget = T.FlopBudget(enabled=bool(args.count_flops))
    if budget.enabled:
        probe_x = torch.zeros(1, student.img_channels, student.img_resolution,
                              student.img_resolution, device=device)
        probe_s = torch.full((1,), 1.0, device=device)
        probe_l = T.one_hot_labels(torch.zeros(1, dtype=torch.long), student.label_dim, device)
        budget.calibrate("unet", student, (probe_x, probe_s, probe_l))
        reward_fn.flops_per_image(budget)
        if teacher_images is not None:
            budget.add("unet", len(teacher_images) * (2 * args.teacher_steps - 1))

    wandb_run = None
    if args.use_wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=f"draft_steps{num_steps}",
                               config={**vars(args), "num_steps": num_steps}, reinit=True)

    global_step = 0
    for epoch in range(start_epoch, args.num_epochs):
        # ====================================================================== TRAIN =========
        student.eval()   # EDM has dropout=0.10; see the note in the PPO script
        optimizer.zero_grad(set_to_none=True)
        epoch_rewards, epoch_parts = [], defaultdict(list)
        n_samples = 0

        for _ in range(args.num_batches_per_epoch):
            if args.online_teacher:
                B = args.sample_batch_size
                cls = torch.randint(0, min(args.num_classes_used, student.label_dim), (B,))
                lat = torch.randn(B, student.img_channels, student.img_resolution,
                                  student.img_resolution, device=device)
                labels = T.one_hot_labels(cls, student.label_dim, device)
                with torch.no_grad():
                    target_u8 = T.to_uint8(T.teacher_sample(
                        base_net.to(device), lat, labels, args.teacher_steps,
                        args.sigma_min, args.sigma_max, args.rho, lambda: torch.no_grad()))
            else:
                n_active = min(args.active_pairs, pool.n) if args.active_pairs > 0 else pool.n
                idx = torch.randint(0, n_active, (args.sample_batch_size,))
                lat, cls = pool.batch(idx)
                lat = lat.to(device)
                labels = T.one_hot_labels(cls, student.label_dim, device)
                target_u8 = teacher_images[idx]

            B = lat.shape[0]
            images = student_sample_with_grad(student, lat, labels, sigmas, args.eta,
                                              backprop_steps=args.backprop_steps,
                                              grad_checkpoint=bool(args.grad_checkpoint),
                                              budget=budget, n_images=B)
            reward, parts = differentiable_reward(reward_fn, images, target_u8, classes=cls)
            budget.add("reward", B, backward=True)   # student side, forward + backward
            budget.add("reward", B)                  # teacher side, forward only

            loss = -reward.mean()
            if args.kl_coeff > 0:
                tgt = T.to_unit(target_u8).to(device) * 2 - 1
                loss = loss + args.kl_coeff * 0.5 * (
                    (images - tgt).flatten(1) ** 2).sum(-1).mean()

            # Accumulate across the epoch's batches, then a single optimizer step -- the same
            # update cadence as --ppo_style coco, so "epoch" means the same thing in both arms.
            (loss / args.num_batches_per_epoch).backward()

            epoch_rewards.append(reward.detach().float().cpu())
            for k, v in parts.items():
                epoch_parts[k].append(v)
            n_samples += B

        gn = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        rewards_all = torch.cat(epoch_rewards).numpy()

        # ============================================================ DETERMINISTIC EVAL ======
        eval_metrics = {}
        if eval_idx is not None:
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            torch.manual_seed(args.eval_seed)
            ev_rewards, ev_parts = [], defaultdict(list)
            for st in range(0, len(eval_idx), args.sample_batch_size):
                sub_idx = eval_idx[st:st + args.sample_batch_size]
                lat_e, cls_e = pool.batch(sub_idx)
                lab_e = T.one_hot_labels(cls_e, student.label_dim, device)
                with torch.no_grad():
                    img_e = T.student_sample_det(student, lat_e.to(device), lab_e, sigmas,
                                                 args.eval_eta)
                budget.add("unet", len(sub_idx) * num_steps)
                r_e, p_e = reward_fn(img_e, teacher_images[sub_idx], classes=cls_e)
                budget.add("reward", 2 * len(sub_idx))
                ev_rewards.append(r_e)
                for k, v in p_e.items():
                    ev_parts[k].append(v)
            eval_metrics["eval_reward"] = float(torch.cat(ev_rewards).mean())
            for k, v in ev_parts.items():
                eval_metrics[f"eval_{k}"] = float(torch.cat(v).mean())
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

        if budget.enabled:
            eval_metrics["nfe_cum"] = budget.nfe
            eval_metrics["pflops_cum"] = budget.total_flops() / 1e15
        if args.lora_rank > 0:
            eval_metrics["lora_drift"] = T.lora_drift(student)
        else:
            eval_metrics["param_drift"] = T.param_drift(student, base_net)

        # ====================================================================== LOG ===========
        log = {
            "method": "draft",
            "steps": num_steps,
            "epoch": epoch,
            "reward_mean": float(rewards_all.mean()),
            "reward_std": float(rewards_all.std()),
            "pg_loss": float(-rewards_all.mean()),   # same key as the PPO arm so plots/parsers match
            "grad_norm": float(gn),
            "opt_steps": 1.0,
            "backprop_steps": float(k_bp),
            **{f"reward_{k}": float(torch.cat(v).mean()) for k, v in epoch_parts.items()},
            **eval_metrics,
        }
        print(json.dumps(log), flush=True)
        with open(os.path.join(run_dir, "log.jsonl"), "a") as f:
            f.write(json.dumps(log) + "\n")
        if wandb_run is not None:
            wandb_run.log(log, step=global_step)

        for key in ("epoch", "reward_mean", "reward_std", "pg_loss", "eval_reward"):
            history[key].append(log.get(key))
        history["n_samples"].append(n_samples)
        history["drift"].append(log.get("lora_drift", log.get("param_drift")))
        history["drift_label"].append("LoRA delta / base weight norm" if "lora_drift" in log
                                      else "Parameter drift (RMS, relative)")

        if args.sample_every and (epoch % args.sample_every == 0):
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            torch.manual_seed(args.eval_seed)
            with torch.no_grad():
                idx = torch.arange(0, min(16, pool.n))
                lat, cls = pool.batch(idx)
                labels = T.one_hot_labels(cls, student.label_dim, device)
                img = T.student_sample_det(student, lat.to(device), labels, sigmas, args.eval_eta)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            T.save_grid(img, os.path.join(run_dir, f"student_ep{epoch:04d}.png"))
            if teacher_images is not None:
                T.save_grid(T.to_unit(teacher_images[idx]).to(device) * 2 - 1,
                            os.path.join(run_dir, "teacher_reference.png"))

        # ---- checkpointing: same format as the PPO arm, so the eval script is shared ----------
        meta = {"epoch": epoch, "num_steps": num_steps, "sigmas": sigmas.cpu(),
                "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
                "lora_targets": args.lora_targets, "method": "draft",
                "reward_mean": log["reward_mean"], "eval_reward": log.get("eval_reward"),
                "args": vars(args)}
        best_payload = {"model": student.state_dict(), **meta}
        payload = {**best_payload, "optimizer": optimizer.state_dict()}

        torch.save(payload, os.path.join(run_dir, "last.pt"))
        select_on = log.get("eval_reward", log["reward_mean"])
        if select_on > best_reward:
            best_reward = select_on
            best_epoch = epoch
            best_path = os.path.join(run_dir, "best.pt")
            torch.save(best_payload, best_path)
            with open(os.path.join(run_dir, "best.json"), "w") as f:
                json.dump({"epoch": epoch, "reward_mean": best_reward,
                           **{f"reward_{k}": log.get(f"reward_{k}") for k in epoch_parts}},
                          f, indent=2)
            print(f"[ckpt] new best at epoch {epoch} "
                  f"({'eval_reward' if 'eval_reward' in log else 'reward'}={best_reward:.4f}) "
                  f"-> {os.path.abspath(best_path)}")

        T.save_training_curves(run_dir, history, T.short_reward_label(reward_fn.terms), best_epoch,
                               band=args.plot_band)
        if budget.enabled:
            with open(os.path.join(run_dir, "flops.json"), "w") as f:
                json.dump(budget.summary(epoch), f, indent=2)

        if args.save_every and (epoch % args.save_every == 0):
            ckpt = os.path.join(run_dir, f"student_ep{epoch:04d}.pt")
            torch.save(payload, ckpt)
            periodic_ckpts.append(ckpt)
            while len(periodic_ckpts) > args.num_checkpoint_limit:
                old = periodic_ckpts.pop(0)
                if os.path.exists(old):
                    os.remove(old)
            print(f"[ckpt] {ckpt}")

    if wandb_run is not None:
        wandb_run.finish()
    del student, optimizer
    torch.cuda.empty_cache()
    if budget.enabled:
        print(f"[flops] total {budget.total_flops() / 1e15:.3f} PFLOPs "
              f"({budget.nfe:.0f} student NFE-images)")
    print(f"[done] steps={num_steps}, best reward {best_reward:.4f} -> {run_dir}/best.pt")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.run_name:
        terms = args.reward_fn.replace("+", "_").replace(",", "_")
        args.run_name = (f"DRAFT_{terms}_kl{args.kl_coeff}_imagenet64"
                         f"_bs{args.sample_batch_size}_lr{args.lr}_k{args.backprop_steps}")
    print(f"[init] run_name={args.run_name}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = args.device
    autocast_ctx = T.make_autocast(device, None)

    steps_list = [int(s) for s in str(args.student_steps).replace(" ", "").split(",") if s]
    print(f"[init] student step counts: {steps_list}")

    print("[init] loading EDM teacher ...")
    teacher = T.load_edm(args.teacher_pkl, device, cache_dir=args.ckpt_dir,
                         edm_repo=args.edm_repo).eval().requires_grad_(False)
    assert teacher.label_dim > 0
    teacher.use_fp16 = bool(args.teacher_fp16)

    blob = None if args.online_teacher else T.peek_teacher_cache(args.teacher_cache)
    if (blob is not None and args.reuse_cache_pool
            and blob.get("pool_seed") == args.pool_seed
            and blob.get("num_pairs") != args.num_pairs):
        print(f"[cache] adopting the cached pool size: --num_pairs {args.num_pairs} -> "
              f"{blob['num_pairs']}")
        args.num_pairs = blob["num_pairs"]

    pool = T.PairPool(args.num_pairs, args.num_classes_used, teacher.label_dim,
                      teacher.img_channels, teacher.img_resolution, args.pool_seed)
    base_net = copy.deepcopy(teacher).cpu()
    base_net.use_fp16 = bool(args.student_fp16)

    if args.online_teacher:
        teacher_images = None
    else:
        teacher_images = T.build_teacher_cache(args, teacher, pool, device, autocast_ctx, blob=blob)
        if args.precompute_only:
            print("[done] teacher cache written; exiting (--precompute_only).")
            return
        del teacher
    torch.cuda.empty_cache()

    reward_fn = T.RewardBank(args, device)

    for num_steps in steps_list:
        train_student(args, num_steps, base_net, pool, teacher_images, reward_fn, device)

    print("[done] all runs finished")


if __name__ == "__main__":
    main()
