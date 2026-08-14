"""
Evaluation for the ReDiF EDM/ImageNet-64 student.

This is the ImageNet counterpart of the SD/COCO evaluation script. The COCO one cannot be reused:
it is prompt-driven, builds a StableDiffusionPipeline, and scores CLIP image-text alignment against
captions. Here conditioning is a class label, samples are 64x64, and the model is an EDM pickle.

What it reports
---------------
  FID                 student vs. the reference set (real ImageNet images if you have them,
                      otherwise the teacher's own samples -- see --ref).
  FID_teacher         teacher vs. the same reference set. The gap FID - FID_teacher is the part of
                      the degradation actually caused by distillation, as opposed to by the teacher.
  paired_*            per-pair student-vs-teacher agreement from the SAME (z_T, class): CLIP / DINOv3
                      cosine, LPIPS, MSE. This is the training reward measured on held-out pairs, so
                      compare it against the training curve to detect overfitting to the pool.
  precision/recall/   PRDC on DINOv3 features (fidelity vs. diversity; FID alone hides mode collapse,
  density/coverage    which is the failure mode RL fine-tuning is most prone to).
  class_acc           Top-1 agreement of a pretrained ImageNet classifier with the conditioning
                      label -- does the student still generate the class it was asked for?

Usage
-----
  python eval_imagenet_edm.py \\
      --student_ckpt logs/<run>/steps4/best.pt \\
      --edm_repo /path/to/edm \\
      --num_images 10000 --ref teacher

  # against real data instead (folder of images, any layout, recursively globbed):
  python eval_imagenet_edm.py --student_ckpt ... --ref real --real_dir /data/imagenet/val
"""

import argparse
import glob
import importlib.util
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


def load_train_module():
    """Reuse the sampler / loader / reward code from the training script."""
    path = os.path.join(HERE, "train_ppo_imagenet_edm.py")
    spec = importlib.util.spec_from_file_location("redif_train", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["redif_train"] = mod
    spec.loader.exec_module(mod)
    return mod


T = load_train_module()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--student_ckpt", type=str, required=True, help="e.g. logs/<run>/steps4/best.pt")
    p.add_argument("--teacher_pkl", type=str,
                   default="https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl")
    p.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    p.add_argument("--edm_repo", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--source", type=str, default="cache", choices=["cache", "fresh"],
                   help="cache = reuse the teacher images already generated for training "
                        "(--teacher_cache), no teacher sampling at all. fresh = generate new "
                        "held-out (z_T, class) pairs with the teacher (expensive: 511 NFE each).")
    p.add_argument("--teacher_cache", type=str, default="./teacher_cache_in64.pt")
    p.add_argument("--pool_seed", type=int, default=1234,
                   help="Must match the training run, otherwise the cached teacher images do not "
                        "correspond to the latents rebuilt here.")
    p.add_argument("--num_pairs", type=int, default=1000, help="Pool size used during training.")
    p.add_argument("--num_images", type=int, default=0,
                   help="0 => all cached pairs (source=cache) or 10000 (source=fresh).")
    p.add_argument("--gen_batch", type=int, default=128)
    p.add_argument("--eval_seed", type=int, default=12345,
                   help="Only used with --source fresh: a seed deliberately different from "
                        "--pool_seed, so the pairs scored were never trained on.")
    p.add_argument("--num_classes_used", type=int, default=1000)

    p.add_argument("--student_steps", type=int, default=0,
                   help="0 => take it from the checkpoint.")
    p.add_argument("--student_eta", type=float, default=0.0,
                   help="0 = deterministic Euler at eval time (report this as the headline number). "
                        "Set it to the training eta to measure the stochastic policy itself.")
    p.add_argument("--teacher_steps", type=int, default=256)
    p.add_argument("--skip_teacher", action="store_true",
                   help="Reuse a previously written teacher sample cache.")

    p.add_argument("--ref", type=str, default="real", choices=["real", "teacher"],
                   help="real = FID against actual ImageNet images (reported for BOTH student and "
                        "teacher). teacher = fallback when no ImageNet data is available; then only "
                        "the student-vs-teacher distance is meaningful.")
    p.add_argument("--real_dir", type=str, default="/media/external20/amirhossein_tighkhorshid/diffusion_distillation/Dataset/imagenet_val_64",
                   help="Folder of ImageNet images, globbed recursively (e.g. the val/ directory).")
    p.add_argument("--num_real", type=int, default=10000,
                   help="How many real images to use. Independent of the number of generated samples: "
                        "a larger real set costs almost nothing and lowers FID variance.")
    p.add_argument("--real_resize", type=str, default="box", choices=["box", "bicubic"],
                   help="Downsampling filter for real images. box matches the standard downsampled-"
                        "ImageNet-64 protocol EDM was trained and evaluated on; bicubic does not, and "
                        "shifts FID by a non-trivial amount.")
    p.add_argument("--real_cache", type=str, default=None,
                   help="Path to cache the preprocessed 64x64 real images (default: alongside --out_dir).")
    p.add_argument("--fid_batch", type=int, default=128)

    p.add_argument("--reward_fn", type=str, default="clip+dino+lpips+mse",
                   help="Paired student-vs-teacher metrics to report.")
    p.add_argument("--reward_weights", type=str, default=None)
    p.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--dino_model", type=str, default="vit_base_patch16_dinov3.lvd1689m")
    p.add_argument("--dino_tokens", type=str, default="patch", choices=["patch", "cls"])
    p.add_argument("--perception_model", type=str, default="vit_pe_core_base_patch16_224.fb")
    p.add_argument("--reward_resolution", type=int, default=224)
    p.add_argument("--lpips_resolution", type=int, default=64)
    p.add_argument("--kl_bins", type=int, default=32)

    p.add_argument("--classifier", type=str, default="resnet50.a1_in1k",
                   help="timm ImageNet classifier for the class-consistency check. Empty to skip.")
    p.add_argument("--prdc_k", type=int, default=5)
    p.add_argument("--prdc_samples", type=int, default=5000,
                   help="PRDC is O(n^2) in memory; subsample to this many images.")

    p.add_argument("--out_dir", type=str, default=None,
                   help="Defaults to the directory of --student_ckpt.")
    p.add_argument("--sigma_min", type=float, default=0.002)
    p.add_argument("--sigma_max", type=float, default=80.0)
    p.add_argument("--rho", type=float, default=7.0)
    return p.parse_args()


# ---------------------------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------------------------


@torch.no_grad()
def student_sample(net, latents, labels, sigmas, eta):
    """Same ancestral Euler step as training, but tolerant of eta=0 (pure deterministic Euler)."""
    x = latents * sigmas[0]
    for i in range(len(sigmas) - 1):
        s_from, s_to = float(sigmas[i]), float(sigmas[i + 1])
        s_down, s_up = T.ancestral_coeffs(s_from, s_to, eta)
        mean, _ = T.edm_step_mean(net, x, s_from, s_down, labels)
        x = mean.float()
        if s_up > 0:
            x = x + s_up * torch.randn_like(x)
    return x


def build_eval_set(args, net):
    """Return (latents [N,C,H,W] cpu, classes [N], teacher_u8 [N,3,64,64] or None, split tag).

    source=cache: rebuild the training PairPool from --pool_seed/--num_pairs (PairPool is fully
    deterministic given the seed, so the latents match the ones whose teacher images were cached)
    and read the teacher images straight off disk. No teacher forward passes at all.
    """
    if args.source == "cache":
        if not os.path.exists(args.teacher_cache):
            raise FileNotFoundError(f"{args.teacher_cache} not found -- run training (or "
                                    "--precompute_only) first, or use --source fresh.")
        blob = torch.load(args.teacher_cache, map_location="cpu")
        print(f"[cache] {os.path.abspath(args.teacher_cache)}: pool_seed={blob.get('pool_seed')}, "
              f"num_pairs={blob.get('num_pairs')}, teacher_steps={blob.get('teacher_steps')}")
        if blob["pool_seed"] != args.pool_seed:
            raise ValueError(
                f"Cache was built with pool_seed={blob['pool_seed']} but you passed {args.pool_seed}. "
                "The latents rebuilt here would not correspond to the cached teacher images.")
        if blob["num_pairs"] != args.num_pairs:
            # Adopt the cache's pool size: a different size changes the class labels, not just the
            # count, so the pool must be rebuilt at exactly the size the cache was written with.
            print(f"[cache] adopting the cached pool size ({args.num_pairs} -> {blob['num_pairs']})")
            args.num_pairs = blob["num_pairs"]
        pool = T.PairPool(args.num_pairs, args.num_classes_used, net.label_dim,
                          net.img_channels, net.img_resolution, args.pool_seed)
        n = args.num_images or pool.n
        n = min(n, pool.n)
        idx = torch.arange(n)
        lat, cls = pool.batch(idx)
        print(f"[gen] reusing {n} cached teacher images from {args.teacher_cache} "
              f"({blob['teacher_steps']}-step Heun) -- no teacher sampling needed")
        return lat, cls, blob["images"][idx], "train_pool"

    g = torch.Generator().manual_seed(args.eval_seed)
    n = args.num_images or 10000
    cls = torch.randint(0, min(args.num_classes_used, net.label_dim), (n,), generator=g)
    lat = torch.randn(n, net.img_channels, net.img_resolution, net.img_resolution, generator=g)
    return lat, cls, None, "held_out"


@torch.no_grad()
def sample_batched(net, latents, classes, args, device, kind, sigmas=None, eta=0.0):
    """Sample from fixed latents/classes so student and teacher share every (z_T, class) pair."""
    n = len(latents)
    out = torch.empty(n, net.img_channels, net.img_resolution, net.img_resolution, dtype=torch.uint8)
    for start in range(0, n, args.gen_batch):
        idx = torch.arange(start, min(start + args.gen_batch, n))
        lat = latents[idx].to(device)
        labels = T.one_hot_labels(classes[idx], net.label_dim, device)
        if kind == "student":
            img = student_sample(net, lat, labels, sigmas, eta)
        else:
            img = T.teacher_sample(net, lat, labels, args.teacher_steps,
                                   args.sigma_min, args.sigma_max, args.rho, lambda: torch.no_grad())
        out[idx] = T.to_uint8(img).cpu()
        print(f"\r  [{kind}] {idx[-1].item() + 1}/{n}", end="", flush=True)
    print()
    return out


def load_real_images(real_dir, n, resolution=64, resize="box", cache=None, seed=0):
    """Preprocess a folder of ImageNet images into uint8 [N,3,64,64].

    Two details that quietly ruin FID if you get them wrong:

    * Ordering. A recursive glob over an ImageNet directory comes back sorted by class folder, so
      taking the first N images would give you a handful of classes. The paths are shuffled with a
      fixed seed first, which keeps the reference set class-balanced and reproducible.
    * Filter. Downsampled ImageNet-64 (Chrabaszcz et al.) is built with a box filter, and that is
      the distribution EDM was trained on. Using bicubic here makes the reference sharper than the
      training data and inflates FID for both models.
    """
    if cache and os.path.exists(cache):
        blob = torch.load(cache, map_location="cpu")
        if blob["n"] >= n and blob["resize"] == resize:
            print(f"[real] loaded {n}/{blob['n']} cached images from {cache}")
            return blob["images"][:n]

    from PIL import Image
    filt = Image.BOX if resize == "box" else Image.BICUBIC
    exts = ("*.png", "*.jpg", "*.jpeg", "*.JPEG", "*.webp", "*.JPG")
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(real_dir, "**", e), recursive=True)
    paths = sorted(set(paths))
    if len(paths) < n:
        raise RuntimeError(f"Found only {len(paths)} images under {real_dir}, need {n}.")
    rng = np.random.default_rng(seed)
    paths = [paths[i] for i in rng.permutation(len(paths))[:n]]

    out = torch.empty(n, 3, resolution, resolution, dtype=torch.uint8)
    for i, path in enumerate(paths):
        im = Image.open(path).convert("RGB")
        w, h = im.size
        side = min(w, h)
        im = im.crop(((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2))
        im = im.resize((resolution, resolution), filt)
        out[i] = torch.from_numpy(np.array(im)).permute(2, 0, 1)
        if (i + 1) % 1000 == 0:
            print(f"\r  [real] {i + 1}/{n}", end="", flush=True)
    print()
    if cache:
        torch.save({"images": out, "n": n, "resize": resize}, cache)
        print(f"[real] cached to {cache}")
    return out


# ---------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------


@torch.no_grad()
def compute_fid(real_u8, fake_u8, device, batch=128):
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(normalize=False).to(device)
    for tag, arr, is_real in (("real", real_u8, True), ("fake", fake_u8, False)):
        for i in range(0, len(arr), batch):
            x = arr[i:i + batch].to(device)
            # Inception expects >=75px; the 64px samples are upsampled, exactly as EDM's own FID does.
            x = F.interpolate(x.float(), size=(299, 299), mode="bilinear",
                              align_corners=False).clamp(0, 255).to(torch.uint8)
            fid.update(x, real=is_real)
    return float(fid.compute())


def pairwise_distances(a, b):
    a2 = (a * a).sum(1).reshape(-1, 1)
    b2 = (b * b).sum(1).reshape(1, -1)
    return np.sqrt(np.maximum(a2 + b2 - 2.0 * a @ b.T, 0.0))


def compute_prdc(real_features, fake_features, nearest_k=5):
    d_rr = pairwise_distances(real_features, real_features)
    d_ff = pairwise_distances(fake_features, fake_features)
    rk = np.sort(d_rr, axis=1)[:, nearest_k]
    fk = np.sort(d_ff, axis=1)[:, nearest_k]
    d_fr = pairwise_distances(fake_features, real_features)
    d_rf = pairwise_distances(real_features, fake_features)

    precision = (d_fr.min(axis=1) <= rk[np.argmin(d_fr, axis=1)]).mean()
    recall = (d_rf.min(axis=1) <= fk[np.argmin(d_rf, axis=1)]).mean()
    density = ((d_fr.T <= rk.reshape(-1, 1)).sum(axis=1) / float(nearest_k)).mean()
    coverage = ((d_fr.T <= rk.reshape(-1, 1)).sum(axis=1) > 0).mean()
    return {"precision": float(precision), "recall": float(recall),
            "density": float(density), "coverage": float(coverage)}


@torch.no_grad()
def dino_features(bank, images_u8, device, batch=128):
    """Global (CLS) DINOv3 descriptors for PRDC -- patch tokens would be far too large here."""
    saved, bank.dino_tokens = bank.dino_tokens, "cls"
    feats = []
    for i in range(0, len(images_u8), batch):
        x01 = T.to_unit(images_u8[i:i + batch]).to(device)
        feats.append(bank._embed_dino(x01).float().cpu().numpy())
    bank.dino_tokens = saved
    return np.concatenate(feats, 0)


@torch.no_grad()
def class_accuracy(model_name, images_u8, classes, device, batch=128):
    import timm
    clf = timm.create_model(model_name, pretrained=True).to(device).eval()
    cfg = timm.data.resolve_data_config({}, model=clf)
    mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
    size = cfg["input_size"][-1]

    correct = 0
    for i in range(0, len(images_u8), batch):
        x = T.to_unit(images_u8[i:i + batch]).to(device)
        x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)
        pred = clf((x - mean) / std).argmax(dim=-1).cpu()
        correct += (pred == classes[i:i + batch]).sum().item()
    del clf
    torch.cuda.empty_cache()
    return correct / len(images_u8)


@torch.no_grad()
def paired_metrics(bank, student_u8, teacher_u8, classes, batch=64):
    """Reward terms evaluated on held-out pairs. Reuses the training-time reward implementations."""
    acc = {}
    for i in range(0, len(student_u8), batch):
        s_pm1 = T.to_unit(student_u8[i:i + batch]) * 2 - 1
        _, parts = bank(s_pm1, teacher_u8[i:i + batch], classes=classes[i:i + batch])
        for k, v in parts.items():
            acc.setdefault(k, []).append(v)
    return {f"paired_{k}": float(torch.cat(v).mean()) for k, v in acc.items()}


# ---------------------------------------------------------------------------------------------


def main():
    args = parse_args()
    device = args.device
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.student_ckpt))
    os.makedirs(out_dir, exist_ok=True)

    ck = torch.load(args.student_ckpt, map_location="cpu")
    num_steps = args.student_steps or int(ck.get("num_steps", 4))
    print(f"[init] student checkpoint: epoch {ck.get('epoch')}, {num_steps} steps, "
          f"train reward {ck.get('reward_mean')}")

    net = T.load_edm(args.teacher_pkl, device, cache_dir=args.ckpt_dir, edm_repo=args.edm_repo)
    net.eval().requires_grad_(False)

    # ---- evaluation set ---------------------------------------------------------------------
    latents, classes, teacher_u8, split = build_eval_set(args, net)
    n_eval = len(latents)
    if teacher_u8 is None:
        fresh_cache = os.path.join(out_dir, f"eval_teacher_n{n_eval}_seed{args.eval_seed}.pt")
        if args.skip_teacher and os.path.exists(fresh_cache):
            teacher_u8 = torch.load(fresh_cache, map_location="cpu")["teacher"]
            print(f"[gen] reusing {fresh_cache}")
        else:
            print(f"[gen] teacher, {args.teacher_steps} steps ...")
            teacher_u8 = sample_batched(net, latents, classes, args, device, "teacher")
            torch.save({"teacher": teacher_u8, "classes": classes}, fresh_cache)

    if split == "train_pool":
        print("[warn] These are the SAME pairs the student was trained on. The paired_* numbers "
              "therefore measure fit, not generalisation -- for a generalisation number rerun with "
              "--source fresh. FID/PRDC are less affected but still optimistic.")
    if n_eval < 10000:
        print(f"[warn] FID on {n_eval} images is strongly biased upward and not comparable with "
              "published 50k-sample FIDs. Treat it as a relative number within this study only.")

    student = T.copy.deepcopy(net)
    # Detect LoRA from the checkpoint's own keys rather than from a metadata field: older
    # checkpoints predate that field, and the weights can never lie about their own layout.
    lora_names = T.apply_lora_from_state_dict(student, ck["model"], ck.get("lora_alpha", 0))
    if lora_names:
        print(f"[init] rebuilt {len(lora_names)} LoRA adapters "
              f"(rank {ck['model'][lora_names[0] + '.A'].shape[0]}) before loading")
        student.to(device)
    student.load_state_dict(ck["model"])
    student.eval().requires_grad_(False)
    sigmas = (ck["sigmas"].to(device) if "sigmas" in ck else
              T.karras_sigmas(num_steps + 1, args.sigma_min, args.sigma_max, args.rho, device, True))
    print(f"[gen] student, {num_steps} steps, eta={args.student_eta} ...")
    student_u8 = sample_batched(student, latents, classes, args, device, "student",
                                sigmas=sigmas, eta=args.student_eta)
    del student, net
    torch.cuda.empty_cache()

    # ---- reference set ----------------------------------------------------------------------
    if args.ref == "real":
        if not args.real_dir:
            raise ValueError("--ref real requires --real_dir (a folder of ImageNet images).")
        real_cache = args.real_cache or os.path.join(
            out_dir, f"real_in64_n{args.num_real}_{args.real_resize}.pt")
        print(f"[ref] preparing {args.num_real} real ImageNet images ...")
        ref_u8 = load_real_images(args.real_dir, args.num_real, resize=args.real_resize,
                                  cache=real_cache)
    else:
        ref_u8 = teacher_u8
        print("[ref] no real data: using the teacher's own samples as the reference. This measures "
              "distance to the TEACHER, not to ImageNet -- label it as such in the paper.")

    results = {"student_ckpt": args.student_ckpt, "epoch": ck.get("epoch"),
               "num_steps": num_steps, "student_eta": args.student_eta,
               "teacher_steps": args.teacher_steps, "num_images": n_eval,
               "split": split, "reference": args.ref}

    # ---- FID --------------------------------------------------------------------------------
    # Both models are scored against the SAME reference with the SAME sample count, so the two
    # numbers share whatever small-sample bias the setup has and their difference is meaningful
    # even when the absolute values are not comparable with published 50k-sample FIDs.
    print(f"[metric] FID student vs {args.ref} ...")
    results["FID_student"] = compute_fid(ref_u8, student_u8, device, args.fid_batch)
    results["num_real"] = len(ref_u8)

    if args.ref == "real":
        print("[metric] FID teacher vs real ...")
        results["FID_teacher"] = compute_fid(ref_u8, teacher_u8, device, args.fid_batch)
        # The part of the degradation caused by distillation rather than by the teacher itself.
        results["FID_gap"] = results["FID_student"] - results["FID_teacher"]
        print(f"[metric] student {results['FID_student']:.3f} | teacher "
              f"{results['FID_teacher']:.3f} | gap {results['FID_gap']:.3f}")

    # ---- paired reward terms ------------------------------------------------------------------
    print("[metric] paired student-vs-teacher terms ...")
    bank = T.RewardBank(args, device)
    results.update(paired_metrics(bank, student_u8, teacher_u8, classes))

    # ---- PRDC -------------------------------------------------------------------------------
    if "dino" in bank.terms:
        k = min(args.prdc_samples, n_eval)
        print(f"[metric] PRDC on {k} DINOv3 features ...")
        f_ref = dino_features(bank, ref_u8[:k], device)
        f_gen = dino_features(bank, student_u8[:k], device)
        results.update(compute_prdc(f_ref, f_gen, args.prdc_k))

    # ---- class consistency --------------------------------------------------------------------
    if args.classifier:
        print(f"[metric] class consistency with {args.classifier} ...")
        results["class_acc_student"] = class_accuracy(args.classifier, student_u8, classes, device)
        results["class_acc_teacher"] = class_accuracy(args.classifier, teacher_u8, classes, device)

    # ---- report -----------------------------------------------------------------------------
    # Everything lands next to the .pt being evaluated, in three forms:
    #   metrics_<ckpt>_...json  one file per (checkpoint, config) -- the checkpoint name is in the
    #                           filename so evaluating ep0000 and best.pt does not overwrite one
    #                           with the other, which is the whole point of comparing them.
    #   eval_log.jsonl          append-only, one line per evaluation, same format as the training
    #                           log.jsonl so both can be read back with the same parser.
    #   eval_results.txt        human-readable, appended, for reading over ssh.
    import datetime
    ckpt_stem = os.path.splitext(os.path.basename(args.student_ckpt))[0]
    results["checkpoint"] = ckpt_stem
    results["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    path = os.path.join(out_dir, f"metrics_{ckpt_stem}_steps{num_steps}_{args.ref}_{split}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    log_path = os.path.join(out_dir, "eval_log.jsonl")
    with open(log_path, "a") as f:
        f.write(json.dumps(results) + "\n")

    txt_path = os.path.join(out_dir, "eval_results.txt")
    with open(txt_path, "a") as f:
        f.write(f"\n=== {results['timestamp']} | {ckpt_stem} | epoch {results['epoch']} | "
                f"{num_steps} steps | eta {args.student_eta} | ref={args.ref} | split={split} "
                f"| n={n_eval} ===\n")
        for k, v in results.items():
            f.write(f"  {k}: {v:.6f}\n" if isinstance(v, float) else f"  {k}: {v}\n")

    print("\n=== results ===")
    for k, v in results.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\nwritten to {path}")
    print(f"appended to {log_path} and {txt_path}")


if __name__ == "__main__":
    main()