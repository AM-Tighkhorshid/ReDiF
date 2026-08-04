"""
Evaluation for the ReDiF EDM/ImageNet-64 student.

This is the ImageNet counterpart of the SD/COCO evaluation script. The COCO one cannot be reused:
it is prompt-driven, builds a StableDiffusionPipeline, and scores CLIP image-text alignment against
captions. Here conditioning is a class label, samples are 64x64, and the model is an EDM pickle.

What it reports
---------------
  FID                 student vs. the reference set (real ImageNet images by default, or the
                      teacher's own samples with --ref teacher).
  FID_teacher         teacher vs. the same reference set. The gap FID - FID_teacher is the part of
                      the degradation actually caused by distillation, as opposed to by the teacher.
  paired_*            per-pair student-vs-teacher agreement from the SAME (z_T, class): CLIP / DINOv3
                      cosine, LPIPS, MSE. This is the training reward measured on held-out pairs, so
                      compare it against the training curve to detect overfitting to the pool.
  precision/recall/   PRDC on DINOv3 features (fidelity vs. diversity; FID alone hides mode collapse,
  density/coverage    which is the failure mode RL fine-tuning is most prone to).
  class_acc           Top-1 agreement of a pretrained ImageNet classifier with the conditioning
                      label -- does the student still generate the class it was asked for?

Real evaluation data
--------------------
With --ref real (the default) the script needs a folder of real ImageNet images. If
--real_dir does not exist or holds too few images it is downloaded automatically into
<repo>/../Dataset/imagenet_val_64 (override with --real_dir / --no_download). Teacher images are
NEVER regenerated for this: --source cache still reuses the teacher pool built during training.

Usage
-----
  python eval_imagenet_edm.py \\
      --student_ckpt logs/<run>/steps8/best.pt \\
      --edm_repo /path/to/edm \\
      --num_pairs 10000 --num_images 10000
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
# scripts/ -> ddpo-pytorch-main/ -> ddpo-pytorch-main/ -> diffusion_distillation/
DATA_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, os.pardir, "Dataset"))
DEFAULT_REAL_DIR = os.path.join(DATA_ROOT, "imagenet_val_64")

# Tried in order when the real-image folder is missing. All ImageNet mirrors on HuggingFace
# require accepting the licence once; `huggingface-cli login` afterwards is enough.
HF_CANDIDATES = [
    ("benjamin-paine/imagenet-1k-64x64", "validation"),
    ("evanarlian/imagenet_1k_resized_256", "val"),
    ("ILSVRC/imagenet-1k", "validation"),
]


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

    p.add_argument("--ref", type=str, default="real", choices=["teacher", "real"],
                   help="real = FID of BOTH student and teacher against real ImageNet images "
                        "(downloaded if missing). teacher = distance to the teacher's own samples.")
    p.add_argument("--real_dir", type=str, default=DEFAULT_REAL_DIR,
                   help="Folder of real ImageNet images. Downloaded here if absent.")
    p.add_argument("--real_seed", type=int, default=0,
                   help="Seed for subsampling the real folder; fixed => reproducible reference set.")
    p.add_argument("--real_resample", type=str, default="box", choices=["box", "bicubic"],
                   help="Down-sampling filter for real images. 'box' (area) matches the ADM/EDM "
                        "ImageNet-64 preprocessing; 'bicubic' does not and inflates FID.")
    p.add_argument("--hf_dataset", type=str, default=None,
                   help="Force a specific HuggingFace dataset for the download, e.g. "
                        "'benjamin-paine/imagenet-1k-64x64'. Default: try a built-in list.")
    p.add_argument("--hf_split", type=str, default=None, help="Split to use with --hf_dataset.")
    p.add_argument("--no_download", action="store_true",
                   help="Fail instead of downloading real images.")
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
        if blob["pool_seed"] != args.pool_seed or blob["num_pairs"] != args.num_pairs:
            raise ValueError(
                f"Cache was built with pool_seed={blob['pool_seed']}, num_pairs={blob['num_pairs']} "
                f"but you passed {args.pool_seed}/{args.num_pairs}. The latents rebuilt here would "
                f"not correspond to the cached teacher images. Pass "
                f"--pool_seed {blob['pool_seed']} --num_pairs {blob['num_pairs']}.")
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


# ---------------------------------------------------------------------------------------------
# Real evaluation data (ADDITION 1): presence check + automatic download
# ---------------------------------------------------------------------------------------------


IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.JPEG", "*.webp")


def list_images(directory):
    if not os.path.isdir(directory):
        return []
    paths = []
    for e in IMAGE_EXTS:
        paths += glob.glob(os.path.join(directory, "**", e), recursive=True)
    return sorted(paths)


def center_crop_resize(im, resolution, resample):
    """Square centre crop then down-sample, matching the ADM/EDM ImageNet-64 preprocessing."""
    from PIL import Image
    filt = Image.BOX if resample == "box" else Image.BICUBIC
    w, h = im.size
    side = min(w, h)
    im = im.crop(((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2))
    return im.resize((resolution, resolution), filt)


def _stream_split(dataset, split, n, dest_dir, resolution, resample, seed):
    """Write n images from a streamed HuggingFace split into dest_dir as 64x64 PNGs."""
    from datasets import load_dataset
    ds = load_dataset(dataset, split=split, streaming=True)
    # Shuffling the stream keeps the subset class-balanced: the student covers all 1000 classes
    # uniformly, so a class-ordered prefix would be the wrong reference distribution.
    ds = ds.shuffle(seed=seed, buffer_size=min(10000, max(1000, 5 * n)))
    os.makedirs(dest_dir, exist_ok=True)
    saved = 0
    for rec in ds:
        im = rec.get("image", rec.get("img", rec.get("jpg")))
        if im is None:
            raise RuntimeError(f"No image column in {dataset}; columns are {list(rec)}.")
        center_crop_resize(im.convert("RGB"), resolution, resample).save(
            os.path.join(dest_dir, f"{saved:06d}.png"))
        saved += 1
        if saved % 250 == 0:
            print(f"\r  [real] downloaded {saved}/{n}", end="", flush=True)
        if saved >= n:
            break
    print()
    return saved


def download_imagenet_val(dest_dir, n, resolution, resample, seed, dataset=None, split=None):
    """Download an ImageNet validation subset into dest_dir. Streaming: the full set is never pulled."""
    try:
        import datasets  # noqa: F401
    except ImportError as e:
        raise RuntimeError("`pip install datasets` is needed to download the evaluation set, "
                           "or point --real_dir at an existing ImageNet folder.") from e

    candidates = [(dataset, split or "validation")] if dataset else HF_CANDIDATES
    errors = []
    for ds_name, ds_split in candidates:
        print(f"[real] downloading {n} images from {ds_name}:{ds_split} -> {dest_dir}")
        try:
            got = _stream_split(ds_name, ds_split, n, dest_dir, resolution, resample, seed)
            if got >= n:
                return dest_dir
            errors.append(f"{ds_name}: only {got} images")
        except Exception as e:  # gated repo, missing split, network, ...
            print(f"[real] {ds_name} failed: {type(e).__name__}: {e}")
            errors.append(f"{ds_name}: {type(e).__name__}")

    raise RuntimeError(
        "Could not fetch an ImageNet validation subset automatically (" + "; ".join(errors) + ").\n"
        "Every ImageNet mirror on HuggingFace is licence-gated: open the dataset page, accept the "
        "terms once, run `huggingface-cli login`, then rerun. Alternatively copy any folder of "
        f"ImageNet val images to {dest_dir} (any layout, globbed recursively) and rerun.")


def ensure_real_images(args, n, resolution=64):
    """Check that enough real images exist, download them if not, then load n of them as uint8."""
    have = len(list_images(args.real_dir))
    print(f"[real] {args.real_dir}: {have} images found, {n} needed")
    if have < n:
        if args.no_download:
            raise RuntimeError(f"{args.real_dir} has {have} images, need {n}, --no_download set.")
        os.makedirs(os.path.dirname(os.path.abspath(args.real_dir)), exist_ok=True)
        download_imagenet_val(args.real_dir, n, resolution, args.real_resample,
                              args.real_seed, args.hf_dataset, args.hf_split)
    return load_real_images(args.real_dir, n, resolution, args.real_resample, args.real_seed)


def load_real_images(real_dir, n, resolution=64, resample="box", seed=0):
    """Load a random (hence class-balanced) subset of a real-image folder as uint8 [N,3,64,64]."""
    from PIL import Image
    paths = list_images(real_dir)
    if len(paths) < n:
        raise RuntimeError(f"Found only {len(paths)} images in {real_dir}, need {n}.")

    # Sorted order on ImageNet is class-ordered (n01440764/, n01443537/, ...), so taking a prefix
    # would cover only the first few classes while the student spans all 1000. Sample uniformly.
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(len(paths), size=n, replace=False))
    paths = [paths[i] for i in sel]

    out = torch.empty(n, 3, resolution, resolution, dtype=torch.uint8)
    for i, path in enumerate(paths):
        im = Image.open(path).convert("RGB")
        if im.size != (resolution, resolution):
            im = center_crop_resize(im, resolution, resample)
        out[i] = torch.from_numpy(np.array(im)).permute(2, 0, 1)
        if (i + 1) % 500 == 0:
            print(f"\r  [real] {i + 1}/{n}", end="", flush=True)
    print()
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
        # ADDITION 1: check the evaluation dataset is there, download it if not. The teacher
        # images above are untouched by this -- they still come from the training cache.
        ref_u8 = ensure_real_images(args, n_eval)
    else:
        ref_u8 = teacher_u8
        print("[ref] using the teacher's own samples as the reference distribution. This measures "
              "distance to the TEACHER, not to real ImageNet -- label it as such in the paper.")

    results = {"student_ckpt": args.student_ckpt, "epoch": ck.get("epoch"),
               "num_steps": num_steps, "student_eta": args.student_eta,
               "teacher_steps": args.teacher_steps, "num_images": n_eval,
               "split": split, "reference": args.ref}

    # ---- FID --------------------------------------------------------------------------------
    # ADDITION 2: with --ref real both the student AND the teacher are scored against the real
    # data, so the table can show how much of the FID is the teacher's and how much distillation's.
    print("[metric] FID (student) ...")
    results["FID"] = compute_fid(ref_u8, student_u8, device, args.fid_batch)
    if args.ref == "real":
        results["real_dir"] = args.real_dir
        print("[metric] FID (teacher) ...")
        results["FID_teacher"] = compute_fid(ref_u8, teacher_u8, device, args.fid_batch)
        results["FID_gap"] = results["FID"] - results["FID_teacher"]

    # ---- paired reward terms ------------------------------------------------------------------
    print("[metric] paired student-vs-teacher terms ...")
    bank = T.RewardBank(args, device)
    results.update(paired_metrics(bank, student_u8, teacher_u8, classes))

    # ---- PRDC -------------------------------------------------------------------------------
    if "dino" in bank.terms:
        k = min(args.prdc_samples, n_eval)
        print(f"[metric] PRDC on {k} DINOv3 features (reference: {args.ref}) ...")
        f_ref = dino_features(bank, ref_u8[:k], device)
        f_gen = dino_features(bank, student_u8[:k], device)
        results.update(compute_prdc(f_ref, f_gen, args.prdc_k))

    # ---- class consistency --------------------------------------------------------------------
    if args.classifier:
        print(f"[metric] class consistency with {args.classifier} ...")
        results["class_acc_student"] = class_accuracy(args.classifier, student_u8, classes, device)
        results["class_acc_teacher"] = class_accuracy(args.classifier, teacher_u8, classes, device)

    # ---- report -----------------------------------------------------------------------------
    path = os.path.join(out_dir, f"metrics_steps{num_steps}_{args.ref}_{split}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== results ===")
    for k, v in results.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()