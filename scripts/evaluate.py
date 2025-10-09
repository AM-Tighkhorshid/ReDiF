# scripts/evaluate.py
import os
import json
import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from transformers import CLIPProcessor, CLIPModel
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
import argparse

# --------------------
# Args
parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["val", "test"], default="val", help="COCO split to evaluate on")
parser.add_argument("--coco_root", default="coco_dataset", help="Root folder of COCO dataset")
parser.add_argument("--num_images", type=int, default=100, help="Number of images to evaluate")
parser.add_argument("--teacher_model", default="output/distill_clip/teacher_model", help="Teacher model dir")
parser.add_argument("--student_model", default="PPO_clip_dino_text_image_aesthetic_kl_1.0_coco_prompts/student_model", help="Student model dir")
parser.add_argument("--teacher_steps", type=int, default=50, help="Teacher sampling steps")
parser.add_argument("--student_steps", type=int, default=5, help="Student sampling steps")
args = parser.parse_args()

COCO_ROOT = args.coco_root
SPLIT = args.split
NUM_IMAGES = args.num_images
TEACHER_MODEL = args.teacher_model
STUDENT_MODEL = args.student_model
TEACHER_DIR = os.path.join(TEACHER_MODEL, "evaluation_images_teacher")
STUDENT_DIR = os.path.join(STUDENT_MODEL, "evaluation_images_student")
TEACHER_STEPS = args.teacher_steps
STUDENT_STEPS = args.student_steps

ANN_FILE = os.path.join(COCO_ROOT, "annotations", f"captions_{SPLIT}2017.json")
IMG_DIR = os.path.join(COCO_ROOT, f"{SPLIT}2017")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(TEACHER_DIR, exist_ok=True)
os.makedirs(STUDENT_DIR, exist_ok=True)

# --------------------
# Load COCO captions & gt image paths (unique images)
with open(ANN_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
annotations = data["annotations"]

seen = set()
prompts, gt_images = [], []
for ann in annotations:
    img_id = ann["image_id"]
    if img_id in seen:
        continue
    seen.add(img_id)
    caption = ann["caption"]
    file_name = id_to_file[img_id]
    img_path = os.path.join(IMG_DIR, file_name)
    if os.path.exists(img_path):
        prompts.append(caption)
        gt_images.append(img_path)
    if len(prompts) >= NUM_IMAGES:
        break

if len(prompts) < NUM_IMAGES:
    raise RuntimeError(f"Only found {len(prompts)} valid GT images; need {NUM_IMAGES}")

print(f"Loaded {len(prompts)} prompts and GT images from COCO {SPLIT}2017")

# --------------------
# Generation helper
def generate_images(model_dir, prompts, out_dir, num_steps):
    existing_pngs = sorted([f for f in os.listdir(out_dir) if f.endswith(".png")])
    if len(existing_pngs) >= len(prompts):
        print(f"Skipping generation for {model_dir}, found {len(existing_pngs)} images.")
        return

    pipe = StableDiffusionPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)

    for i, p in enumerate(tqdm(prompts, desc=f"Generate {model_dir}")):
        img = pipe(p, num_inference_steps=num_steps).images[0]
        img.save(os.path.join(out_dir, f"{i:04}.png"))

    del pipe
    torch.cuda.empty_cache()

# Generate (or skip if exist)
generate_images(TEACHER_MODEL, prompts, TEACHER_DIR, TEACHER_STEPS)
generate_images(STUDENT_MODEL, prompts, STUDENT_DIR, STUDENT_STEPS)

# --------------------
# Utilities for metrics
resize_299 = T.Resize((299, 299))
to_tensor = T.ToTensor()  # float [0,1]

def load_images_as_uint8(folder, limit=None):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".png")])
    if limit:
        files = files[:limit]
    imgs = []
    for fn in files:
        im = Image.open(os.path.join(folder, fn)).convert("RGB")
        im = resize_299(im)
        t = to_tensor(im) * 255.0   # float [0,255]
        t = t.to(torch.uint8)       # cast to uint8
        imgs.append(t)
    return torch.stack(imgs).to(DEVICE)

def load_gt_images_as_uint8(paths):
    imgs = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        im = resize_299(im)
        t = to_tensor(im) * 255.0
        t = t.to(torch.uint8)
        imgs.append(t)
    return torch.stack(imgs).to(DEVICE)

# --------------------
# CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

@torch.no_grad()
def compute_clip_pairwise_scores_from_files(images_folder, prompts):
    text_inputs = clip_processor(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
    text_feats = clip_model.get_text_features(**text_inputs)
    text_feats = F.normalize(text_feats, dim=-1).cpu().numpy()

    image_feats = []
    files = sorted([f for f in os.listdir(images_folder) if f.endswith(".png")])[: len(prompts)]
    for fn in files:
        im = Image.open(os.path.join(images_folder, fn)).convert("RGB")
        im = clip_processor(images=im, return_tensors="pt").to(DEVICE)
        f = clip_model.get_image_features(**im)
        f = F.normalize(f, dim=-1).cpu().numpy()
        image_feats.append(f)
    image_feats = np.concatenate(image_feats, axis=0)

    sims = (image_feats * text_feats).sum(axis=1)
    sims01 = (sims + 1.0) / 2.0
    return float(sims01.mean())

# --------------------
# PRDC implementation (pure numpy)
def pairwise_distances_np(a, b):
    a2 = np.sum(a * a, axis=1).reshape(-1, 1)
    b2 = np.sum(b * b, axis=1).reshape(1, -1)
    d2 = a2 + b2 - 2.0 * np.dot(a, b.T)
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(d2)

def compute_prdc(real_features, fake_features, nearest_k=5):
    d_rr = pairwise_distances_np(real_features, real_features)
    d_ff = pairwise_distances_np(fake_features, fake_features)
    rk = np.sort(d_rr, axis=1)[:, nearest_k]
    fk = np.sort(d_ff, axis=1)[:, nearest_k]

    d_fr = pairwise_distances_np(fake_features, real_features)
    d_rf = pairwise_distances_np(real_features, fake_features)

    nearest_real_idx = np.argmin(d_fr, axis=1)
    nearest_real_dist = d_fr.min(axis=1)
    precision = (nearest_real_dist <= rk[nearest_real_idx]).mean()

    nearest_fake_idx = np.argmin(d_rf, axis=1)
    nearest_fake_dist = d_rf.min(axis=1)
    recall = (nearest_fake_dist <= fk[nearest_fake_idx]).mean()

    d_fr_T = d_fr.T
    counts_in_radius = (d_fr_T <= rk.reshape(-1, 1)).sum(axis=1)
    density = (counts_in_radius / float(nearest_k)).mean()
    coverage = (counts_in_radius > 0).mean()

    return {"precision": float(precision), "recall": float(recall),
            "density": float(density), "coverage": float(coverage)}

# --------------------
# Evaluate helper
def compute_fid_between(gt_paths, gen_folder):
    fid = FrechetInceptionDistance().to(DEVICE)
    real_t = load_gt_images_as_uint8(gt_paths)
    fake_t = load_images_as_uint8(gen_folder, limit=len(gt_paths))
    fid.update(real_t, real=True)
    fid.update(fake_t, real=False)
    return float(fid.compute())

def compute_prdc_and_clip_and_save(model_name, gen_folder, prompts, gt_paths, save_metrics_path):
    results = {}
    results["FID"] = compute_fid_between(gt_paths, gen_folder)
    results["CLIP_score"] = compute_clip_pairwise_scores_from_files(gen_folder, prompts)

    real_feats, fake_feats = [], []
    for p in gt_paths:
        inputs = clip_processor(images=Image.open(p).convert("RGB"), return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            f = clip_model.get_image_features(**inputs)
            f = F.normalize(f, dim=-1).cpu().numpy()
            real_feats.append(f)
    real_feats = np.concatenate(real_feats, axis=0)

    files = sorted([f for f in os.listdir(gen_folder) if f.endswith(".png")])[: len(prompts)]
    for fn in files:
        inputs = clip_processor(images=Image.open(os.path.join(gen_folder, fn)).convert("RGB"), return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            f = clip_model.get_image_features(**inputs)
            f = F.normalize(f, dim=-1).cpu().numpy()
            fake_feats.append(f)
    fake_feats = np.concatenate(fake_feats, axis=0)

    prdc_metrics = compute_prdc(real_feats, fake_feats, nearest_k=5)
    results.update(prdc_metrics)

    print(f"\nResults for {model_name}:")
    for k, v in results.items():
        print(f"  {k}: {v:.6f}")

    with open(save_metrics_path, "w") as f:
        f.write(f"Evaluation results for {model_name} on COCO {SPLIT}2017\n")
        for k, v in results.items():
            f.write(f"{k}: {v}\n")
    return results

# --------------------
# Run evaluations
compute_prdc_and_clip_and_save("Teacher", TEACHER_DIR, prompts, gt_images, os.path.join(TEACHER_MODEL, "metrics.txt"))
compute_prdc_and_clip_and_save("Student", STUDENT_DIR, prompts, gt_images, os.path.join(STUDENT_MODEL, "metrics.txt"))
