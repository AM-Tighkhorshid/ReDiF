import os
import json
import csv
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
# --------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["coco", "laion"], default="coco", help="Dataset type to evaluate on")
parser.add_argument("--split", choices=["val", "test"], default="val", help="Dataset split to evaluate on")
parser.add_argument("--coco_root", default="coco_dataset", help="Root folder of COCO dataset")
parser.add_argument("--laion_root", default="laion_dataset", help="Root folder of LAION dataset")
parser.add_argument("--num_images", type=int, default=100, help="Number of images to evaluate")
parser.add_argument("--teacher_model", default="/media/external20/amirhossein_tighkhorshid/diffusion_distillation/ddpo-pytorch-main/ddpo-pytorch-main/output/distill_clip/teacher_model", help="Teacher model directory")
parser.add_argument("--student_model", default="/media/external20/amirhossein_tighkhorshid/diffusion_distillation/ddpo-pytorch-main/ddpo-pytorch-main/DMD2_distill_coco_prompts/student_model", help="Student model directory")
parser.add_argument("--teacher_steps", type=int, default=50, help="Teacher diffusion steps")
parser.add_argument("--student_steps", type=int, default=5, help="Student diffusion steps")

args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEACHER_MODEL = args.teacher_model
STUDENT_MODEL = args.student_model
TEACHER_DIR = os.path.join(TEACHER_MODEL, "evaluation_images_teacher_" + args.dataset)
STUDENT_DIR = os.path.join(STUDENT_MODEL, "evaluation_images_student_" + args.dataset)
os.makedirs(TEACHER_DIR, exist_ok=True)
os.makedirs(STUDENT_DIR, exist_ok=True)

# --------------------
# Dataset Loading
# --------------------
prompts, gt_images = [], []

def load_coco_data(root, split, num_images):
    ann_file = os.path.join(root, "annotations", f"captions_{split}2017.json")
    img_dir = os.path.join(root, f"{split}2017")

    with open(ann_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
    seen = set()
    prompts, gt_images = [], []
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id in seen:
            continue
        seen.add(img_id)
        caption = ann["caption"]
        file_name = id_to_file[img_id]
        img_path = os.path.join(img_dir, file_name)
        if os.path.exists(img_path):
            prompts.append(caption)
            gt_images.append(img_path)
        if len(prompts) >= num_images:
            break

    print("-" * 50)
    print(f"List of {len(prompts)} Prompts Used for Image Generation:")
    print("-" * 50)

    for i, prompt in enumerate(prompts):
        # Prints the index (0-99) followed by the prompt
        print(f"Prompt {i+1} ({i:04}.png): {prompt}") 

    print("-" * 50)
    return prompts, gt_images

def load_laion_data(root, split, num_images):
    ann_file = os.path.join(root, f"captions_{split}.json")
    img_dir = os.path.join(root,"images", f"{split}")
    if not os.path.exists(ann_file):
        raise FileNotFoundError(f"Missing LAION annotation file: {ann_file}")
    with open(ann_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts, gt_images = [], []
    for entry in data:
        caption = entry.get("caption")
        file_name = entry.get("file_name")
        img_path = os.path.join(img_dir, file_name)
        if not os.path.exists(img_path):
            continue
        if not caption or len(caption.strip()) < 3:
            continue
        prompts.append(caption.strip())
        gt_images.append(img_path)
        if len(prompts) >= num_images:
            break
    return prompts, gt_images


if args.dataset == "coco":
    prompts, gt_images = load_coco_data(args.coco_root, args.split, args.num_images)
    print(f"Loaded {len(prompts)} samples from COCO {args.split}2017")
elif args.dataset == "laion":
    prompts, gt_images = load_laion_data(args.laion_root, args.split, args.num_images)
    print(f"Loaded {len(prompts)} samples from LAION {args.split}2017")

if len(prompts) < args.num_images:
    raise RuntimeError(f"Found only {len(prompts)} valid images; need {args.num_images}")

# --------------------
# Generation
# --------------------
def generate_images(model_dir, prompts, out_dir, num_steps):
    existing_pngs = sorted([f for f in os.listdir(out_dir) if f.endswith(".png")])
    if len(existing_pngs) >= len(prompts):
        print(f"Skipping generation for {model_dir}, found {len(existing_pngs)} images.")
        return

    pipe = StableDiffusionPipeline.from_pretrained(
        model_dir, torch_dtype=torch.float16, safety_checker=None
    ).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)

    for i, p in enumerate(tqdm(prompts, desc=f"Generate {model_dir}")):
        img = pipe(p, num_inference_steps=num_steps).images[0]
        img.save(os.path.join(out_dir, f"{i:04}.png"))

    del pipe
    torch.cuda.empty_cache()

generate_images(TEACHER_MODEL, prompts, TEACHER_DIR, args.teacher_steps)
generate_images(STUDENT_MODEL, prompts, STUDENT_DIR, args.student_steps)

# --------------------
# Metrics
# --------------------
resize_299 = T.Resize((299, 299))
to_tensor = T.ToTensor()

def load_images_as_uint8(folder, limit=None):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".png")])
    if limit:
        files = files[:limit]
    imgs = []
    for fn in files:
        im = Image.open(os.path.join(folder, fn)).convert("RGB")
        im = resize_299(im)
        t = to_tensor(im) * 255.0
        imgs.append(t.to(torch.uint8))
    return torch.stack(imgs).to(DEVICE)

def load_gt_images_as_uint8(paths):
    imgs = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        im = resize_299(im)
        t = to_tensor(im) * 255.0
        imgs.append(t.to(torch.uint8))
    return torch.stack(imgs).to(DEVICE)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

@torch.no_grad()
def compute_clip_pairwise_scores_from_files(images_folder, prompts):
    from transformers import CLIPModel, CLIPProcessor
    import torch.nn.functional as F

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = clip_processor.tokenizer

    # ----- Strict truncation to CLIP limit (77 tokens) -----
    encodings = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt"
    )

    # Hard truncate if still exceeds max_length
    for key in encodings:
        encodings[key] = encodings[key][:, :77]
    text_inputs = {k: v.to(DEVICE) for k, v in encodings.items()}
    # --------------------------------------------------------

    # Get normalized text features
    text_feats = clip_model.get_text_features(**text_inputs)
    text_feats = F.normalize(text_feats, dim=-1).cpu().numpy()

    # Get normalized image features
    image_feats = []
    files = sorted([f for f in os.listdir(images_folder) if f.endswith(".png")])[: len(prompts)]
    for fn in files:
        im = Image.open(os.path.join(images_folder, fn))
        im_inputs = clip_processor(images=im, return_tensors="pt").to(DEVICE)
        f = clip_model.get_image_features(**im_inputs)
        f = F.normalize(f, dim=-1).cpu().numpy()
        image_feats.append(f)

    image_feats = np.concatenate(image_feats, axis=0)
    sims = (image_feats * text_feats).sum(axis=1)
    sims01 = (sims + 1.0) / 2.0
    return float(sims01.mean())


def pairwise_distances_np(a, b):
    a2 = np.sum(a * a, axis=1).reshape(-1, 1)
    b2 = np.sum(b * b, axis=1).reshape(1, -1)
    d2 = np.maximum(a2 + b2 - 2.0 * np.dot(a, b.T), 0.0)
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
    density = ((d_fr.T <= rk.reshape(-1, 1)).sum(axis=1) / float(nearest_k)).mean()
    coverage = ((d_fr.T <= rk.reshape(-1, 1)).sum(axis=1) > 0).mean()
    return dict(precision=float(precision), recall=float(recall), density=float(density), coverage=float(coverage))

def compute_fid_between(gt_paths, gen_folder):
    fid = FrechetInceptionDistance().to(DEVICE)
    real_t = load_gt_images_as_uint8(gt_paths)
    fake_t = load_images_as_uint8(gen_folder, limit=len(gt_paths))
    fid.update(real_t, real=True)
    fid.update(fake_t, real=False)
    return float(fid.compute())/3

def compute_prdc_and_clip_and_save(model_name, gen_folder, prompts, gt_paths, save_metrics_path):
    results = {}
    results["FID"] = compute_fid_between(gt_paths, gen_folder)
    results["CLIP_score"] = compute_clip_pairwise_scores_from_files(gen_folder, prompts)

    real_feats, fake_feats = [], []
    for p in gt_paths:
        im = Image.open(p)
        inputs = clip_processor(images=im, return_tensors="pt").to(DEVICE)
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

    prdc_metrics = compute_prdc(real_feats, fake_feats)
    results.update(prdc_metrics)

    print(f"\nResults for {model_name}:")
    for k, v in results.items():
        print(f"  {k}: {v:.6f}")

    with open(save_metrics_path, "w") as f:
        f.write(f"Evaluation results for {model_name} on {args.dataset.upper()} dataset\n")
        for k, v in results.items():
            f.write(f"{k}: {v}\n")

    return results

compute_prdc_and_clip_and_save("Teacher", TEACHER_DIR, prompts, gt_images, os.path.join(TEACHER_MODEL, "metrics.txt"))
compute_prdc_and_clip_and_save("Student", STUDENT_DIR, prompts, gt_images, os.path.join(STUDENT_MODEL, "metrics.txt"))