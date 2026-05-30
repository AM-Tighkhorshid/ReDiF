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
from torch.utils.data import Dataset, DataLoader

# --------------------
# Args
# --------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["coco", "laion"], default="coco", help="Dataset type to evaluate on")
parser.add_argument("--split", choices=["val", "test"], default="val", help="Dataset split to evaluate on")
parser.add_argument("--coco_root", default="coco_dataset", help="Root folder of COCO dataset")
parser.add_argument("--laion_root", default="laion_dataset", help="Root folder of LAION dataset")
parser.add_argument("--num_images", type=int, default=5000, help="Number of images to evaluate")
parser.add_argument("--teacher_model", default="/media/external20/amirhossein_tighkhorshid/diffusion_distillation/ddpo-pytorch-main/ddpo-pytorch-main/output/distill_clip/teacher_model", help="Teacher model directory")
parser.add_argument("--student_model", default="/media/external20/amirhossein_tighkhorshid/diffusion_distillation/ddpo-pytorch-main/ddpo-pytorch-main/PPO_clip_dino_coco_prompts/student_model", help="Student model directory")
parser.add_argument("--teacher_steps", type=int, default=50, help="Teacher diffusion steps")
parser.add_argument("--student_steps", type=int, default=5, help="Student diffusion steps")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for metrics calculation") # batch size is added

args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEACHER_MODEL = args.teacher_model
STUDENT_MODEL = args.student_model
TEACHER_DIR = os.path.join(TEACHER_MODEL, "evaluation_images_teacher5000_" + args.dataset)
STUDENT_DIR = os.path.join(STUDENT_MODEL, "evaluation_images_student5000_" + args.dataset)
os.makedirs(TEACHER_DIR, exist_ok=True)
os.makedirs(STUDENT_DIR, exist_ok=True)

# --------------------
# Dataset Loading (unchaged)
# --------------------
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
        if img_id in seen: continue
        seen.add(img_id)
        caption, file_name = ann["caption"], id_to_file[img_id]
        img_path = os.path.join(img_dir, file_name)
        if os.path.exists(img_path):
            prompts.append(caption)
            gt_images.append(img_path)
        if len(prompts) >= num_images: break
    return prompts, gt_images

def load_laion_data(root, split, num_images):
    ann_file = os.path.join(root, f"captions_{split}.json")
    img_dir = os.path.join(root,"images", f"{split}")
    with open(ann_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts, gt_images = [], []
    for entry in data:
        caption, file_name = entry.get("caption"), entry.get("file_name")
        img_path = os.path.join(img_dir, file_name)
        if not os.path.exists(img_path) or not caption or len(caption.strip()) < 3: continue
        prompts.append(caption.strip())
        gt_images.append(img_path)
        if len(prompts) >= num_images: break
    return prompts, gt_images

if args.dataset == "coco":
    prompts, gt_images = load_coco_data(args.coco_root, args.split, args.num_images)
elif args.dataset == "laion":
    prompts, gt_images = load_laion_data(args.laion_root, args.split, args.num_images)

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

    # torch.imference_model for better and faster generation
    with torch.inference_mode():
        for i, p in enumerate(tqdm(prompts, desc=f"Generate {model_dir}")):
            img = pipe(p, num_inference_steps=num_steps).images[0]
            img.save(os.path.join(out_dir, f"{i:04}.png"))

    del pipe
    torch.cuda.empty_cache()

generate_images(TEACHER_MODEL, prompts, TEACHER_DIR, args.teacher_steps)
generate_images(STUDENT_MODEL, prompts, STUDENT_DIR, args.student_steps)

# --------------------
# Metrics & Dataloaders (optimized version)
# --------------------

# Fast loader for paralel image loading
class ImagePathDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        im = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            im = self.transform(im)
        return im

# Appling Transformations
fid_transform = T.Compose([
    T.Resize((299, 299)),
    T.ToTensor(),
    T.Lambda(lambda x: (x * 255).to(torch.uint8))
])

clip_transform = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])

# loading CLIP model one for all usages
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to(DEVICE).eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)

@torch.inference_mode()
def extract_clip_features_batched(image_paths, batch_size=args.batch_size):
    dataset = ImagePathDataset(image_paths, transform=clip_transform)
    # num_workers=4 makes cpu working faster
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    
    features = []
    for batch in tqdm(loader, desc="Extracting CLIP Image Features"):
        batch = batch.to(DEVICE)
        feat = clip_model.get_image_features(batch)
        feat = F.normalize(feat, dim=-1)
        features.append(feat.cpu().numpy())
    
    return np.concatenate(features, axis=0)

@torch.inference_mode()
def extract_text_features_batched(prompts, batch_size=args.batch_size):
    tokenizer = clip_processor.tokenizer
    features = []
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Extracting CLIP Text Features"):
        batch_prompts = prompts[i:i+batch_size]
        encodings = tokenizer(
            batch_prompts, padding="max_length", truncation=True, max_length=77, return_tensors="pt"
        )
        for key in encodings:
            encodings[key] = encodings[key][:, :77].to(DEVICE)
            
        feat = clip_model.get_text_features(**encodings)
        feat = F.normalize(feat, dim=-1)
        features.append(feat.cpu().numpy())
        
    return np.concatenate(features, axis=0)

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

@torch.inference_mode()
def compute_fid_batched(gt_paths, gen_paths, batch_size=args.batch_size):
    fid = FrechetInceptionDistance().to(DEVICE)
    
    real_dataset = ImagePathDataset(gt_paths, transform=fid_transform)
    real_loader = DataLoader(real_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    for batch in tqdm(real_loader, desc="FID - Processing Real"):
        fid.update(batch.to(DEVICE), real=True)
        
    fake_dataset = ImagePathDataset(gen_paths, transform=fid_transform)
    fake_loader = DataLoader(fake_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    for batch in tqdm(fake_loader, desc="FID - Processing Fake"):
        fid.update(batch.to(DEVICE), real=False)
        
    return float(fid.compute())

def evaluate_model(model_name, gen_folder, prompts, gt_paths, real_img_feats, text_feats, save_metrics_path):
    gen_files = sorted([os.path.join(gen_folder, f) for f in os.listdir(gen_folder) if f.endswith(".png")])[:len(prompts)]
    
    results = {}
    print(f"\n--- Evaluating {model_name} ---")
    
    # 1. FID
    results["FID"] = compute_fid_batched(gt_paths, gen_files)
    
    # 2. Extract Fake Image Features for CLIP & PRDC
    fake_img_feats = extract_clip_features_batched(gen_files)
    
    # 3. CLIP Score
    sims = (fake_img_feats * text_feats).sum(axis=1)
    sims01 = (sims + 1.0) / 2.0
    results["CLIP_score"] = float(sims01.mean())
    
    # 4. PRDC
    print("Computing PRDC (this may take a moment)...")
    prdc_metrics = compute_prdc(real_img_feats, fake_img_feats)
    results.update(prdc_metrics)

    print(f"\nResults for {model_name} ({gen_folder.split('/')[-3]}):")
    for k, v in results.items():
        print(f"  {k}: {v:.6f}")

    with open(save_metrics_path, "w") as f:
        f.write(f"Evaluation results for {model_name} on {args.dataset.upper()} dataset\n")
        for k, v in results.items():
            f.write(f"{k}: {v}\n")

# --- Finding mutual features ---
print("\nExtracting Global Features (Real Images & Texts)...")
global_text_feats = extract_text_features_batched(prompts)
global_real_feats = extract_clip_features_batched(gt_images)

# ---  Models Evaluation ---
evaluate_model("Teacher", TEACHER_DIR, prompts, gt_images, global_real_feats, global_text_feats, os.path.join(TEACHER_MODEL, "metrics.txt"))
evaluate_model("Student", STUDENT_DIR, prompts, gt_images, global_real_feats, global_text_feats, os.path.join(STUDENT_MODEL, "metrics.txt"))