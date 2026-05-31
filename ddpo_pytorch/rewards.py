"""
rewards.py — reward functions for teacher→student distillation via PPO.

Unified call signature for ALL reward functions inside composite_reward_fn:
    fn(student_images, teacher_images, prompts) -> (scores: list[float], info: dict|None)

Reward types
------------
"clip"        – CLIP image-image cosine similarity        (student ↔ teacher)
"dino"        – DINOv2 patch-token cosine similarity      (student ↔ teacher)
"perception"  – negative LPIPS perceptual distance        (student ↔ teacher)
"text_image"  – CLIP image-text alignment                 (student image ↔ prompt)
"aesthetic"   – LAION aesthetic score                     (student image only)
"mse"         – negative MSE between student and teacher  (student ↔ teacher)
               This is the diffusion reconstruction loss turned into a reward:
               reward = -MSE(student_pixels, teacher_pixels)
               A less negative (higher) reward means the student image is
               closer to the teacher image in pixel space, identical to the
               standard L2 diffusion loss but used as an RL reward signal.
"""

import io
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import CLIPProcessor, CLIPModel


# ---------------------------------------------------------------------------
# Shared image-preprocessing helpers
# ---------------------------------------------------------------------------

def _pil_or_tensor_to_tensor(images, size: int, mean, std, device):
    """
    Accept a list[PIL.Image] or a (B,C,H,W) float tensor in [0,1] or [0,255].
    Returns a (B,C,size,size) float32 tensor on `device`, normalised with
    the supplied mean/std.
    """
    transform = Compose([
        Resize(size, interpolation=Image.BICUBIC),
        CenterCrop(size),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])

    if isinstance(images, (list, tuple)) and isinstance(images[0], Image.Image):
        imgs = torch.stack([transform(img.convert("RGB")) for img in images])
        return imgs.to(device=device, dtype=torch.float32)

    if isinstance(images, torch.Tensor):
        imgs = images.to(device=device, dtype=torch.float32)
        if imgs.max() > 1.0 + 1e-3:          # [0,255] → [0,1]
            imgs = imgs / 255.0
        if imgs.shape[2] != size or imgs.shape[3] != size:
            imgs = F.interpolate(imgs, size=(size, size),
                                 mode="bicubic", align_corners=False)
        mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        std_t  = torch.tensor(std,  device=device).view(1, 3, 1, 1)
        return (imgs - mean_t) / std_t

    raise TypeError(f"Images must be list[PIL.Image] or torch.Tensor, got {type(images)}")


def _to_float_tensor(images, device):
    """
    Convert images to a (B,C,H,W) float32 tensor in [0,1] on `device`,
    WITHOUT any normalisation. Used for pixel-space MSE.
    """
    if isinstance(images, (list, tuple)) and isinstance(images[0], Image.Image):
        t = Compose([ToTensor()])
        imgs = torch.stack([t(img.convert("RGB")) for img in images])
        return imgs.to(device=device, dtype=torch.float32)

    if isinstance(images, torch.Tensor):
        imgs = images.to(device=device, dtype=torch.float32)
        if imgs.max() > 1.0 + 1e-3:
            imgs = imgs / 255.0
        return imgs

    raise TypeError(f"Images must be list[PIL.Image] or torch.Tensor, got {type(images)}")


# ---------------------------------------------------------------------------
# MSE Reconstruction Reward
# ---------------------------------------------------------------------------

def mse_reconstruction_reward_fn(teacher_pipeline, student_pipeline):
    """
    Negative pixel-space MSE between student and teacher final output images.

    This directly mirrors the standard diffusion reconstruction loss (L2),
    but used as an RL reward signal so the student learns via policy gradients
    instead of direct backpropagation through the teacher.

    reward_i = -MSE(student_image_i, teacher_image_i)

    A reward of 0.0 means pixel-perfect match. More negative means worse.
    Typical range is roughly [-0.5, 0.0] for SD outputs.

    Args:
        teacher_pipeline: the frozen teacher StableDiffusionPipeline
        student_pipeline: the trainable student StableDiffusionPipeline

    Returns:
        reward_fn(student_images, teacher_images, prompts)
            -> (scores: list[float], info: dict)
    """
    device = student_pipeline.device

    def reward_fn(student_images, teacher_images, prompts=None):
        # Convert both to unnormalised [0,1] float tensors — same as diffusion L2 target
        s = _to_float_tensor(student_images, device)   # (B, 3, H, W)
        t = _to_float_tensor(teacher_images, device)   # (B, 3, H, W)

        # Resize to a common resolution if they differ (edge case)
        if s.shape != t.shape:
            t = F.interpolate(t, size=s.shape[2:], mode="bicubic", align_corners=False)

        with torch.no_grad():
            # Per-sample MSE: mean over (C, H, W), keep batch dim
            mse_per_sample = ((s - t) ** 2).mean(dim=[1, 2, 3])   # (B,)

            # Negate: higher reward = lower MSE = better reconstruction
            reward = -mse_per_sample                               # (B,)

        info = {
            "mse": mse_per_sample.tolist(),
            "rmse": mse_per_sample.sqrt().tolist(),
        }
        return reward.tolist(), info

    return reward_fn


# ---------------------------------------------------------------------------
# CLIP image–image similarity reward
# ---------------------------------------------------------------------------

class CLIPSimilarity:
    """Cosine similarity between student and teacher images in CLIP space."""

    _MEAN = [0.48145466, 0.4578275,  0.40821073]
    _STD  = [0.26862954, 0.26130258, 0.27577711]

    def __init__(self, device="cuda"):
        self.device = device
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(device).eval()
        self.model.requires_grad_(False)

    def encode_images(self, images) -> torch.Tensor:
        """Returns (B, D) L2-normalised float32 embeddings."""
        imgs = _pil_or_tensor_to_tensor(
            images, size=224,
            mean=self._MEAN, std=self._STD,
            device=self.device,
        )
        with torch.no_grad():
            embeds = self.model.get_image_features(pixel_values=imgs)
        return F.normalize(embeds.float(), dim=-1)

    def __call__(self, student_images, teacher_images, prompts=None):
        s = self.encode_images(student_images)   # (B, D)
        t = self.encode_images(teacher_images)   # (B, D)
        sims = F.cosine_similarity(s, t, dim=-1) # (B,)
        return sims.tolist(), None


def clip_similarity_reward_fn(teacher_pipeline, student_pipeline):
    """Factory — returns a reward_fn with signature (student, teacher, prompts)."""
    scorer = CLIPSimilarity(device=student_pipeline.device)
    def reward_fn(student_images, teacher_images, prompts=None):
        return scorer(student_images, teacher_images, prompts)
    reward_fn.feature_fn = scorer.encode_images
    return reward_fn


# ---------------------------------------------------------------------------
# CLIP text–image alignment reward  (student image ↔ prompt text)
# ---------------------------------------------------------------------------

def clip_text_image_alignment_reward_fn(student_pipeline):
    """
    Measures how well the STUDENT image matches the text prompt.
    Does NOT compare to teacher; useful as an auxiliary signal.
    """
    device = student_pipeline.device
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    model.requires_grad_(False)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    _MEAN = [0.48145466, 0.4578275,  0.40821073]
    _STD  = [0.26862954, 0.26130258, 0.27577711]

    def encode_images(images):
        imgs = _pil_or_tensor_to_tensor(images, 224, _MEAN, _STD, device)
        with torch.no_grad():
            embeds = model.get_image_features(pixel_values=imgs)
        return F.normalize(embeds.float(), dim=-1)

    def encode_text(prompts):
        inputs = processor(
            text=prompts, return_tensors="pt",
            padding=True, truncation=True,
        ).to(device)
        with torch.no_grad():
            embeds = model.get_text_features(**inputs)
        return F.normalize(embeds.float(), dim=-1)

    def reward_fn(student_images, teacher_images, prompts):
        if prompts is None:
            raise ValueError("prompts are required for text-image alignment reward")
        img_e  = encode_images(student_images)   # (B, D)
        text_e = encode_text(prompts)             # (B, D)
        sims = (img_e * text_e).sum(dim=-1)       # (B,)
        return sims.tolist(), None

    reward_fn.feature_fn = lambda si, ti: (encode_images(si), encode_text)
    return reward_fn


# ---------------------------------------------------------------------------
# DINOv2 image–image similarity reward
# ---------------------------------------------------------------------------

def dinov3_feature_extractor(device="cuda"):
    """Loads a DINOv2 ViT-B/16 via timm and returns an encode() closure."""
    import timm
    model = timm.create_model(
        'vit_base_patch16_dinov2', pretrained=True
    ).to(device).eval()
    model.requires_grad_(False)

    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]

    def encode(images):
        imgs = _pil_or_tensor_to_tensor(images, 512, _MEAN, _STD, device)
        with torch.no_grad():
            feats = model.forward_features(imgs)
        if isinstance(feats, dict):
            feats = feats.get('x_norm_patchtokens', feats.get('x', feats))
        return F.normalize(feats.float(), dim=-1)

    return encode


def dinov3_reward_fn(teacher_pipeline, student_pipeline):
    device = student_pipeline.device
    encode = dinov3_feature_extractor(device)

    def reward_fn(student_images, teacher_images, prompts=None):
        s = encode(student_images)
        t = encode(teacher_images)
        if s.dim() == 3:
            sims = F.cosine_similarity(s, t, dim=2).mean(dim=1)
        else:
            sims = F.cosine_similarity(s, t, dim=-1)
        return sims.tolist(), None

    def feature_fn(student_images, teacher_images):
        return encode(student_images), encode(teacher_images)

    reward_fn.feature_fn = feature_fn
    return reward_fn


# ---------------------------------------------------------------------------
# Perceptual (LPIPS) reward  — lower LPIPS = better = higher reward
# ---------------------------------------------------------------------------

def perception_reward_fn(teacher_pipeline, student_pipeline):
    import lpips
    device = student_pipeline.device
    model = lpips.LPIPS(net='vgg').to(device).eval()
    model.requires_grad_(False)

    def _prep(images):
        if isinstance(images, (list, tuple)) and isinstance(images[0], Image.Image):
            from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
            t = Compose([Resize(224), CenterCrop(224), ToTensor()])
            imgs = torch.stack([t(img.convert("RGB")) for img in images])
        elif isinstance(images, torch.Tensor):
            imgs = images.float()
            if imgs.max() > 1.0 + 1e-3:
                imgs = imgs / 255.0
            if imgs.shape[2] != 224 or imgs.shape[3] != 224:
                imgs = F.interpolate(imgs, (224, 224), mode="bicubic", align_corners=False)
        else:
            raise TypeError(f"Unsupported image type: {type(images)}")
        return (imgs * 2.0 - 1.0).to(device=device, dtype=torch.float32)

    def reward_fn(student_images, teacher_images, prompts=None):
        s = _prep(student_images)
        t = _prep(teacher_images)
        with torch.no_grad():
            dists = model(s, t)
        sims = -dists.view(-1)
        return sims.tolist(), None

    def feature_fn(student_images, teacher_images):
        return _prep(student_images), _prep(teacher_images)

    reward_fn.feature_fn = feature_fn
    return reward_fn


# ---------------------------------------------------------------------------
# Aesthetic score reward  (student image only, no teacher comparison)
# ---------------------------------------------------------------------------

def aesthetic_reward_fn(student_pipeline):
    from ddpo_pytorch.aesthetic_scorer import AestheticScorer
    device = student_pipeline.device
    scorer = AestheticScorer(dtype=torch.float32).to(device).eval()
    scorer.requires_grad_(False)

    def _prep(images):
        if isinstance(images, (list, tuple)) and isinstance(images[0], Image.Image):
            from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
            t = Compose([Resize(224), CenterCrop(224), ToTensor()])
            imgs = torch.stack([t(img.convert("RGB")) for img in images])
            return (imgs * 255).round().clamp(0, 255).to(torch.uint8).to(device)
        if isinstance(images, torch.Tensor):
            imgs = images.to(device)
            if imgs.dtype != torch.uint8:
                imgs = (imgs * 255).round().clamp(0, 255).to(torch.uint8)
            if imgs.shape[2] != 224 or imgs.shape[3] != 224:
                imgs_f = imgs.float() / 255.0
                imgs_f = F.interpolate(imgs_f, (224, 224), mode="bicubic", align_corners=False)
                imgs = (imgs_f * 255).round().clamp(0, 255).to(torch.uint8)
            return imgs
        raise TypeError(f"Unsupported image type: {type(images)}")

    def reward_fn(student_images, teacher_images, prompts=None):
        imgs = _prep(student_images)
        with torch.no_grad():
            scores = scorer(imgs)
        if isinstance(scores, torch.Tensor):
            return scores.tolist(), None
        return list(scores), None

    return reward_fn


# ---------------------------------------------------------------------------
# Composite reward  (the single entry-point used by the training script)
# ---------------------------------------------------------------------------

def get_reward_fn(reward_types, teacher_pipeline, student_pipeline):
    """
    Build a composite reward that sums all requested reward types.

    Supported reward_types
    ----------------------
    "clip"        – CLIP image-image cosine similarity (student ↔ teacher)
    "dino"        – DINOv2 patch-token cosine similarity (student ↔ teacher)
    "perception"  – negative LPIPS perceptual distance  (student ↔ teacher)
    "text_image"  – CLIP image-text alignment           (student image ↔ prompt)
    "aesthetic"   – LAION aesthetic score               (student image only)
    "mse"         – negative pixel MSE                  (student ↔ teacher)
                    This is the diffusion reconstruction loss as an RL reward.
                    reward = -MSE(student_pixels, teacher_pixels)
                    Range: roughly [-0.5, 0.0]. Combines naturally with clip
                    (which lives in [0,1]) for a joint pixel+semantic reward.

    Returns
    -------
    composite_reward_fn(student_images, teacher_images, prompts)
        → (total: list[float], details: dict[str, list[float]])
    """
    if isinstance(reward_types, str):
        reward_types = [reward_types]

    reward_fns = []
    for rtype in reward_types:
        if rtype == "clip":
            reward_fns.append(clip_similarity_reward_fn(teacher_pipeline, student_pipeline))
        elif rtype == "dino":
            reward_fns.append(dinov3_reward_fn(teacher_pipeline, student_pipeline))
        elif rtype == "perception":
            reward_fns.append(perception_reward_fn(teacher_pipeline, student_pipeline))
        elif rtype == "text_image":
            reward_fns.append(clip_text_image_alignment_reward_fn(student_pipeline))
        elif rtype == "aesthetic":
            reward_fns.append(aesthetic_reward_fn(student_pipeline))
        elif rtype == "mse":
            reward_fns.append(mse_reconstruction_reward_fn(teacher_pipeline, student_pipeline))
        else:
            raise ValueError(
                f"Unknown reward_type: '{rtype}'. "
                "Choose from: clip, dino, perception, text_image, aesthetic, mse."
            )

    def composite_reward_fn(student_images, teacher_images, prompts=None):
        total_np = None
        details  = {}
        for fn, rtype in zip(reward_fns, reward_types):
            rew, info = fn(student_images, teacher_images, prompts)

            if isinstance(rew, torch.Tensor):
                rew_np = rew.detach().cpu().float().numpy()
            else:
                rew_np = np.array(rew, dtype=np.float32)

            details[rtype] = rew_np.tolist()

            total_np = rew_np if total_np is None else total_np + rew_np

        return total_np.tolist(), details

    composite_reward_fn.reward_fns   = reward_fns
    composite_reward_fn.reward_types = reward_types
    return composite_reward_fn


# ---------------------------------------------------------------------------
# Legacy / standalone utilities (unchanged, kept for backward compatibility)
# ---------------------------------------------------------------------------

def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)
        images = [Image.fromarray(img) for img in images]
        buffers = [io.BytesIO() for _ in images]
        for img, buf in zip(images, buffers):
            img.save(buf, format="JPEG", quality=95)
        sizes = [buf.tell() / 1000 for buf in buffers]
        return np.array(sizes), {}
    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()
    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew, meta
    return _fn


def llava_strict_satisfaction():
    import requests, pickle
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO

    batch_size = 4
    url  = "http://127.0.0.1:8085"
    sess = requests.Session()
    sess.mount("http://", HTTPAdapter(max_retries=Retry(
        total=1000, backoff_factor=1,
        status_forcelist=[500], allowed_methods=False
    )))

    def _fn(images, prompts, metadata):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)
        n = len(images)
        images_batched   = np.array_split(images,   int(np.ceil(n / batch_size)))
        metadata_batched = np.array_split(metadata, int(np.ceil(n / batch_size)))
        all_scores, all_info = [], {"answers": []}
        for img_batch, meta_batch in zip(images_batched, metadata_batched):
            jpegs = []
            for img in img_batch:
                buf = io.BytesIO()
                Image.fromarray(img).save(buf, format="JPEG", quality=80)
                jpegs.append(buf.getvalue())
            resp = sess.post(url, data=pickle.dumps({
                "images": jpegs,
                "queries": [m["questions"] for m in meta_batch],
            }), timeout=120)
            rd = pickle.loads(resp.content)
            correct = np.array([
                [ans in r for ans, r in zip(m["answers"], responses)]
                for m, responses in zip(meta_batch, rd["outputs"])
            ])
            all_scores += correct.mean(axis=-1).tolist()
            all_info["answers"] += rd["outputs"]
        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}
    return _fn


def llava_bertscore():
    import requests, pickle
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO

    batch_size = 16
    url  = "http://127.0.0.1:8085"
    sess = requests.Session()
    sess.mount("http://", HTTPAdapter(max_retries=Retry(
        total=1000, backoff_factor=1,
        status_forcelist=[500], allowed_methods=False
    )))

    def _fn(images, prompts, metadata):
        del metadata
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)
        n = len(images)
        images_batched  = np.array_split(images,  int(np.ceil(n / batch_size)))
        prompts_batched = np.array_split(prompts, int(np.ceil(n / batch_size)))
        all_scores, all_info = [], {"precision": [], "f1": [], "outputs": []}
        for img_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpegs = []
            for img in img_batch:
                buf = io.BytesIO()
                Image.fromarray(img).save(buf, format="JPEG", quality=80)
                jpegs.append(buf.getvalue())
            resp = sess.post(url, data=pickle.dumps({
                "images":  jpegs,
                "queries": [["Answer concisely: what is going on in this image?"]] * len(img_batch),
                "answers": [[f"The image contains {p}"] for p in prompt_batch],
            }), timeout=120)
            rd = pickle.loads(resp.content)
            all_scores += np.array(rd["recall"]).squeeze().tolist()
            all_info["precision"] += np.array(rd["precision"]).squeeze().tolist()
            all_info["f1"]        += np.array(rd["f1"]).squeeze().tolist()
            all_info["outputs"]   += np.array(rd["outputs"]).squeeze().tolist()
        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}
    return _fn