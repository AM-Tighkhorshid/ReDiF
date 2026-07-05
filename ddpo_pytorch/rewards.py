"""
ddpo_pytorch/rewards.py - reward registry for teacher -> student distillation via PPO.

Unified call signature for every reward function returned by `get_reward_fn`:

    fn(student_images, teacher_images, prompts) -> (scores: list[float], info: dict | None)

`student_images` / `teacher_images` are (B, C, H, W) float tensors in [0, 1]
(or lists of PIL images); `prompts` is the list[str] of text prompts used to
condition both models for that batch.

Reward types
------------
"clip"        - CLIP image-image cosine similarity (student <-> teacher).
                This is the reward described in the ReDiF paper; the actual
                implementation lives in `ddpo_pytorch/clip_distill.py`.
"dino"        - DINOv3 patch-token cosine similarity (student <-> teacher).
                Backed by timm's `vit_base_patch16_dinov3.lvd1689m` checkpoint.
"perception"  - negative LPIPS perceptual distance (student <-> teacher).
"text_image"  - CLIP image-text alignment (student image <-> prompt).
"aesthetic"   - LAION aesthetic score (student image only).
"mse"         - negative pixel-space MSE (student <-> teacher):
                    reward = -MSE(student_pixels, teacher_pixels)
                A less-negative (higher) reward means the student image is
                closer to the teacher image in pixel space.

Notes on why every encoder call below is wrapped in `torch.no_grad()`
----------------------------------------------------------------------
In DDPO-style PPO, the reward is a black-box scalar, not a differentiable
term in the loss. The student is updated purely through the policy-gradient
term (importance ratio x advantage), computed from
`log_prob(next_latent | latent)` in the PPO training loop - never by
back-propagating through a reward model. Every reward encoder here (CLIP,
DINOv3, LPIPS, the aesthetic scorer) is loaded frozen and run under
`torch.no_grad()` for that reason. If you ever want a *differentiable*
distillation term (e.g. an auxiliary feature-matching loss added directly to
the PPO loss instead of used as a reward), that must be implemented
separately in the training script with the encoder's gradients allowed to
flow into the student's log-prob term's inputs - it should NOT be bolted
onto these reward functions.
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import CLIPModel, CLIPProcessor

from ddpo_pytorch.clip_distill import clip_distillation_reward_fn

ImageBatch = Union[Sequence[Image.Image], torch.Tensor]
RewardFn = "callable[[ImageBatch, ImageBatch, Optional[List[str]]], Tuple[List[float], Optional[dict]]]"


# ---------------------------------------------------------------------------
# Shared image-preprocessing helpers
# ---------------------------------------------------------------------------

def _pil_or_tensor_to_tensor(images, size: int, mean, std, device) -> torch.Tensor:
    """
    Accept a list[PIL.Image] or a (B,C,H,W) float tensor in [0,1] or [0,255].
    Returns a (B,C,size,size) float32 tensor on `device`, normalized with the
    supplied mean/std.
    """
    transform = Compose(
        [
            Resize(size, interpolation=Image.BICUBIC),
            CenterCrop(size),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ]
    )

    if isinstance(images, (list, tuple)) and isinstance(images[0], Image.Image):
        imgs = torch.stack([transform(img.convert("RGB")) for img in images])
        return imgs.to(device=device, dtype=torch.float32)

    if isinstance(images, torch.Tensor):
        imgs = images.to(device=device, dtype=torch.float32)
        if imgs.max() > 1.0 + 1e-3:  # [0,255] -> [0,1]
            imgs = imgs / 255.0
        if imgs.shape[2] != size or imgs.shape[3] != size:
            imgs = F.interpolate(imgs, size=(size, size), mode="bicubic", align_corners=False)
        mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
        return (imgs - mean_t) / std_t

    raise TypeError(f"Images must be list[PIL.Image] or torch.Tensor, got {type(images)}")


def _to_float_tensor(images, device) -> torch.Tensor:
    """
    Convert images to a (B,C,H,W) float32 tensor in [0,1] on `device`,
    WITHOUT any normalization. Used for pixel-space MSE.
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
# MSE reconstruction reward
# ---------------------------------------------------------------------------

def mse_reconstruction_reward_fn(teacher_pipeline, student_pipeline):
    """Negative pixel-space MSE between student and teacher output images."""
    device = student_pipeline.device

    def reward_fn(student_images, teacher_images, prompts=None):
        s = _to_float_tensor(student_images, device)
        t = _to_float_tensor(teacher_images, device)
        if s.shape != t.shape:
            t = F.interpolate(t, size=s.shape[2:], mode="bicubic", align_corners=False)

        with torch.no_grad():
            mse_per_sample = ((s - t) ** 2).mean(dim=[1, 2, 3])
            reward = -mse_per_sample

        info = {"mse": mse_per_sample.tolist(), "rmse": mse_per_sample.sqrt().tolist()}
        return reward.tolist(), info

    return reward_fn


# ---------------------------------------------------------------------------
# CLIP text-image alignment reward (student image <-> prompt)
# ---------------------------------------------------------------------------

def clip_text_image_alignment_reward_fn(student_pipeline):
    """Measures how well the STUDENT image matches the text prompt. Does not
    compare to the teacher; useful as an auxiliary signal alongside "clip"."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else student_pipeline.device
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    model.requires_grad_(False)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    _MEAN = [0.48145466, 0.4578275, 0.40821073]
    _STD = [0.26862954, 0.26130258, 0.27577711]

    def encode_images(images):
        imgs = _pil_or_tensor_to_tensor(images, 224, _MEAN, _STD, device)
        with torch.no_grad():
            embeds = model.get_image_features(pixel_values=imgs)
        return F.normalize(embeds.float(), dim=-1)

    def encode_text(prompts):
        inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            embeds = model.get_text_features(**inputs)
        return F.normalize(embeds.float(), dim=-1)

    def reward_fn(student_images, teacher_images, prompts):
        if prompts is None:
            raise ValueError("prompts are required for text-image alignment reward")
        img_e = encode_images(student_images)
        text_e = encode_text(prompts)
        sims = (img_e * text_e).sum(dim=-1)
        return sims.tolist(), None

    return reward_fn


# ---------------------------------------------------------------------------
# DINOv3 image-image similarity reward
# ---------------------------------------------------------------------------

def dinov3_feature_extractor(device="cuda"):
    """Loads a DINOv3 ViT-B/16 via timm and returns an encode() closure.

    Requires a recent timm build with DINOv3 support
    (`pip install -U timm`). The registered name needs the pretrained-weight
    tag suffix - `vit_base_patch16_dinov3` alone is NOT a valid timm model
    name and will raise `RuntimeError: Unknown model`.
    """
    import timm

    # Other valid sizes (all need the `.lvd1689m` / `.sat493m` tag):
    #   vit_small_patch16_dinov3.lvd1689m
    #   vit_base_patch16_dinov3.lvd1689m
    #   vit_large_patch16_dinov3.lvd1689m / .sat493m
    #   vit_7b_patch16_dinov3.lvd1689m / .sat493m
    model_name = "vit_base_patch16_dinov3.lvd1689m"
    try:
        model = timm.create_model(model_name, pretrained=True).to(device)
    except RuntimeError as e:
        raise RuntimeError(
            f"Could not load '{model_name}' ({e}). DINOv3 entrypoints were "
            "only added to timm relatively recently - try `pip install -U timm`."
        ) from e
    model.eval()
    model.requires_grad_(False)

    # Number of non-patch tokens (CLS + register tokens, if any) prepended
    # to the patch-token sequence by timm's ViT implementation. We slice
    # these off so the reward reflects pure patch-level (local/dense)
    # similarity rather than mixing in the global CLS embedding.
    num_prefix_tokens = getattr(model, "num_prefix_tokens", 1)

    _MEAN = [0.485, 0.456, 0.406]
    _STD = [0.229, 0.224, 0.225]

    def encode(images):
        imgs = _pil_or_tensor_to_tensor(images, 512, _MEAN, _STD, device)
        with torch.no_grad():
            feats = model.forward_features(imgs)
            if isinstance(feats, dict):
                # HF-style wrapper (e.g. transformers Dinov3Model) - already
                # patch-tokens-only.
                feats = feats.get("x_norm_patchtokens", feats.get("x", feats))
            elif feats.dim() == 3 and num_prefix_tokens > 0:
                # Plain timm ViT: [CLS, (registers...), patch_tokens...]
                feats = feats[:, num_prefix_tokens:, :]
        return F.normalize(feats.float(), dim=-1)

    return encode


def dinov3_reward_fn(teacher_pipeline, student_pipeline):
    device = student_pipeline.device
    encode = dinov3_feature_extractor(device)

    def reward_fn(student_images, teacher_images, prompts=None):
        s_feats = encode(student_images)
        t_feats = encode(teacher_images)
        if s_feats.dim() == 3 and t_feats.dim() == 3:
            # [batch, num_patch_tokens, dim] -> per-token cosine sim, then
            # averaged over tokens for a single scalar per image.
            sims = F.cosine_similarity(s_feats, t_feats, dim=2).mean(dim=1)
        else:
            sims = F.cosine_similarity(s_feats, t_feats, dim=-1)
        return sims.tolist(), None

    def feature_fn(student_images, teacher_images):
        return encode(student_images), encode(teacher_images)

    reward_fn.feature_fn = feature_fn
    return reward_fn


# ---------------------------------------------------------------------------
# Perceptual (LPIPS) reward - lower LPIPS = better = higher reward
# ---------------------------------------------------------------------------

def perception_reward_fn(teacher_pipeline, student_pipeline):
    import lpips

    device = student_pipeline.device
    model = lpips.LPIPS(net="vgg").to(device).eval()
    model.requires_grad_(False)

    def _prep(images):
        if isinstance(images, (list, tuple)) and isinstance(images[0], Image.Image):
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
            # NOTE: use view(-1) instead of squeeze(). squeeze() collapses a
            # batch of size 1 down to a 0-d tensor, and `.tolist()` on a 0-d
            # tensor returns a bare float instead of a list, which breaks
            # any downstream code (e.g. composite_reward_fn below) expecting
            # one score per sample as a list.
            dists = dists.view(-1)
        return (-dists).tolist(), None

    def feature_fn(student_images, teacher_images):
        return _prep(student_images), _prep(teacher_images)

    reward_fn.feature_fn = feature_fn
    return reward_fn


# ---------------------------------------------------------------------------
# Aesthetic score reward (student image only, no teacher comparison)
# ---------------------------------------------------------------------------

def aesthetic_reward_fn(student_pipeline):
    from ddpo_pytorch.aesthetic_scorer import AestheticScorer

    device = student_pipeline.device
    scorer = AestheticScorer(dtype=torch.float32).to(device).eval()
    scorer.requires_grad_(False)

    def _prep(images):
        if isinstance(images, (list, tuple)) and isinstance(images[0], Image.Image):
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
# Composite reward (the single entry-point used by the training script)
# ---------------------------------------------------------------------------

_REWARD_BUILDERS = {
    "clip": lambda teacher_pipeline, student_pipeline: clip_distillation_reward_fn(
        teacher_pipeline, student_pipeline
    ),
    "dino": lambda teacher_pipeline, student_pipeline: dinov3_reward_fn(
        teacher_pipeline, student_pipeline
    ),
    "perception": lambda teacher_pipeline, student_pipeline: perception_reward_fn(
        teacher_pipeline, student_pipeline
    ),
    "text_image": lambda teacher_pipeline, student_pipeline: clip_text_image_alignment_reward_fn(
        student_pipeline
    ),
    "aesthetic": lambda teacher_pipeline, student_pipeline: aesthetic_reward_fn(student_pipeline),
    "mse": lambda teacher_pipeline, student_pipeline: mse_reconstruction_reward_fn(
        teacher_pipeline, student_pipeline
    ),
}


def get_reward_fn(reward_types: Union[str, List[str]], teacher_pipeline, student_pipeline):
    """
    Build a composite reward that sums all requested reward types.

    Returns
    -------
    composite_reward_fn(student_images, teacher_images, prompts)
        -> (total: list[float], details: dict[str, list[float]])
    """
    if isinstance(reward_types, str):
        reward_types = [reward_types]
    if not reward_types:
        raise ValueError("reward_types must contain at least one reward name.")

    reward_fns = []
    for rtype in reward_types:
        builder = _REWARD_BUILDERS.get(rtype)
        if builder is None:
            raise ValueError(
                f"Unknown reward_type: '{rtype}'. Choose from: {sorted(_REWARD_BUILDERS)}."
            )
        reward_fns.append(builder(teacher_pipeline, student_pipeline))

    def composite_reward_fn(student_images, teacher_images, prompts=None):
        total_np = None
        details: Dict[str, List[float]] = {}
        for fn, rtype in zip(reward_fns, reward_types):
            rew, info = fn(student_images, teacher_images, prompts)
            rew_np = (
                rew.detach().cpu().float().numpy()
                if isinstance(rew, torch.Tensor)
                else np.array(rew, dtype=np.float32)
            )
            details[rtype] = rew_np.tolist()
            total_np = rew_np if total_np is None else total_np + rew_np

        return total_np.tolist(), details

    composite_reward_fn.reward_fns = reward_fns
    composite_reward_fn.reward_types = reward_types
    return composite_reward_fn