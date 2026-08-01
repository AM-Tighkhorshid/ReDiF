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
"perception"  - Meta Perception Encoder (PE) image-image cosine similarity
                (student <-> teacher). NOTE: "Perception Encoder" is NOT the
                same model family as LPIPS (see the "lpips" vs "perception"
                note below) - PE is a large-scale CLIP-style contrastive
                vision(-language) encoder (arXiv:2504.13181, Meta, Apr 2025),
                loaded here via timm's `vit_pe_core_base_patch16_224.fb`
                checkpoint. It measures similarity in a general-purpose
                semantic embedding space, not a human-perceptual-similarity-
                calibrated one.
"lpips"       - negative LPIPS perceptual distance (student <-> teacher).
                This is the metric that was previously (mis-)labeled
                "perception" in this file - LPIPS ("Learned Perceptual Image
                Patch Similarity") is a distance calibrated to match human
                judgments of patch similarity using a fixed, frozen backbone
                (VGG here), which is a different model family and a
                different training objective from Perception Encoder above.
                Kept as its own reward type since the two capture different
                notions of "similarity" and are not interchangeable.
"text_image"  - CLIP image-text alignment (student image <-> prompt).
"aesthetic"   - LAION aesthetic score (student image only).
"mse"         - negative pixel-space MSE (student <-> teacher):
                    reward = -MSE(student_pixels, teacher_pixels)
                A less-negative (higher) reward means the student image is
                closer to the teacher image in pixel space.
"kl"          - negative KL divergence between per-channel pixel-intensity
                histograms of the student and teacher images:
                    reward = -KL(P_teacher || P_student)
                See `kl_divergence_reward_fn` docstring for the exact
                histogram construction, bin count, and why the *forward* KL
                (teacher as reference distribution) is used here rather than
                the reverse KL.

Notes on why every encoder call below is wrapped in `torch.no_grad()`
----------------------------------------------------------------------
In DDPO-style PPO, the reward is a black-box scalar, not a differentiable
term in the loss. The student is updated purely through the policy-gradient
term (importance ratio x advantage), computed from
`log_prob(next_latent | latent)` in the PPO training loop - never by
back-propagating through a reward model. Every reward encoder here (CLIP,
DINOv3, Perception Encoder, LPIPS, the aesthetic scorer) is loaded frozen and
run under `torch.no_grad()` for that reason. If you ever want a
*differentiable* distillation term (e.g. an auxiliary feature-matching loss
added directly to the PPO loss instead of used as a reward), that must be
implemented separately in the training script with the encoder's gradients
allowed to flow into the student's log-prob term's inputs - it should NOT be
bolted onto these reward functions.
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
    WITHOUT any normalization. Used for pixel-space MSE and for the
    histogram-based KL reward.
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
# KL-divergence reconstruction reward
# ---------------------------------------------------------------------------

def _batched_channel_histogram(images: torch.Tensor, n_bins: int, value_range=(0.0, 1.0)) -> torch.Tensor:
    """
    Vectorized per-image, per-channel intensity histogram.

    images : (B, C, H, W) float tensor with values in `value_range`.
    Returns: (B, C, n_bins) tensor of raw (unnormalized) bin counts.

    Implemented with `scatter_add_` instead of a per-sample `torch.histc`
    loop - `torch.histc` only supports a single flat tensor per call (no
    batch dimension), so a Python-level loop over B*C histograms would be
    both slow and awkward to keep on-device across a whole rollout batch.
    """
    B, C, H, W = images.shape
    lo, hi = value_range
    bin_width = (hi - lo) / n_bins
    # clamp so a pixel exactly at `hi` still lands in the last bin instead of
    # overflowing to index n_bins
    clamped = images.clamp(lo, hi - 1e-6)
    bin_idx = ((clamped - lo) / bin_width).long().clamp(0, n_bins - 1)  # (B, C, H, W)
    bin_idx = bin_idx.reshape(B, C, H * W)

    counts = torch.zeros(B, C, n_bins, device=images.device, dtype=images.dtype)
    counts.scatter_add_(dim=2, index=bin_idx, src=torch.ones_like(bin_idx, dtype=images.dtype))
    return counts


def kl_divergence_reward_fn(teacher_pipeline, student_pipeline, n_bins: int = 32, eps: float = 1e-6):
    """
    Negative KL divergence between student and teacher images, estimated
    from per-channel pixel-intensity histograms:

        reward = -KL(P_teacher || P_student)
                = -sum_bins  P_teacher(bin) * log( P_teacher(bin) / P_student(bin) )

    This is deliberately the SAME kind of cheap, no-encoder, purely
    pixel-space signal as `mse_reconstruction_reward_fn` (as requested) - it
    does not load any extra network, it just compares the two images'
    intensity distributions directly. Two design choices worth being
    explicit about:

    1. Direction of the KL: this uses the *forward* KL, KL(teacher ||
       student), i.e. the teacher's histogram is the reference ("true")
       distribution and the student's is the approximation. This mirrors
       the usual distillation convention (student approximates teacher) and
       is mode-covering: it penalizes the student most heavily for having
       near-zero probability mass in an intensity bin the teacher image
       actually uses. The reverse KL(student || teacher) would instead most
       heavily penalize the student for putting mass somewhere the teacher
       doesn't - a different (mode-seeking) failure mode. Swap the two `_p`
       tensors in the `kl_per_channel` line below if you want that instead.
    2. Per-channel, not joint-RGB: histograms are computed independently
       per color channel and the resulting per-channel KLs are summed. This
       is a standard simplification (a full joint RGB histogram needs
       exponentially more bins to stay statistically meaningful at typical
       image resolutions) but it does mean this reward is blind to
       cross-channel correlation and, like MSE, completely blind to spatial
       structure - two images with identical per-channel intensity
       histograms but totally different content would score a KL of ~0.
       Use alongside "clip"/"dino"/"perception"/"lpips" if spatial/semantic
       structure matters for your use case, not as a spatial-structure
       reward on its own.

    `eps` is Laplace-style smoothing added to every bin before normalizing,
    so an empty student bin where the teacher has mass doesn't produce a
    +inf KL from a single outlier pixel.
    """
    device = student_pipeline.device

    def reward_fn(student_images, teacher_images, prompts=None):
        s = _to_float_tensor(student_images, device)
        t = _to_float_tensor(teacher_images, device)
        if s.shape != t.shape:
            t = F.interpolate(t, size=s.shape[2:], mode="bicubic", align_corners=False)

        with torch.no_grad():
            s_counts = _batched_channel_histogram(s, n_bins=n_bins)  # (B, C, n_bins)
            t_counts = _batched_channel_histogram(t, n_bins=n_bins)

            s_p = (s_counts + eps) / (s_counts.sum(dim=-1, keepdim=True) + eps * n_bins)
            t_p = (t_counts + eps) / (t_counts.sum(dim=-1, keepdim=True) + eps * n_bins)

            # KL(teacher || student), per (batch, channel), then summed over
            # channels - see docstring point 1 for the direction rationale.
            kl_per_channel = (t_p * (t_p.log() - s_p.log())).sum(dim=-1)  # (B, C)
            kl_per_sample = kl_per_channel.sum(dim=-1)  # (B,)
            reward = -kl_per_sample

        info = {"kl": kl_per_sample.tolist()}
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
# Perception Encoder (PE) image-image similarity reward
#
# NOT the same thing as LPIPS (see `lpips_reward_fn` below and the module
# docstring). Perception Encoder (Bolya et al., "Perception Encoder: The
# best visual embeddings are not at the output of the network",
# arXiv:2504.13181, Meta, Apr 2025) is a family of large-scale CLIP-style
# vision(-language) encoders trained with contrastive vision-language
# learning, released via https://github.com/facebookresearch/perception_models
# and integrated into timm (as of the `.fb`-suffixed `vit_pe_core_*`
# checkpoints, timm >= ~1.0.16). It produces general-purpose semantic image
# embeddings - closer in spirit to CLIP/DINO here than to LPIPS, which is a
# small, fixed backbone (VGG/AlexNet/SqueezeNet) *calibrated specifically to
# match human patch-similarity judgments* rather than trained for general
# semantic retrieval/classification.
# ---------------------------------------------------------------------------

def perception_encoder_feature_extractor(device="cuda", model_name="vit_pe_core_base_patch16_224.fb"):
    """Loads a Meta Perception Encoder (PE-Core) ViT via timm and returns an
    encode() closure producing a single global (pooled) embedding per image.

    Requires a recent timm build with PE support (`pip install -U timm`).
    Other available sizes/resolutions include:
        vit_pe_core_large_patch14_336.fb
        vit_pe_core_gigantic_patch14_448.fb
    """
    import timm
    from huggingface_hub import logout as _hf_logout

    
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    os.environ.pop("HUGGINGFACE_HUB_TOKEN", None)

    try:
        _hf_logout()
    except Exception:
        pass

    try:
        # num_classes=0 drops the classifier head and returns the pooled
        # embedding directly from model(imgs) - the PE-Core analogue of
        # CLIP's `get_image_features`.
        model = timm.create_model(model_name, pretrained=True, num_classes=0).to(device)
    except RuntimeError as e:
        raise RuntimeError(
            f"Could not load '{model_name}' ({e}). Perception Encoder entrypoints "
            "were only added to timm relatively recently - try `pip install -U timm`."
        ) from e
    model.eval()
    model.requires_grad_(False)

    try:
        data_cfg = timm.data.resolve_data_config({}, model=model)
        mean = list(data_cfg.get("mean", (0.5, 0.5, 0.5)))
        std = list(data_cfg.get("std", (0.5, 0.5, 0.5)))
        size = data_cfg.get("input_size", (3, 224, 224))[-1]
    except Exception:
        # Fall back to PE-Core's documented normalization/resolution if
        # timm's own config resolution fails for some reason.
        mean, std, size = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], 224

    def encode(images):
        imgs = _pil_or_tensor_to_tensor(images, size, mean, std, device)
        with torch.no_grad():
            embeds = model(imgs)  # (B, embed_dim) - already globally pooled
        return F.normalize(embeds.float(), dim=-1)

    return encode


def perception_encoder_reward_fn(teacher_pipeline, student_pipeline):
    device = student_pipeline.device
    encode = perception_encoder_feature_extractor(device)

    def reward_fn(student_images, teacher_images, prompts=None):
        s_feats = encode(student_images)
        t_feats = encode(teacher_images)
        sims = F.cosine_similarity(s_feats, t_feats, dim=-1)
        return sims.tolist(), None

    def feature_fn(student_images, teacher_images):
        return encode(student_images), encode(teacher_images)

    reward_fn.feature_fn = feature_fn
    return reward_fn


# ---------------------------------------------------------------------------
# LPIPS perceptual-distance reward - lower LPIPS = better = higher reward
#
# This is the metric that lived under the name "perception" before; renamed
# to "lpips" to avoid clashing with the actual Perception Encoder reward
# above, since the two measure different things and neither is a substitute
# for the other (see module docstring).
# ---------------------------------------------------------------------------

def lpips_reward_fn(teacher_pipeline, student_pipeline):
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
    "perception": lambda teacher_pipeline, student_pipeline: perception_encoder_reward_fn(
        teacher_pipeline, student_pipeline
    ),
    "lpips": lambda teacher_pipeline, student_pipeline: lpips_reward_fn(
        teacher_pipeline, student_pipeline
    ),
    "text_image": lambda teacher_pipeline, student_pipeline: clip_text_image_alignment_reward_fn(
        student_pipeline
    ),
    "aesthetic": lambda teacher_pipeline, student_pipeline: aesthetic_reward_fn(student_pipeline),
    "mse": lambda teacher_pipeline, student_pipeline: mse_reconstruction_reward_fn(
        teacher_pipeline, student_pipeline
    ),
    "kl": lambda teacher_pipeline, student_pipeline: kl_divergence_reward_fn(
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