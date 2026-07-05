"""
ddpo_pytorch/clip_distill.py

Core ReDiF reward: CLIP-space image-image similarity between a frozen
teacher's output and the student's output for the *same prompt*.

Renamed from `distill_clip.py` -> `clip_distill.py` purely for import-path
consistency with the rest of the package (`ddpo_pytorch.clip_distill`), and to
avoid the previous naming collision described below.

--------------------------------------------------------------------------
Bug this file fixes (naming collision that silently killed the old module)
--------------------------------------------------------------------------
The original implementation had BOTH of these on disk at the same time:

    ddpo_pytorch/rewards.py                          (a module)
    ddpo_pytorch/rewards/clip_similarity_reward.py    (a namespace package,
                                                        no __init__.py)

Python cannot have a module and a package share the same name inside the
same parent package. Because `rewards.py` is a regular module, it always
wins the import; `ddpo_pytorch/rewards/clip_similarity_reward.py` was
therefore **unreachable dead code** - `import ddpo_pytorch.rewards` gives you
the .py file, and there is no way to `import
ddpo_pytorch.rewards.clip_similarity_reward` once that happens. Anyone
editing the reward logic in the `rewards/` folder was silently editing code
that could never run.

Fix: delete `ddpo_pytorch/rewards/` entirely and keep exactly one
implementation of the CLIP distillation reward, living here.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import CLIPModel

ImageBatch = Union[Sequence[Image.Image], torch.Tensor]

_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def _resolve_device(preferred: Optional[torch.device]) -> torch.device:
    """
    Pick the correct per-rank CUDA device.

    Never fall back to a bare "cuda" string in a multi-GPU / Accelerate run:
    that always resolves to cuda:0 and, when every rank does it, causes every
    process to silently load a duplicate copy of the CLIP model on GPU 0,
    wasting memory on GPU 0 and eventually OOM-ing while other GPUs sit idle.
    """
    if preferred is not None:
        return torch.device(preferred)
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


class CLIPDistillationReward:
    """
    Computes reward_i = cos_sim(CLIP(student_image_i), CLIP(teacher_image_i)).

    This is the reward described in the ReDiF paper: it does not require any
    ground-truth label, only that the teacher and student were conditioned on
    the *same prompt* (and, ideally, the same initial noise - see
    `scripts/train_ppo_coco.py` for how that is enforced upstream).

    The reward is used purely as a black-box scalar for the PPO advantage;
    the CLIP encoder is not, and must not, be part of the policy-gradient
    computation graph. Wrapping every forward pass in `torch.no_grad()` here
    is therefore correct, not a bug: gradients for the student are supposed
    to flow through `log_prob(next_latent | latent)` (see
    `ddim_with_logprob.py`), never through the reward model itself.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        model_name: str = "openai/clip-vit-base-patch32",
        dtype: torch.dtype = torch.float32,
    ):
        self.device = _resolve_device(device)
        self.dtype = dtype
        self.model = (
            CLIPModel.from_pretrained(model_name).to(self.device, dtype=dtype).eval()
        )
        # Belt-and-suspenders: this reward model is never trained, so make
        # sure autograd never even builds a graph for it.
        self.model.requires_grad_(False)

        self._transform = Compose(
            [
                Resize(224, interpolation=Image.BICUBIC),
                CenterCrop(224),
                ToTensor(),
                Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
            ]
        )

    def _to_clip_input(self, images: ImageBatch) -> torch.Tensor:
        """Normalize a batch of PIL images or a (B,C,H,W) tensor to CLIP's
        expected 224x224, ImageNet/CLIP-normalized input."""
        if isinstance(images, (list, tuple)):
            if len(images) == 0:
                raise ValueError("Received an empty image batch.")
            if not isinstance(images[0], Image.Image):
                raise TypeError(
                    f"List input must contain PIL.Image, got {type(images[0])}"
                )
            batch = torch.stack(
                [self._transform(img.convert("RGB")) for img in images]
            )
            return batch.to(device=self.device, dtype=self.dtype)

        if isinstance(images, torch.Tensor):
            imgs = images.to(device=self.device, dtype=self.dtype)
            if imgs.max() > 1.0 + 1e-3:
                imgs = imgs / 255.0
            if imgs.shape[-2:] != (224, 224):
                imgs = F.interpolate(
                    imgs, size=(224, 224), mode="bicubic", align_corners=False
                )
            mean = torch.tensor(_CLIP_MEAN, device=self.device, dtype=self.dtype).view(
                1, 3, 1, 1
            )
            std = torch.tensor(_CLIP_STD, device=self.device, dtype=self.dtype).view(
                1, 3, 1, 1
            )
            return (imgs - mean) / std

        raise TypeError(f"Unsupported image batch type: {type(images)}")

    @torch.no_grad()
    def encode_images(self, images: ImageBatch) -> torch.Tensor:
        """Returns (B, D) L2-normalized CLIP image embeddings."""
        pixel_values = self._to_clip_input(images)
        embeds = self.model.get_image_features(pixel_values=pixel_values)
        return F.normalize(embeds.float(), dim=-1)

    @torch.no_grad()
    def __call__(
        self,
        student_images: ImageBatch,
        teacher_images: ImageBatch,
        prompts: Optional[List[str]] = None,
    ) -> Tuple[List[float], dict]:
        del prompts  # unused: this reward only compares images
        student_embeds = self.encode_images(student_images)  # (B, D)
        teacher_embeds = self.encode_images(teacher_images)  # (B, D)
        sims = F.cosine_similarity(student_embeds, teacher_embeds, dim=-1)  # (B,)
        info = {"clip_cosine_sim": sims.detach().float().cpu().tolist()}
        return sims.detach().float().cpu().tolist(), info


def clip_distillation_reward_fn(teacher_pipeline, student_pipeline):
    """
    Factory matching the composite-reward call signature used throughout
    `rewards.py`:

        reward_fn(student_images, teacher_images, prompts) -> (scores, info)

    Both pipelines are accepted (rather than just reading a global device)
    so this factory can be called *after* `accelerator.prepare()`, ensuring
    each rank builds its own CLIP model on its own GPU.
    """
    device = student_pipeline.device
    scorer = CLIPDistillationReward(device=device)

    def reward_fn(student_images, teacher_images, prompts=None):
        return scorer(student_images, teacher_images, prompts)

    # Exposed so callers (e.g. a future feature-level distillation loss) can
    # reuse the same encoder without reloading CLIP a second time.
    reward_fn.encode_images = scorer.encode_images
    return reward_fn