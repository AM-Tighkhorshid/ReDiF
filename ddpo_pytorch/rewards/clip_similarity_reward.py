# File: ddpo_pytorch/rewards/clip_similarity_reward.py

import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

class CLIPSimilarity:
    def __init__(self, device="cuda"):
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.transform = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                      std=[0.26862954, 0.26130258, 0.27577711]),
        ])

    def encode_images(self, images):
        if isinstance(images[0], Image.Image):
            images = [self.transform(img.convert("RGB")).to(self.device) for img in images]
            images = torch.stack(images)
        elif isinstance(images, torch.Tensor):
            images = images.to(self.device)
            if images.max() > 1:
                images = images / 255.0
            if images.shape[1] == 3 and images.shape[2] != 224:
                images = torch.nn.functional.interpolate(images, size=(224, 224), mode="bicubic", align_corners=False)
            images = images.to(dtype=torch.float32)  # 👈 ensure correct dtype for CLIP
        else:
            raise TypeError("Images must be PIL or Tensor")

        with torch.no_grad():
            image_embeds = self.model.get_image_features(pixel_values=images)
            image_embeds = F.normalize(image_embeds, dim=-1)
        return image_embeds

    def __call__(self, student_images, teacher_images, prompts=None):
        student_embeds = self.encode_images(student_images)
        teacher_embeds = self.encode_images(teacher_images)
        sims = (student_embeds * teacher_embeds).sum(dim=-1)
        return sims.tolist(), None


def clip_similarity_reward_fn(teacher_pipeline, student_pipeline):
    device = student_pipeline.device
    scorer = CLIPSimilarity(device=device)
    def reward_fn(student_images, teacher_images, prompts=None):
        return scorer(student_images, teacher_images, prompts)
    return reward_fn
