import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import io

# === DINOv3 Reward ===
def dinov3_feature_extractor(device="cuda"):
    import timm
    model = timm.create_model('vit_base_patch16_dinov3', pretrained=True).to(device)
    model.eval()
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    def encode(images):
        if isinstance(images[0], Image.Image):
            images = [transform(img.convert("RGB")).to(device) for img in images]
            images = torch.stack(images)
        elif isinstance(images, torch.Tensor):
            images = images.to(device)
            if images.max() > 1:
                images = images / 255.0
            if images.shape[1] == 3 and images.shape[2] != 512:
                images = torch.nn.functional.interpolate(images, size=(512, 512), mode="bicubic", align_corners=False)
            images = images.to(dtype=torch.float32)
        else:
            raise TypeError("Images must be PIL or Tensor")
        with torch.no_grad():
            feats = model.forward_features(images)
            if isinstance(feats, dict) and 'x_norm_patchtokens' in feats:
                feats = feats['x_norm_patchtokens']
            feats = F.normalize(feats, dim=-1)
        return feats
    return encode

def dinov3_reward_fn(teacher_pipeline, student_pipeline):
    device = student_pipeline.device
    encode = dinov3_feature_extractor(device)
    def reward_fn(student_images, teacher_images, prompts=None):
        s_feats = encode(student_images)
        t_feats = encode(teacher_images)
        # If features are [batch, num_tokens, dim], compute cosine similarity for each token, then mean over tokens
        if s_feats.dim() == 3 and t_feats.dim() == 3:
            # [batch, num_tokens, dim]
            sims = F.cosine_similarity(s_feats, t_feats, dim=2)  # [batch, num_tokens]
            sims = sims.mean(dim=1)  # [batch]
        else:
            sims = F.cosine_similarity(s_feats, t_feats, dim=-1)  # [batch]
        # print(f"[DINOv3 Reward] reward shape: {sims.shape}")
        return sims.tolist(), None
    def feature_fn(student_images, teacher_images):
        s_feats = encode(student_images)
        t_feats = encode(teacher_images)
        return s_feats, t_feats
    reward_fn.feature_fn = feature_fn
    return reward_fn

# === Perception Reward (LPIPS) ===
def perception_feature_extractor(device="cuda"):
    import lpips
    model = lpips.LPIPS(net='vgg').to(device)
    model.eval()
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    def encode(images):
        if isinstance(images[0], Image.Image):
            images = [transform(img.convert("RGB")).unsqueeze(0).to(device) for img in images]
            images = torch.cat(images, dim=0)
        elif isinstance(images, torch.Tensor):
            images = images.to(device)
            if images.max() > 1:
                images = images / 255.0
            if images.shape[1] == 3 and images.shape[2] != 224:
                images = torch.nn.functional.interpolate(images, size=(224, 224), mode="bicubic", align_corners=False)
            images = images.to(dtype=torch.float32)
        else:
            raise TypeError("Images must be PIL or Tensor")
        return images
    return encode, model

def perception_reward_fn(teacher_pipeline, student_pipeline):
    device = student_pipeline.device
    encode, model = perception_feature_extractor(device)
    def reward_fn(student_images, teacher_images, prompts=None):
        s_imgs = encode(student_images)
        t_imgs = encode(teacher_images)
        with torch.no_grad():
            dists = model(s_imgs, t_imgs)
            sims = -dists.squeeze()
        # print(f"[Perception Reward] reward shape: {sims.shape}")
        return sims.tolist(), None
    def feature_fn(student_images, teacher_images):
        s_imgs = encode(student_images)
        t_imgs = encode(teacher_images)
        return s_imgs, t_imgs
    reward_fn.feature_fn = feature_fn
    return reward_fn


# === Text-Image Alignment Reward ===
def clip_text_image_alignment_reward_fn(student_pipeline):
    from transformers import CLIPProcessor, CLIPModel
    device = student_pipeline.device
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
    transform = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                  std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    def encode_images(images):
        if isinstance(images[0], Image.Image):
            images = [transform(img.convert("RGB")).to(device) for img in images]
            images = torch.stack(images)
        elif isinstance(images, torch.Tensor):
            images = images.to(device)
            if images.max() > 1:
                images = images / 255.0
            if images.shape[1] == 3 and images.shape[2] != 224:
                images = torch.nn.functional.interpolate(images, size=(224, 224), mode="bicubic", align_corners=False)
            images = images.to(dtype=torch.float32)
        else:
            raise TypeError("Images must be PIL or Tensor")
        with torch.no_grad():
            image_embeds = model.get_image_features(pixel_values=images)
            image_embeds = F.normalize(image_embeds, dim=-1)
        return image_embeds
    def encode_text(prompts):
        inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_embeds = model.get_text_features(**inputs)
            text_embeds = F.normalize(text_embeds, dim=-1)
        return text_embeds
    def reward_fn(student_images, teacher_images, prompts):
        # Only use student_images and prompts
        if prompts is None:
            raise ValueError("prompts required for text-image alignment reward")
        image_embeds = encode_images(student_images)
        text_embeds = encode_text(prompts)
        sims = (image_embeds * text_embeds).sum(dim=-1)
        # print(f"[Text-Image Alignment Reward] reward shape: {sims.shape}")
        return sims.tolist(), None
    def feature_fn(student_images, prompts):
        image_embeds = encode_images(student_images)
        text_embeds = encode_text(prompts)
        return image_embeds, text_embeds
    reward_fn.feature_fn = feature_fn
    return reward_fn

# === Composite Reward Function ===
def get_reward_fn(reward_types, teacher_pipeline, student_pipeline):
    """
    reward_types: str or list of str, e.g. 'clip' or ['clip', 'dino', 'aesthetic', 'text_image']
    Returns a function that computes all selected rewards and returns their sum and details.
    """
    if isinstance(reward_types, str):
        reward_types = [reward_types]
    reward_fns = []
    for rtype in reward_types:
        if rtype == "clip":
            reward_fns.append(clip_similarity_reward_fn(teacher_pipeline, student_pipeline))
        elif rtype == "dino":
            reward_fns.append(dinov3_reward_fn(teacher_pipeline, student_pipeline))
        elif rtype == "aesthetic":
            reward_fns.append(aesthetic_score())
        elif rtype == "perception":
            reward_fns.append(perception_reward_fn(teacher_pipeline, student_pipeline))
        elif rtype == "text_image":
            reward_fns.append(clip_text_image_alignment_reward_fn(student_pipeline))
        else:
            raise ValueError(f"Unknown reward_type: {rtype}")
    def composite_reward_fn(student_images, teacher_images, prompts=None):
        total = None
        details = {}
        for fn, rtype in zip(reward_fns, reward_types):
            if rtype == "text_image":
                rew, info = fn(student_images, teacher_images, prompts)
            else:
                rew, info = fn(student_images, teacher_images, prompts)
            details[rtype] = rew
            # Handle torch.Tensor on CUDA
            if isinstance(rew, torch.Tensor):
                rew_np = rew.detach().cpu().numpy()
            else:
                rew_np = np.array(rew)
            if total is None:
                total = rew_np
            else:
                total += rew_np
        return total.tolist(), details
    composite_reward_fn.reward_fns = reward_fns
    return composite_reward_fn

from PIL import Image
import io
import numpy as np
import torch

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
        sims = F.cosine_similarity(student_embeds, teacher_embeds, dim=-1)
        # print(f"[CLIP Reward] reward shape: {sims.shape}")
        return sims.tolist(), None


def clip_similarity_reward_fn(teacher_pipeline, student_pipeline):
    device = student_pipeline.device
    scorer = CLIPSimilarity(device=device)
    def reward_fn(student_images, teacher_images, prompts=None):
        return scorer(student_images, teacher_images, prompts)
    return reward_fn



def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew, meta

    return _fn


def aesthetic_score():
    from ddpo_pytorch.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


def llava_strict_satisfaction():
    """Submits images to LLaVA and computes a reward by matching the responses to ground truth answers directly without
    using BERTScore. Prompt metadata must have "questions" and "answers" keys. See
    https://github.com/kvablack/LLaVA-server for server-side code.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 4
    url = "http://127.0.0.1:8085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadata_batched = np.array_split(metadata, np.ceil(len(metadata) / batch_size))

        all_scores = []
        all_info = {
            "answers": [],
        }
        for image_batch, metadata_batch in zip(images_batched, metadata_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "queries": [m["questions"] for m in metadata_batch],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)

            response_data = pickle.loads(response.content)

            correct = np.array(
                [
                    [ans in resp for ans, resp in zip(m["answers"], responses)]
                    for m, responses in zip(metadata_batch, response_data["outputs"])
                ]
            )
            scores = correct.mean(axis=-1)

            all_scores += scores.tolist()
            all_info["answers"] += response_data["outputs"]

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn


def llava_bertscore():
    """Submits images to LLaVA and computes a reward by comparing the responses to the prompts using BERTScore. See
    https://github.com/kvablack/LLaVA-server for server-side code.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 16
    url = "http://127.0.0.1:8085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del metadata
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))

        all_scores = []
        all_info = {
            "precision": [],
            "f1": [],
            "outputs": [],
        }
        for image_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "queries": [["Answer concisely: what is going on in this image?"]]
                * len(image_batch),
                "answers": [
                    [f"The image contains {prompt}"] for prompt in prompt_batch
                ],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)

            response_data = pickle.loads(response.content)

            # use the recall score as the reward
            scores = np.array(response_data["recall"]).squeeze()
            all_scores += scores.tolist()

            # save the precision and f1 scores for analysis
            all_info["precision"] += (
                np.array(response_data["precision"]).squeeze().tolist()
            )
            all_info["f1"] += np.array(response_data["f1"]).squeeze().tolist()
            all_info["outputs"] += np.array(response_data["outputs"]).squeeze().tolist()

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn
