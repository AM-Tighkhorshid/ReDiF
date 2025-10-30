import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import AutoTokenizer
from accelerate import Accelerator
from tqdm import tqdm
import json
# import itertools # اگر میخواستیم بر اساس Step تمرین دهیم، اما فعلا نیازی نیست

# ============================================================
# Config
# ============================================================
DATA_ROOT = "./coco_dataset"
TRAIN_IMAGES = os.path.join(DATA_ROOT, "val2017")
ANNOTATION_FILE = os.path.join(DATA_ROOT, "annotations/captions_val2017.json")
MODEL_NAME = "/media/external20/amirhossein_tighkhorshid/diffusion_distillation/ddpo-pytorch-main/ddpo-pytorch-main/checkpoints/epoch_9"
OUTPUT_DIR = "./checkpoints2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === تغییرات کلیدی در Config ===
EPOCHS = 30                # افزایش اپوک‌ها برای تمرین کامل
BATCH_SIZE = 16            # ثابت (بستگی به VRAM شما دارد)
# NUM_SAMPLES_PER_EPOCH = 10_000 # حذف شد، از کل دیتاست استفاده می‌کنیم
IMAGE_SIZE = 512
LEARNING_RATE = 1e-5       # نرخ یادگیری برای UNet
TEXT_ENCODER_LR = 1e-6     # نرخ یادگیری پایین‌تر برای Text Encoder
GRAD_ACCUM_STEPS = 32      # افزایش چشمگیر برای بچ سایز مؤثر 512 = 32 * 16

# ============================================================
# Dataset
# ============================================================
class COCODataset(Dataset):
    # --- تغییر: 'num_samples' حذف شد ---
    def __init__(self, image_dir, annotation_file):
        with open(annotation_file, "r") as f:
            data = json.load(f)
        self.images = {img["id"]: img["file_name"] for img in data["images"]}
        self.annotations = data["annotations"]
        self.image_dir = image_dir
        
        # --- تغییر: به جای سمپل کردن، از کل دیتاست استفاده می‌کنیم ---
        self.selected = self.annotations 
        
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=Image.BICUBIC),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        
        # --- تغییر: 'self.num_samples' و '_resample()' حذف شدند ---

    # --- تغییر: متد '_resample' حذف شد ---

    def __len__(self):
        return len(self.selected)

    def __getitem__(self, idx):
        ann = self.selected[idx]
        img_path = os.path.join(self.image_dir, self.images[ann["image_id"]])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        caption = ann["caption"]
        return {"pixel_values": image, "caption": caption}

# ============================================================
# Training Loop
# ============================================================
def main():
    # --- تغییر: فعال‌سازی mixed precision ---
    accelerator = Accelerator(mixed_precision='fp16') 
    device = accelerator.device

    # Load tokenizer and models
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    
    # --- توجه: pipeline را فقط برای نگه داشتن کامپوننت‌ها لود می‌کنیم ---
    pipeline = StableDiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    
    unet = pipeline.unet.to(device).train()
    text_encoder = pipeline.text_encoder.to(device).train()
    vae = pipeline.vae.to(device).eval()
    vae.requires_grad_(False) # freeze VAE

    # --- تغییر: استفاده از نرخ یادگیری متفاوت ---
    optimizer = torch.optim.AdamW(
        [
            {"params": unet.parameters(), "lr": LEARNING_RATE},
            {"params": text_encoder.parameters(), "lr": TEXT_ENCODER_LR},
        ],
    )

    # --- تغییر: ساخت دیتاست و لودر *خارج* از حلقه اپوک ---
    dataset = COCODataset(TRAIN_IMAGES, ANNOTATION_FILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # --- تغییر: accelerator.prepare *خارج* از حلقه اپوک ---
    # VAE نیازی به prepare ندارد چون eval است و freeze شده
    unet, text_encoder, optimizer, dataloader = accelerator.prepare(
        unet, text_encoder, optimizer, dataloader
    )

    accelerator.print("=== Training Started ===")
    accelerator.print(f"Total Epochs: {EPOCHS}")
    accelerator.print(f"Batch Size (per device): {BATCH_SIZE}")
    accelerator.print(f"Grad Accumulation Steps: {GRAD_ACCUM_STEPS}")
    accelerator.print(f"Effective Batch Size: {BATCH_SIZE * GRAD_ACCUM_STEPS * accelerator.num_processes}")
    accelerator.print(f"Total Batches per Epoch: {len(dataloader)}")
    accelerator.print(f"Total Optimization Steps per Epoch: {len(dataloader) // GRAD_ACCUM_STEPS}")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(
            dataloader, 
            desc=f"Epoch {epoch+1}/{EPOCHS}", 
            disable=not accelerator.is_main_process
        )

        optimizer.zero_grad()
        for step, batch in enumerate(progress_bar):
            # Tokenize captions
            input_ids = tokenizer(
                batch["caption"],
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt"
            ).input_ids.to(device)

            pixel_values = batch["pixel_values"]
            # pixel_values.to(device) # accelerator.prepare(dataloader) خودش این کار را انجام می‌دهد

            # Encode text
            encoder_hidden_states = text_encoder(input_ids)[0]

            # Encode images -> latents using frozen VAE
            with torch.no_grad():
                # VAE را به device منتقل می‌کنیم چون prepare نشده
                latents = vae.encode(pixel_values.to(device)).latent_dist.sample() * 0.18215

            # Add noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Predict noise
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            loss = loss / GRAD_ACCUM_STEPS # scale loss for accumulation

            accelerator.backward(loss)

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                # --- نکته: گرادیان کلیپینگ (اختیاری اما پیشنهادی) ---
                # if accelerator.sync_gradients:
                #     accelerator.clip_grad_norm_(
                #         list(unet.parameters()) + list(text_encoder.parameters()), 1.0
                #     )
                
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * GRAD_ACCUM_STEPS # بازگرداندن به loss اسکیل نشده برای لاگ
            progress_bar.set_postfix({"loss": f"{(total_loss / (step + 1)):.4f}"})

        accelerator.print(f"Epoch {epoch+1} done. Mean loss: {total_loss/len(dataloader):.4f}")

        # --- تغییر: رفع باگ حیاتی در ذخیره‌سازی ---
        if accelerator.is_main_process:
            accelerator.print("Unwrapping and saving models...")
            
            # مدل‌های تمرین دیده را از accelerator پس می‌گیریم
            unwrapped_unet = accelerator.unwrap_model(unet)
            unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
            
            # آنها را در pipeline کپی می‌کنیم
            pipeline.unet = unwrapped_unet
            pipeline.text_encoder = unwrapped_text_encoder
            
            # اکنون pipeline شامل وزن‌های به‌روز شده است
            save_path = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}")
            pipeline.save_pretrained(save_path)
            accelerator.print(f"Saved checkpoint to {save_path}")

    accelerator.print("Training finished successfully!")


if __name__ == "__main__":
    main()