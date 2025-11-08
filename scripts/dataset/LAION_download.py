#!/usr/bin/env python3
"""
Script to extract a clean, COCO-like subset from LAION (laion/laion2B-en).
- Streams laion dataset from Hugging Face.
- Filters "noisy" captions using heuristics (does NOT modify captions).
- Saves train captions (only metadata, no image download).
- Downloads images for val/test (200 each) and saves captions+file names in JSON.
- Outputs JSON files similar to COCO caption lists for train/val/test.

Usage:
    python laion_extract_clean.py

Requirements:
    pip install datasets tqdm requests pillow

Author: Generated for user (comments in English as requested)
"""

import os
import re
import json
import time
import hashlib
from pathlib import Path
from typing import Iterator, Dict, Any

import requests
from PIL import Image
from datasets import load_dataset
from tqdm.auto import tqdm

# -------------------------
# User-configurable params
# -------------------------
HF_DATASET = "laion/relaion1B-nolang-aesthetic"  # Hugging Face streaming dataset
OUTPUT_ROOT = "/media/external20/amirhossein_tighkhorshid/diffusion_distillation/ddpo-pytorch-main/ddpo-pytorch-main/laion_dataset"
TRAIN_SIZE = 10  # target number of train captions (no images downloaded for train)
VAL_SIZE = 1000
TEST_SIZE = 10
MAX_TRIES_PER_IMAGE = 3
IMAGE_DOWNLOAD_TIMEOUT = 10  # seconds
MIN_WORDS_IN_CAPTION = 3  # keep captions with at least this many words
# -------------------------

# create dirs
os.makedirs(OUTPUT_ROOT, exist_ok=True)
images_dir = Path(OUTPUT_ROOT) / "images"
train_dir = images_dir / "train"   # we won't download train images, but create folder for compatibility
val_dir = images_dir / "val"
test_dir = images_dir / "test"
for d in (train_dir, val_dir, test_dir):
    d.mkdir(parents=True, exist_ok=True)

# regex patterns to detect noisy / deprecated captions
FILENAME_PATTERN = re.compile(r"\bIMG[_-]?\d{2,6}\b|\bDSC[_-]?\d{2,6}\b|\b\d{3,6}\.jpg\b", re.IGNORECASE)
URL_PATTERN = re.compile(r"https?://|www\.\w+", re.IGNORECASE)
SOCIAL_PATTERN = re.compile(r"\b(facebook|instagram|twitter|pinterest|tumblr|flickr|unsplash|getty|shutterstock|reddit)\b", re.IGNORECASE)
META_PATTERN = re.compile(r"\b(photo by|uploaded by|copyright|©|©)\b", re.IGNORECASE)
FILEEXT_PATTERN = re.compile(r"\.(jpg|jpeg|png|gif|bmp|webp)\b", re.IGNORECASE)
SHORT_NONALPHA = re.compile(r"^[\W_]+$")  # non-alphabetic garbage
PIPE_SLASH_PATTERN = re.compile(r"[\|\n\r]{1,}|\/\/")  # separators indicating multi-part noisy captions
# Matches nearly all emoji ranges (Unicode 13+)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002700-\U000027BF"  # dingbats
    "\U0001F900-\U0001F9FF"  # supplemental symbols & pictographs
    "\U00002600-\U000026FF"  # miscellaneous symbols
    "\U0001FA70-\U0001FAFF"  # extended symbols
    "\U00002500-\U00002BEF"  # box drawing and misc
    "]+", flags=re.UNICODE
)

# additional heuristics: exclude captions which look like filenames or very noisy tokens
BAD_TOKENS = {"img", "image", "file", "jpg", "jpeg", "png", "raw", "upload", "photo", "photos", "thumbnail"}


def is_caption_clean(caption: str) -> bool:
    """
    Heuristic filter: return True if caption is likely clean/usable.
    Similar to the original version, just with a few extra sanity filters.
    Captions are never modified.
    """
    if not isinstance(caption, str):
        return False
    txt = caption.strip()
    if len(txt) < 3:
        return False

    # obvious noisy patterns
    if FILENAME_PATTERN.search(txt):
        return False
    if URL_PATTERN.search(txt):
        return False
    if SOCIAL_PATTERN.search(txt):
        return False
    if META_PATTERN.search(txt):
        return False
    if FILEEXT_PATTERN.search(txt):
        return False
    if PIPE_SLASH_PATTERN.search(txt):
        return False
    if SHORT_NONALPHA.match(txt):
        return False
    if EMOJI_PATTERN.search(txt):
        return False

    # reject hashtags, emojis, or prompt artifacts
    if re.search(r"[\#\@\*\~\^\<\>\=\[\]\{\}\(\)🙂😀😅😂😭😍❤️⭐🌟🔥✨💀]", txt):
        return False
    if re.search(r"--[a-z]+|\(\(|\)\)|masterpiece|best quality|nsfw", txt.lower()):
        return False

    # minimal word count
    words = re.findall(r"\w+", txt)
    if len(words) < MIN_WORDS_IN_CAPTION:
        return False

    # avoid uppercase spam
    if len(txt) > 8 and txt.isupper():
        return False

    # avoid short non-descriptive captions like "a dog", "car", "photo"
    lower = txt.lower()
    toks = re.findall(r"[a-zA-Z]+", lower)
    if any(t in BAD_TOKENS for t in toks) and len(toks) <= 3:
        return False

    # prefer descriptive captions (keep 2/3 randomly if no adjectives found)
    if not any(x in lower for x in ["beautiful", "realistic", "detailed", "portrait", "lighting", "color", "painting", "illustration", "render", "scene", "photo of", "close-up"]):
        if hash(txt) % 3 == 0:  # keep some variety
            return False

    return True


def safe_download_image(url: str, dest_path: Path, max_tries: int = 3, timeout: int = 8) -> bool:
    """
    Download an image with simple retry and validation (open via PIL).
    Returns True on success, False otherwise.
    """
    headers = {"User-Agent": "laion-extract/1.0 (+https://example.com)"}
    attempt = 0
    while attempt < max_tries:
        attempt += 1
        try:
            resp = requests.get(url, timeout=timeout, headers=headers, stream=True)
            if resp.status_code != 200:
                # bad status
                time.sleep(0.5)
                continue
            # stream to disk
            tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            # try to open with PIL for basic validation
            try:
                with Image.open(tmp_path) as im:
                    im.verify()  # verify does not load full image but checks file integrity
                tmp_path.rename(dest_path)
                return True
            except Exception:
                # corrupted image file
                tmp_path.unlink(missing_ok=True)
                time.sleep(0.2)
                continue
        except Exception:
            time.sleep(0.2)
            continue
    return False

def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def stream_and_collect():
    """
    Main logic:
    - Stream thru laion dataset.
    - Keep clean captions and store their metadata.
    - For val/test, download images.
    """
    from huggingface_hub import login
    login("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")  # your HF read token here
    ds = load_dataset(HF_DATASET, split="train", streaming=True)
    train_list = []
    val_list = []
    test_list = []

    # We'll collect candidate records and fill splits progressively.
    train_needed = TRAIN_SIZE
    val_needed = VAL_SIZE
    test_needed = TEST_SIZE

    total_found = 0
    pbar = tqdm(ds, desc="Streaming laion", unit="rec")
    for rec in pbar:
        # expected fields: 'URL', 'TEXT', 'similarity', 'width', 'height', 'punsafe', 'aesthetic'
        url = rec.get("URL") or rec.get("url") or rec.get("image_url") or rec.get("image")
        caption = rec.get("TEXT") or rec.get("text") or rec.get("caption")
        if url is None or caption is None:
            continue
        # we only want English subset; HF dataset is laion2B-en, so likely English already.
        if not is_caption_clean(caption):
            continue

        # create an id and a file name (for train we won't download image)
        cid = sha1_hex(url)[:12]
        # attempt to find an ext from URL, fallback to .jpg
        ext = ".jpg"
        mext = re.search(r"\.(jpg|jpeg|png|webp|bmp|gif)(?:\?|$)", url, re.IGNORECASE)
        if mext:
            ext = "." + mext.group(1).lower()

        filename = f"{cid}{ext}"
        entry = {
            "id": cid,
            "caption": caption,
            "url": url,
            "file_name": filename,
            "width": rec.get("width"),
            "height": rec.get("height"),
            "aesthetic": rec.get("aesthetic"),
            "similarity": rec.get("similarity"),
        }

        # Fill val/test first (we want exact 200 each and download images)
        if val_needed > 0:
            # try to download image
            dest = val_dir / filename
            ok = safe_download_image(url, dest, max_tries=MAX_TRIES_PER_IMAGE, timeout=IMAGE_DOWNLOAD_TIMEOUT)
            if not ok:
                # skip if image couldn't be downloaded/verified
                continue
            val_list.append(entry)
            val_needed -= 1
            total_found += 1
            pbar.set_postfix({"found": total_found, "train_kept": len(train_list), "val": len(val_list), "test": len(test_list)})
            if val_needed == 0:
                # continue to fill test
                pass
            continue

        if test_needed > 0:
            dest = test_dir / filename
            ok = safe_download_image(url, dest, max_tries=MAX_TRIES_PER_IMAGE, timeout=IMAGE_DOWNLOAD_TIMEOUT)
            if not ok:
                continue
            test_list.append(entry)
            test_needed -= 1
            total_found += 1
            pbar.set_postfix({"found": total_found, "train_kept": len(train_list), "val": len(val_list), "test": len(test_list)})
            continue

        # Otherwise fill train metadata (don't download images)
        if train_needed > 0:
            train_list.append(entry)
            train_needed -= 1
            total_found += 1
            pbar.set_postfix({"found": total_found, "train_kept": len(train_list), "val": len(val_list), "test": len(test_list)})
            if train_needed == 0 and val_needed == 0 and test_needed == 0:
                break

    return train_list, val_list, test_list

def save_json_split(data_list, split_name: str):
    """
    Save JSON with simple structure matching COCO-like captions:
    [
      {"id": <id>, "file_name": <file_name>, "caption": <caption>, "url": <url>, ...},
      ...
    ]
    """
    out_path = Path(OUTPUT_ROOT) / f"captions_{split_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data_list)} entries to {out_path}")

def main():
    print("START: streaming LAION and collecting clean captions")
    train_list, val_list, test_list = stream_and_collect()

    # Save splits
    save_json_split(train_list, "train")
    save_json_split(val_list, "val")
    save_json_split(test_list, "test")

    print("Done.")
    print(f"Train: {len(train_list)}, Val: {len(val_list)}, Test: {len(test_list)}")
    print("Images (val/test) saved under:", images_dir)

if __name__ == "__main__":
    main()
