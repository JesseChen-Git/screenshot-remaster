
import os
import json
import torch
import random
import gc
import sys
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from diffusers import FluxFillPipeline

# --- Dependencies Setup ---
# Add local repos to path (Assumes they are in the project root)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT / "BasicSR"))
sys.path.append(str(PROJECT_ROOT / "Real-ESRGAN"))
sys.path.append(str(PROJECT_ROOT / "GFPGAN"))

# Import Enhancement Libs (Try/Except to handle missing envs smoothly)
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    from gfpgan import GFPGANer
    HAS_ENHANCEMENT = True
except ImportError as e:
    print(f"⚠️ Enhancement libraries missing: {e}")
    # We will try to proceed with Flux only if libs are missing, but warn user.
    # But user specifically asked for "realgen+flux+gffpan", so we should ideally fail or fix path.
    # Assuming paths are correct relative to repo root.
    HAS_ENHANCEMENT = False

# --- Config ---
MODEL_ID = "black-forest-labs/FLUX.1-Fill-dev"
BATCH_DIR = "/root/qwen-inpaint/flux_batch_data/full_batch"
# Override BATCH_DIR to local if not on VPS
if not os.path.exists("/root/qwen-inpaint"):
    BATCH_DIR = str(PROJECT_ROOT / "data/flux_batch_data/full_batch")

METADATA_PATH = os.path.join(BATCH_DIR, "metadata.json")
OUTPUT_DIR = os.path.join(BATCH_DIR, "output")
MAX_DIM = 1024
MODELS_DIR = PROJECT_ROOT / "models"
GFPGAN_MODEL_PATH = MODELS_DIR / "GFPGANv1.4.pth"
ESRGAN_MODEL_PATH = MODELS_DIR / "RealESRGAN_x2plus.pth"

# --- Prompts ---
BASE_CONCEPTS_LIST = [
    "clean background", "seamless removal", "high quality", "photorealistic", "soft lighting",
    "sharp details", "natural texture"
]
QUALITY_BOOSTERS_LIST = [
    "raw photo", "dslr quality", "fujifilm simulation", "intricate details", "pores visible", "sharp focus"
]

def generate_dynamic_prompt():
    random.shuffle(BASE_CONCEPTS_LIST)
    core = ", ".join(BASE_CONCEPTS_LIST[:4])
    boost = ", ".join(random.sample(QUALITY_BOOSTERS_LIST, 2))
    return f"{core}, {boost}, removing watermark"

class Enhancer:
    def __init__(self, device):
        self.device = device
        self.realesrgan = None
        self.gfpgan = None
        if HAS_ENHANCEMENT:
            self.load_models()

    def load_models(self):
        print("🔹 Loading RealESRGAN & GFPGAN...")
        # RealESRGAN
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        self.realesrgan = RealESRGANer(
            scale=2,
            model_path=str(ESRGAN_MODEL_PATH),
            model=model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=True if "cuda" in str(self.device) else False,
            device=self.device
        )
        # GFPGAN
        self.gfpgan = GFPGANer(
            model_path=str(GFPGAN_MODEL_PATH),
            upscale=2,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=self.realesrgan,
            device=self.device
        )

    def process(self, img_pil):
        if not self.gfpgan: 
            return img_pil
        
        # Convert PIL -> CV2
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # Enhance (Face + Background)
        # paste_back=True combines the fixed face with the upscaled background
        _, _, output = self.gfpgan.enhance(
            img_cv, 
            has_aligned=False, 
            only_center_face=False, 
            paste_back=True,
            weight=0.5
        )
        
        # Convert CV2 -> PIL
        return Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

def main():
    print(f"🚀 Starting Flux Restoration Pipeline")
    print(f"   Mode: Flux Inpaint + {'RealESRGAN/GFPGAN' if HAS_ENHANCEMENT else 'None'}")
    
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Flux
    print(f"🔹 Loading Flux: {MODEL_ID}...")
    dtype = torch.bfloat16
    offload_folder = PROJECT_ROOT / "data/flux_offload"
    os.makedirs(offload_folder, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    pipe = FluxFillPipeline.from_pretrained(
        MODEL_ID, 
        torch_dtype=dtype,
        offload_folder=str(offload_folder)
    )
    pipe.enable_sequential_cpu_offload()
    
    # 2. Load Enhancer (RealESRGAN / GFPGAN)
    enhancer = Enhancer(device)
    
    # 3. Process Tasks
    if not os.path.exists(METADATA_PATH):
        print(f"❌ Metadata not found: {METADATA_PATH}")
        return
        
    with open(METADATA_PATH, "r") as f:
        tasks = json.load(f)
        
    print(f"📋 Processing {len(tasks)} tasks...")
    
    for i, task in enumerate(tasks):
        task_id = task["id"]
        output_filename = f"{task_id}_flux.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Resume Check
        if os.path.exists(output_path):
            continue
            
        print(f"[{i+1}/{len(tasks)}] Processing {task_id}...")
        
        try:
            # Load Crop & Mask
            crop_path = os.path.join(BATCH_DIR, task["crop_file"])
            mask_path = os.path.join(BATCH_DIR, task["mask_file"])
            
            image = Image.open(crop_path).convert("RGB")
            mask = Image.open(mask_path).convert("RGB")
            
            w, h = image.size
            align_w = (w // 32) * 32
            align_h = (h // 32) * 32
            image_in = image.resize((align_w, align_h), Image.LANCZOS)
            mask_in = mask.resize((align_w, align_h), Image.NEAREST)
            
            # A. Flux Inpainting
            prompt = generate_dynamic_prompt()
            with torch.inference_mode():
                result = pipe(
                    prompt=prompt,
                    image=image_in,
                    mask_image=mask_in,
                    height=align_h,
                    width=align_w,
                    guidance_scale=30.0,
                    num_inference_steps=50,
                    generator=torch.Generator("cuda").manual_seed(random.randint(0, 10000))
                ).images[0]
            
            # Resize back to original
            if align_w != w or align_h != h:
                result = result.resize((w, h), Image.LANCZOS)
            
            # B. Enhancement (RealESRGAN + GFPGAN)
            if HAS_ENHANCEMENT:
                # print("   ✨ Enhancing...")
                result = enhancer.process(result)
                # Resize back again? RealESRGAN upscales 2x. 
                # If we want the flux crop to match the original hole, we should resize back to WxH?
                # User said "realgen... enhancement". Typically this implies keeping the high res?
                # But if we plug this back into `assemble_final.py`, it expects the size to match OR it resizes.
                # Let's keep the upscaled version (2x) as it contains the "Enhancement". 
                # `assemble_final.py` (which I reverted/updated) handles resizing.
                pass

            result.save(output_path)
            
            if i % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"❌ Error processing {task_id}: {e}")

    print("✅ Batch Processing Complete!")

if __name__ == "__main__":
    main()
