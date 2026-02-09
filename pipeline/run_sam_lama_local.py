
import os
import sys
import json
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse

# --- CONFIG ---
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
MODELS_DIR = PROJECT_ROOT / "models"
SAM_CHECKPOINT = MODELS_DIR / "weights/sam_vit_b_01ec64.pth"
LAMA_CHECKPOINT = MODELS_DIR / "big-lama.pt"

if not SAM_CHECKPOINT.exists():
    SAM_CHECKPOINT = MODELS_DIR / "weights/sam_vit_h_4b8939.pth"

OUTPUT_DIR = PROJECT_ROOT / "data/sam_lama_cleaned_twice"

# --- SAM/LaMa Helpers ---
def load_lama(model_path):
    print(f"🔹 Loading LaMa from {model_path.name}...")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    return model, device

def run_lama(model, img, mask, device):
    with torch.no_grad():
        img = img.to(device)
        mask = mask.to(device)
        return model(img, mask)

class LocalSAM:
    def __init__(self, model_path, model_type="vit_b"):
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"🔹 Loading SAM ({model_type}) on {device}...")
        try:
            import segment_anything
            from segment_anything import sam_model_registry, SamPredictor
            self.sam = sam_model_registry[model_type](checkpoint=str(model_path))
            self.sam.to(device=device)
            self.predictor = SamPredictor(self.sam)
        except ImportError:
            print("❌ segment_anything not installed.")
            sys.exit(1)

    def set_image(self, image_np):
        self.predictor.set_image(image_np)

    def predict_box(self, box_coords):
        try:
            masks, _, _ = self.predictor.predict(box=np.array(box_coords)[None, :], multimask_output=False)
            return masks[0]
        except: return np.zeros((1, 1), dtype=bool)

def main():
    parser = argparse.ArgumentParser(description="Step 2: SAM + LaMa Cleaning")
    parser.add_argument("--detections_dir", required=True, help="Folder with JSON detections")
    parser.add_argument("--image_dir", required=True, help="Folder with original images")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR), help="Output folder for cleaned images")
    args = parser.parse_args()

    det_dir = Path(args.detections_dir)
    img_dir = Path(args.image_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not det_dir.exists() or not img_dir.exists():
        print("❌ Detections or Image directory missing.")
        return

    # Initialize Models
    if not SAM_CHECKPOINT.exists():
        print(f"❌ SAM Checkpoint missing: {SAM_CHECKPOINT}")
        return
        
    model_type = "vit_b" if "vit_b" in str(SAM_CHECKPOINT) else "vit_h"
    sam = LocalSAM(SAM_CHECKPOINT, model_type=model_type)
    lama, lama_device = load_lama(LAMA_CHECKPOINT)

    json_files = list(det_dir.glob("*.json"))
    print(f"🚀 Processing {len(json_files)} detection files...")

    for json_path in tqdm(json_files):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            filename = data.get("filename")
            detections = data.get("detections", [])
            
            # Find Image
            img_path = img_dir / filename
            if not img_path.exists():
                print(f"⚠️ Image not found: {img_path}")
                continue
                
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None: continue
            
            current_img_bgr = img_bgr.copy()
            if not detections:
                # No artifacts, just copy original
                cv2.imwrite(str(out_dir / filename), current_img_bgr)
                continue

            # Double Pass Cleaning (Aggressive -> Cleanup)
            for pass_idx in range(2):
                h, w = current_img_bgr.shape[:2]
                img_rgb = cv2.cvtColor(current_img_bgr, cv2.COLOR_BGR2RGB)
                
                # Setup SAM
                sam.set_image(img_rgb)
                combined_mask = np.zeros((h, w), dtype=bool)
                
                for det in detections:
                    mask = sam.predict_box(det['box'])
                    # Resize if needed
                    if mask.shape != (h, w):
                         mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                    combined_mask = np.logical_or(combined_mask, mask)
                
                if not np.any(combined_mask): break

                # Prepare LaMa Inputs
                # LaMa needs divisible by 8
                new_h = (h // 8) * 8
                new_w = (w // 8) * 8
                
                img_resized = cv2.resize(current_img_bgr, (new_w, new_h))
                mask_uint8 = (combined_mask * 255).astype(np.uint8)
                mask_resized = cv2.resize(mask_uint8, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                
                # Dilation
                # Pass 1: 15px, Pass 2: 10px
                k = 15 if pass_idx == 0 else 10
                mask_resized = cv2.dilate(mask_resized, np.ones((k, k), np.uint8), iterations=1)
                
                img_t = torch.from_numpy(img_resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
                mask_t = torch.from_numpy((mask_resized > 127).astype(np.float32)).unsqueeze(0).unsqueeze(0)
                
                # Run Inference
                out_t = run_lama(lama, img_t, mask_t, lama_device)
                
                out_np = out_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
                out_np = np.clip(out_np * 255.0, 0, 255).astype(np.uint8)
                
                # Resize back
                current_img_bgr = cv2.resize(out_np, (w, h))

            # Save Final
            cv2.imwrite(str(out_dir / filename), current_img_bgr)
            
        except Exception as e:
            print(f"❌ Error processing {json_path.name}: {e}")

    print(f"✅ Cleaning complete. Results saved to {out_dir}")

if __name__ == "__main__":
    main()
