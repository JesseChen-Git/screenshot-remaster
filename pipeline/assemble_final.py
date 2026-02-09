
import os
import sys
import argparse
import glob
import cv2
import numpy as np
from PIL import Image, ImageEnhance

# Params
SIGMA = 3.0 # Blur radius for frequency separation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaned_dir", required=True) # Step 1 Output (Structure)
    parser.add_argument("--enhanced_dir", required=True) # Step 2 Output (Texture/Color)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    cleaned_dir = args.cleaned_dir
    enhanced_dir = args.enhanced_dir
    final_dir = args.output_dir

    os.makedirs(final_dir, exist_ok=True)
    
    # Get Enhanced Files (Reference for successful processing)
    enhanced_files = glob.glob(os.path.join(enhanced_dir, "*"))
    print(f"🔹 Assembling {len(enhanced_files)} images...")
    
    for flux_path in enhanced_files:
        stem = os.path.splitext(os.path.basename(flux_path))[0]
        
        # Find corresponding cleaned structure
        # Step 1 output format: stem.png
        cleaned_path = os.path.join(cleaned_dir, f"{stem}.png")
        if not os.path.exists(cleaned_path):
            cleaned_path = os.path.join(cleaned_dir, f"{stem}.jpg")
        
        if not os.path.exists(cleaned_path):
            print(f"⚠️ Cleaned structure not found for {stem}. Skipping.")
            continue
            
        try:
            flux_pil = Image.open(flux_path).convert("RGB")
            cleaned_pil = Image.open(cleaned_path).convert("RGB")
            
            # Resize Cleaned (Structure) UP to Flux (Texture)
            if flux_pil.size != cleaned_pil.size:
                cleaned_pil = cleaned_pil.resize(flux_pil.size, Image.LANCZOS)
                
            # Convert to LAB
            cleaned_np = np.array(cleaned_pil)
            flux_np = np.array(flux_pil)
            
            cleaned_lab = cv2.cvtColor(cleaned_np, cv2.COLOR_RGB2LAB).astype(np.float32)
            flux_lab = cv2.cvtColor(flux_np, cv2.COLOR_RGB2LAB).astype(np.float32)
            
            # Split
            lc, ac, bc = cv2.split(cleaned_lab)
            lf, af, bf = cv2.split(flux_lab)
            
            # Frequency Separation on L
            # Structure from Cleaned (Low Freq)
            lc_low = cv2.GaussianBlur(lc, (0, 0), SIGMA)
            
            # Texture from Flux (High Freq)
            lf_low = cv2.GaussianBlur(lf, (0, 0), SIGMA)
            lf_high = lf - lf_low
            
            # Combine
            l_new = lc_low + lf_high
            l_new = np.clip(l_new, 0, 255)
            
            # Combine Channels: New L + Flux Color
            merged_lab = cv2.merge([l_new, af, bf])
            merged_lab = np.clip(merged_lab, 0, 255).astype(np.uint8)
            
            # Convert back to RGB
            result_rgb = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
            result_pil = Image.fromarray(result_rgb)
            
            # Adjustments
            enhancer = ImageEnhance.Color(result_pil)
            result_pil = enhancer.enhance(1.03)
            enhancer = ImageEnhance.Contrast(result_pil)
            result_pil = enhancer.enhance(1.03)
            
            # Save
            out_name = f"freq_{stem}.webp"
            result_pil.save(os.path.join(final_dir, out_name), format="WEBP", quality=85)
            print(f"   ✅ Assembled {out_name}")
            
        except Exception as e:
            print(f"❌ Error assembling {stem}: {e}")

if __name__ == "__main__":
    main()
