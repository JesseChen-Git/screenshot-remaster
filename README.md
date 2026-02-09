
# Korea Gallery: Local AI Restoration Pipeline

This repository contains the local AI pipeline for restoring and enhancing Instagram screenshots.
The pipeline is consolidated into 4 core scripts located in `pipeline/`.

## 📂 Project Structure

```
├── pipeline/
│   ├── run_yolo_detection.py      # Step 1: Detect artifacts (counters, UI elements)
│   ├── run_sam_lama_local.py      # Step 2: Clean artifacts using SAM masks + LaMa inpainting
│   ├── run_flux_restoration.py    # Step 3: Neural Enhancement (RealESRGAN + Flux + GFPGAN)
│   └── assemble_final.py          # Step 4: Final assembly (Frequency Separation / Blending)
├── models/                        # Checkpoints (YOLO, SAM, LaMa, Flux, RealESRGAN, GFPGAN)
├── input/                         # Place raw images here
├── output/                        # Final results
└── gallery/                       # Local Web Gallery (Flask)
```

## 🚀 Usage

### 0. Setup
Ensure you have the model weights in `models/`.
Install dependencies:
```bash
pip install torch torchvision numpy opencv-python ultralytics segment-anything diffusers transformers accelerate
# Clone/Install BasicSR, Real-ESRGAN, GFPGAN in project root if not present
```

### 1. Detection
Run YOLO to identify UI elements, counters, and watermarks.
Output: JSON detection files in `data/detections/`.
```bash
python pipeline/run_yolo_detection.py --input_dir input/ --output_dir data/detections
```

### 2. Cleaning
Use SAM (Segment Anything) to create masks from detections, and LaMa to erase them.
Output: Cleaned images in `data/sam_lama_cleaned_twice/`.
```bash
python pipeline/run_sam_lama_local.py --detections_dir data/detections --image_dir input/ --output_dir data/sam_lama_cleaned_twice
```

### 3. Enhancement
Run the heavy enhancement stack:
- **Flux Fill**: Inpaint detailed areas (if configured with crops)
- **Real-ESRGAN**: 2x Super-resolution
- **GFPGAN**: Face restoration
Output: Enhanced images in `data/flux_batch_data/output/`.
```bash
python pipeline/run_flux_restoration.py
```

### 4. Final Assembly
Combine the cleaned structural base with the enhanced details using Frequency Separation.
Output: Final WebP images in `output/`.
```bash
python pipeline/assemble_final.py --cleaned_dir data/sam_lama_cleaned_twice --flux_dir data/flux_batch_data/output --output_dir output
```

## 🖼️ Local Gallery
View the results using the included web gallery:
```bash
python gallery/main.py
```
Access at `http://localhost:8081`.

## 📜 Credits
- **YOLOv8** (Ultralytics)
- **SAM** (Meta AI)
- **LaMa** (Samsung)
- **Flux.1-Fill** (Black Forest Labs)
- **Real-ESRGAN** & **GFPGAN** (TencentARC)
