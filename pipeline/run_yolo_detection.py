
import os
import sys
import json
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse
from ultralytics import YOLO

# --- CONFIG ---
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "input"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/detections"

# Model Paths
YOLO_UNIVERSE_PATH = MODELS_DIR / "universe_v1.pt"
YOLO_STRUCTURAL_PATH = MODELS_DIR / "structural_v2.pt"
YOLO_CONTENT_PATH = MODELS_DIR / "content_v2.pt"
YOLO_UI_PATH = MODELS_DIR / "ui_element_v3.pt"

THRESHOLDS = {
    "avatar": 0.50, "page_counter": 0.20, "ui_element": 0.65,
    "watermark": 0.51, "text_label": 0.81, "default": 0.35
}

STRICT_LOCATIONS = {
    "page_counter": ["top_right"],
    "ui_element": ["bottom_left", "bottom_right"]
}

def get_location_desc(ymin, xmin, ymax, xmax):
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    v = "top" if cy < 0.35 else "bottom" if cy > 0.65 else "mid"
    h = "left" if cx < 0.35 else "right" if cx > 0.65 else "center"
    return f"{v}_{h}"

def get_iou_containment(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return max(interArea / float(boxBArea + 1e-6), interArea / float(boxAArea + 1e-6))

class FastPipeline:
    def __init__(self):
        print("🚀 Initializing YOLO Ensemble...")
        self.models = {}
        try:
            if YOLO_UNIVERSE_PATH.exists(): self.models['universe'] = YOLO(YOLO_UNIVERSE_PATH)
            if YOLO_STRUCTURAL_PATH.exists(): self.models['structural'] = YOLO(YOLO_STRUCTURAL_PATH)
            if YOLO_CONTENT_PATH.exists(): self.models['content'] = YOLO(YOLO_CONTENT_PATH)
            if YOLO_UI_PATH.exists(): self.models['ui_v3'] = YOLO(YOLO_UI_PATH)
            print(f"   ✅ Loaded {len(self.models)} YOLO models.")
        except Exception as e:
            print(f"   ❌ Failed to load models: {e}")

    def get_class_priority(self, label, conf):
        if label == 'avatar': return 100
        if label == 'page_counter': return 90
        if label == 'ui_element': return 80
        return 50 + conf

    def detect_artifacts(self, img_rgb, conf_threshold_override=None):
        candidates = []
        h, w = img_rgb.shape[:2]

        def process_model(model_key, source_name):
            if model_key not in self.models: return
            infer_conf = 0.10 if conf_threshold_override is None else min(0.10, conf_threshold_override)
            results = self.models[model_key](img_rgb, verbose=False, iou=0.4, conf=infer_conf)
            
            for r in results:
                for box in r.boxes:
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    label = r.names[int(box.cls[0])]
                    
                    if conf_threshold_override is None:
                        thresh = THRESHOLDS.get(label, THRESHOLDS["default"])
                        if conf < thresh: continue
                        
                        ymin, xmin, ymax, xmax = coords[1]/h, coords[0]/w, coords[3]/h, coords[2]/w
                        loc = get_location_desc(ymin, xmin, ymax, xmax)
                        if label in STRICT_LOCATIONS and loc not in STRICT_LOCATIONS[label]: continue
                    
                    candidates.append({
                        "box": tuple(coords.tolist()), "label": label, "conf": conf,
                        "source": source_name, "priority": self.get_class_priority(label, conf)
                    })

        process_model('structural', 'spec_structural')
        process_model('content', 'spec_content')
        process_model('ui_v3', 'spec_ui_v3')
        
        if 'universe' in self.models:
            results = self.models['universe'](img_rgb, verbose=False, iou=0.4, conf=0.10)
            for r in results:
                for box in r.boxes:
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    label = r.names[int(box.cls[0])]
                    if label in ['page_counter', 'ui_element']: continue
                    if conf < THRESHOLDS.get(label, THRESHOLDS["default"]): continue
                    candidates.append({
                        "box": tuple(coords.tolist()), "label": label, "conf": conf,
                        "source": 'universe', "priority": self.get_class_priority(label, conf)
                    })

        # NMS
        candidates.sort(key=lambda x: x['priority'], reverse=True)
        final_detections = []
        for c in candidates:
            is_dup = False
            for k in final_detections:
                if c['label'] == 'avatar' or k['label'] == 'avatar': continue
                if get_iou_containment(c['box'], k['box']) > 0.5:
                    is_dup = True
                    break
            if not is_dup: final_detections.append(c)
            
        return final_detections

def main():
    parser = argparse.ArgumentParser(description="Step 1: Run YOLO Detection (Outputs JSON)")
    parser.add_argument("--input_dir", default=str(DEFAULT_INPUT_DIR), help="Input images folder")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Output folder for JSON detections")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        print(f"❌ Input dir {in_dir} does not exist.")
        return

    fp = FastPipeline()
    
    images = list(in_dir.glob("*.jpg")) + list(in_dir.glob("*.png")) + list(in_dir.glob("*.webp"))
    print(f"📸 Found {len(images)} images in {in_dir}")

    for img_path in tqdm(images):
        try:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None: continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            detections = fp.detect_artifacts(img_rgb)
            
            # Save Detections
            json_name = f"{img_path.stem}.json"
            out_path = out_dir / json_name
            
            data = {
                "filename": img_path.name,
                "width": img_rgb.shape[1],
                "height": img_rgb.shape[0],
                "detections": detections
            }
            
            with open(out_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"❌ Error processing {img_path.name}: {e}")

    print(f"✅ Detection complete. Results saved to {out_dir}")

if __name__ == "__main__":
    main()
