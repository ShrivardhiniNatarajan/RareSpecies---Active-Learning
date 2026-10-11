"""Dataset for context margin study."""
import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import numpy as np

class DynamicContextDataset(Dataset):
    def __init__(self, csv_path, images_dir, detections_json, class_names, margin=0.0, dual_stream=False, transform=None):
        self.df = pd.read_csv(csv_path)
        self.images_dir = Path(images_dir)
        self.margin = margin
        self.dual_stream = dual_stream
        self.transform = transform
        self.class_names = class_names
        
        # Load detections
        with open(detections_json, 'r') as f:
            data = json.load(f)
            
        self.crop_to_bbox = {}
        self.file_to_bboxes = {}
        
        if isinstance(data, list):
            # Format: [{"file_name": "...", "detections": [{"bbox": [x1, y1, x2, y2], "crop_path": "..."}]}]
            for item in data:
                fname = item.get("file_name", "")
                dets = item.get("detections", [])
                self.file_to_bboxes[fname] = [d.get("bbox", []) for d in dets]
                for d in dets:
                    cp = d.get("crop_path", "").replace("\\", "/")
                    if cp:
                        self.crop_to_bbox[cp] = (fname, d.get("bbox", []))
                        # Also key by basename
                        self.crop_to_bbox[Path(cp).name] = (fname, d.get("bbox", []))
        elif isinstance(data, dict):
            # Standard MegaDetector format: {"images": [{"file": "...", "detections": [...]}]}
            for img in data.get("images", []):
                fname = img.get("file", "")
                dets = img.get("detections", [])
                self.file_to_bboxes[fname] = [d.get("bbox", []) for d in dets]

    def __len__(self):
        return len(self.df)

    def _get_crop_pixels(self, img_pil, bbox, margin):
        width, height = img_pil.size
        # Detect if bbox is normalized [x, y, w, h] or pixel coordinates [x1, y1, x2, y2]
        if all(0.0 <= v <= 1.0 for v in bbox) and len(bbox) == 4:
            x_norm, y_norm, w_norm, h_norm = bbox
            x1 = x_norm * width
            y1 = y_norm * height
            x2 = (x_norm + w_norm) * width
            y2 = (y_norm + h_norm) * height
        else:
            x1, y1, x2, y2 = bbox
            
        box_w = x2 - x1
        box_h = y2 - y1
        
        pad_w = box_w * margin
        pad_h = box_h * margin
        
        new_x1 = max(0, int(round(x1 - pad_w)))
        new_y1 = max(0, int(round(y1 - pad_h)))
        new_x2 = min(width, int(round(x2 + pad_w)))
        new_y2 = min(height, int(round(y2 + pad_h)))
        
        return img_pil.crop((new_x1, new_y1, new_x2, new_y2))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        crop_path_str = str(row['crop_file']).replace("\\", "/")
        crop_file_path = Path(crop_path_str)
        species = row['species']
        label = self.class_names.index(species)

        # Find crop file on disk if it exists
        resolved_crop = None
        for cand in [
            crop_file_path,
            Path.cwd() / crop_path_str,
            Path("data/cropped") / species / crop_file_path.name,
            Path("data/cropped") / crop_file_path.name,
            Path("/content/RareSpecies---Active-Learning") / crop_path_str,
            Path("/content/RareSpecies---Active-Learning/data/cropped") / species / crop_file_path.name,
        ]:
            if cand.exists():
                resolved_crop = cand
                break

        # FAST PATH for Phase 0 (margin == 0.0 and single-stream)
        # If crop already exists on disk, load directly to match baseline recipe perfectly
        if self.margin == 0.0 and not self.dual_stream and resolved_crop is not None:
            crop_pil = Image.open(resolved_crop).convert('RGB')
            if self.transform:
                crop_pil = self.transform(crop_pil)
            return crop_pil, label, row.get('seq_id', '')

        # DYNAMIC PATH: Re-crop from original source images using MegaDetector bounding boxes
        # Resolve bounding box & original image filename
        fname = ""
        bbox = None
        if crop_path_str in self.crop_to_bbox:
            fname, bbox = self.crop_to_bbox[crop_path_str]
        elif crop_file_path.name in self.crop_to_bbox:
            fname, bbox = self.crop_to_bbox[crop_file_path.name]
        else:
            # Fallback deduction from filename
            stem = crop_file_path.stem
            # e.g., 5858bf4a-23d2-11e8-a6a3-ec086b02610b_crop0 -> 5858bf4a-23d2-11e8-a6a3-ec086b02610b.jpg
            base_id = stem.split("_crop")[0] + ".jpg"
            fname = base_id
            if base_id in self.file_to_bboxes and len(self.file_to_bboxes[base_id]) > 0:
                bbox = self.file_to_bboxes[base_id][0]

        # Locate raw image
        raw_img_path = self.images_dir / fname
        if not raw_img_path.exists():
            raw_img_path = self.images_dir / species / fname
        if not raw_img_path.exists():
            # If raw image is still missing but crop_file exists, use crop_file as fallback
            if resolved_crop is not None:
                crop_pil = Image.open(resolved_crop).convert('RGB')
                if self.transform:
                    crop_pil = self.transform(crop_pil)
                if self.dual_stream:
                    return crop_pil, crop_pil, label
                return crop_pil, label, row.get('seq_id', '')
            raise FileNotFoundError(f"Original image not found: {raw_img_path} (checked {fname})")

        img_pil = Image.open(raw_img_path).convert('RGB')

        if not bbox or len(bbox) != 4:
            bbox = [0, 0, img_pil.width, img_pil.height]

        if self.dual_stream:
            crop_tight = self._get_crop_pixels(img_pil, bbox, margin=0.0)
            crop_context = self._get_crop_pixels(img_pil, bbox, margin=self.margin)
            if self.transform:
                crop_tight = self.transform(crop_tight)
                crop_context = self.transform(crop_context)
            return crop_tight, crop_context, label
        else:
            crop = self._get_crop_pixels(img_pil, bbox, margin=self.margin)
            if self.transform:
                crop = self.transform(crop)
            return crop, label, row.get('seq_id', '')
