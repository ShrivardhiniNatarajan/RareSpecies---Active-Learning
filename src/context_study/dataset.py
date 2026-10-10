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
            self.img_to_detections = {img['file']: img.get('detections', []) for img in data.get('images', [])}
            
    def __len__(self):
        return len(self.df)

    def _get_crop(self, img_pil, bbox, margin):
        width, height = img_pil.size
        # MegaDetector format: [x_min, y_min, width_of_box, height_of_box] (normalized)
        x_norm, y_norm, w_norm, h_norm = bbox
        
        # Convert to absolute pixels
        x_min = x_norm * width
        y_min = y_norm * height
        box_w = w_norm * width
        box_h = h_norm * height
        
        # Expand by margin
        pad_w = box_w * margin
        pad_h = box_h * margin
        
        new_x_min = max(0, x_min - pad_w)
        new_y_min = max(0, y_min - pad_h)
        new_x_max = min(width, x_min + box_w + pad_w)
        new_y_max = min(height, y_min + box_h + pad_h)
        
        crop = img_pil.crop((new_x_min, new_y_min, new_x_max, new_y_max))
        return crop

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # the CSV format might have image_file and bbox_index, or crop_path.
        # Assuming crop_file format: "cropped/coyote/imageid_0.jpg", we infer imageid.
        # This will need tuning based on exact CSV format on Colab.
        
        crop_name = Path(row['crop_file']).stem # e.g. imageid_0
        img_id = "_".join(crop_name.split('_')[:-1]) + ".jpg" # fallback guess, user will supply exact logic
        bbox_idx = int(crop_name.split('_')[-1])
        
        img_path = self.images_dir / row['species'] / img_id
        if not img_path.exists():
            img_path = self.images_dir / img_id
            
        img_pil = Image.open(img_path).convert('RGB')
        
        # Get bbox
        detections = self.img_to_detections.get(img_path.name, [])
        if len(detections) > bbox_idx:
            bbox = detections[bbox_idx]['bbox']
        else:
            # Fallback if mismatch
            bbox = [0.0, 0.0, 1.0, 1.0]
            
        label = self.class_names.index(row['species'])
        
        if self.dual_stream:
            crop_tight = self._get_crop(img_pil, bbox, 0.0)
            crop_context = self._get_crop(img_pil, bbox, self.margin)
            
            if self.transform:
                crop_tight = self.transform(crop_tight)
                crop_context = self.transform(crop_context)
                
            return crop_tight, crop_context, label
            
        else:
            crop = self._get_crop(img_pil, bbox, self.margin)
            if self.transform:
                crop = self.transform(crop)
            return crop, label, row.get('seq_id', '')
