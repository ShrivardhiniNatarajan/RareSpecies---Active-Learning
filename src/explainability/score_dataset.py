import csv
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import traceback

from src.explainability.daac import (
    load_model,
    load_detector_results,
    analyze_detection
)

def create_overlay(image_path, cam, bbox, output_path):
    image = cv2.imread(str(image_path))
    if image is None:
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    cam_normalized = cv2.normalize(cam, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cam_resized = cv2.resize(cam_normalized, (image.shape[1], image.shape[0]))
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 3)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path)


def main():
    print("Loading model...")
    model = load_model()
    
    print("Loading detector results...")
    detector_results = load_detector_results()
    
    output_csv = Path("results/explainability/daac_scores.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    processed_records = set()
    file_exists = output_csv.exists()
    
    if file_exists:
        with open(output_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                record_id = f"{row['image_name']}_{row['detection_index']}"
                processed_records.add(record_id)
                
        print(f"Found {len(processed_records)} already processed detections.")
    
    f_out = open(output_csv, 'a', newline='', encoding='utf-8')
    fieldnames = [
        "image_name",
        "crop_path",
        "detection_index",
        "detection_confidence",
        "predicted_species",
        "classifier_confidence",
        "daac_score"
    ]
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    
    if not file_exists:
        writer.writeheader()
        
    print(f"Total images to process: {len(detector_results)}")
    
    # Flatten detections into tasks
    tasks = []
    for record in detector_results:
        file_name = record.get("file_name")
        image_path = Path("data/raw") / file_name
        if not image_path.exists():
            continue
        for idx, detection in enumerate(record.get("detections", [])):
            record_id = f"{file_name}_{idx}"
            if record_id not in processed_records and detection.get("bbox"):
                tasks.append((file_name, image_path, idx, detection))

    import threading
    write_lock = threading.Lock()

    def process_task(task):
        file_name, image_path, idx, detection = task
        try:
            result, cam = analyze_detection(
                model=model,
                cam=None,
                image_path=image_path,
                bbox=detection["bbox"]
            )
            row = {
                "image_name": file_name,
                "crop_path": detection.get("crop_path", ""),
                "detection_index": idx,
                "detection_confidence": detection.get("confidence", 0.0),
                "predicted_species": result["predicted_species"],
                "classifier_confidence": result["classifier_confidence"],
                "daac_score": result["daac_score"]
            }
            with write_lock:
                writer.writerow(row)
                f_out.flush()
        except Exception as e:
            print(f"Error processing {file_name}_{idx}: {str(e)}")

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        list(tqdm(executor.map(process_task, tasks), total=len(tasks), desc="Processing detections (Parallel)"))

    f_out.close()
    print("Finished DAAC scoring.")


if __name__ == "__main__":
    main()
