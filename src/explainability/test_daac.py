import cv2
import numpy as np
from PIL import Image
from pathlib import Path

from src.explainability.daac import (
    load_model,
    load_detector_results,
    analyze_detection,
)


def main():

    print("=" * 60)
    print("DAAC SINGLE-IMAGE TEST")
    print("=" * 60)

    model = load_model()

    results = load_detector_results()

    # Take the first image that has a detection
    for record in results:

        detections = record.get(
            "detections",
            []
        )

        if not detections:
            continue

        detection = detections[0]

        image_path = (
            Path("data/raw")
            / record["file_name"]
        )

        bbox = detection["bbox"]

        result, cam = analyze_detection(
            model=model,
            cam=None,
            image_path=image_path,
            bbox=bbox
        )
            # ------------------------------------------------------------
        # Create Grad-CAM visualization
        # ------------------------------------------------------------

        image = cv2.imread(
            str(image_path)
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # Normalize CAM to 0-255
        cam_normalized = cv2.normalize(
            cam,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        # Resize CAM to original image
        cam_resized = cv2.resize(
            cam_normalized,
            (image.shape[1], image.shape[0])
        )

        # Convert to heatmap
        heatmap = cv2.applyColorMap(
            cam_resized,
            cv2.COLORMAP_JET
        )

        heatmap = cv2.cvtColor(
            heatmap,
            cv2.COLOR_BGR2RGB
        )

        # Blend image + heatmap
        overlay = cv2.addWeighted(
            image,
            0.6,
            heatmap,
            0.4,
            0
        )

        # Draw detector bounding box
        x1, y1, x2, y2 = result["bbox"]

        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            3
        )

        output_path = Path(
            "results/explainability/daac_test_overlay.jpg"
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        Image.fromarray(
            overlay
        ).save(
            output_path
        )

        print(
            "Visualization saved:",
            output_path
        )
        print("\nImage:")
        print(record["file_name"])

        print(
            "MegaDetector confidence:",
            detection["confidence"]
        )

        print(
            "Predicted species:",
            result["predicted_species"]
        )

        print(
            "Classifier confidence:",
            f"{result['classifier_confidence']:.4f}"
        )

        print(
            "DAAC score:",
            f"{result['daac_score']:.4f}"
        )

        print(
            "Bounding box:",
            result["bbox"]
        )

        break


if __name__ == "__main__":
    main()