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