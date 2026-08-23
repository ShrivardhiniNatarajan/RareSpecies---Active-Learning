from pathlib import Path

from PytorchWildlife.models import detection as pw_detection


IMAGE_DIR = Path("data/raw")

images = sorted(IMAGE_DIR.glob("*.jpg"))

if not images:
    raise FileNotFoundError(
        "No JPG files found in data/raw/"
    )

test_image = images[0]

print("=" * 60)
print("MegaDetector single-image test")
print("=" * 60)

print(f"Image: {test_image}")

print("\nLoading MegaDetector V6...")

import torch

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = pw_detection.MegaDetectorV6(
    version="MDV6-yolov10-c",
    device=device,
    pretrained=True
)

print("Model loaded successfully.")

print("\nRunning detection...")

result = model.single_image_detection(
    str(test_image)
)

print("\nDetection result:")
print(result)