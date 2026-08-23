import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

METADATA_FILE = Path(
    "data/metadata/subset_metadata.csv"
)

OUTPUT_DIR = Path("data/raw")

BASE_URL = (
    "https://lilawildlife.blob.core.windows.net/"
    "lila-wildlife/caltech-unzipped/cct_images"
)

MAX_WORKERS = 16
MAX_RETRIES = 3
TIMEOUT = 30


# ============================================================
# SETUP
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

df = pd.read_csv(METADATA_FILE)

file_names = (
    df["file_name"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


# ============================================================
# DOWNLOAD ONE IMAGE
# ============================================================

def download_one(file_name: str):

    output_path = OUTPUT_DIR / file_name

    # Already downloaded
    if output_path.exists() and output_path.stat().st_size > 0:
        return file_name, "already_exists", None

    url = f"{BASE_URL}/{file_name}"

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            response = requests.get(
                url,
                stream=True,
                timeout=TIMEOUT
            )

            response.raise_for_status()

            temp_path = output_path.with_suffix(
                output_path.suffix + ".part"
            )

            with open(temp_path, "wb") as f:

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):

                    if chunk:
                        f.write(chunk)

            temp_path.replace(output_path)

            return file_name, "downloaded", None

        except Exception as exc:

            partial = output_path.with_suffix(
                output_path.suffix + ".part"
            )

            if partial.exists():
                partial.unlink()

            if attempt < MAX_RETRIES:
                time.sleep(2 ** (attempt - 1))

            else:
                return file_name, "failed", str(exc)

    return file_name, "failed", "unknown error"


# ============================================================
# MAIN
# ============================================================

print("=" * 60)
print("RARECAM IMAGE DOWNLOADER")
print("=" * 60)

print(f"Images in metadata : {len(file_names)}")
print(f"Worker threads     : {MAX_WORKERS}")
print(f"Output directory   : {OUTPUT_DIR}")
print()


downloaded = 0
already_exists = 0
failed = []

# Only unfinished images are sent to workers.
pending = [
    name
    for name in file_names
    if not (
        (OUTPUT_DIR / name).exists()
        and (OUTPUT_DIR / name).stat().st_size > 0
    )
]

print(f"Already downloaded : {len(file_names) - len(pending)}")
print(f"Remaining           : {len(pending)}")
print()


with ThreadPoolExecutor(
    max_workers=MAX_WORKERS
) as executor:

    futures = {
        executor.submit(download_one, name): name
        for name in pending
    }

    with tqdm(
        total=len(pending),
        desc="Downloading",
        unit="img"
    ) as progress:

        for future in as_completed(futures):

            file_name, status, error = future.result()

            if status == "downloaded":
                downloaded += 1

            elif status == "already_exists":
                already_exists += 1

            elif status == "failed":
                failed.append({
                    "file_name": file_name,
                    "error": error
                })

            progress.update(1)


# ============================================================
# SAVE FAILURES
# ============================================================

if failed:

    failed_df = pd.DataFrame(failed)

    failed_path = Path(
        "data/metadata/failed_downloads.csv"
    )

    failed_df.to_csv(
        failed_path,
        index=False
    )


# ============================================================
# SUMMARY
# ============================================================

total_present = sum(
    1
    for name in file_names
    if (
        (OUTPUT_DIR / name).exists()
        and (OUTPUT_DIR / name).stat().st_size > 0
    )
)

print()
print("=" * 60)
print("DOWNLOAD SUMMARY")
print("=" * 60)

print(f"Metadata images     : {len(file_names)}")
print(f"Already present     : {len(file_names) - len(pending)}")
print(f"Downloaded now      : {downloaded}")
print(f"Failed              : {len(failed)}")
print(f"Total present       : {total_present}")
print(f"Remaining           : {len(file_names) - total_present}")