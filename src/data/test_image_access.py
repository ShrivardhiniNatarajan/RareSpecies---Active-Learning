from pathlib import Path

import pandas as pd
import requests


METADATA_FILE = Path(
    "data/metadata/subset_metadata.csv"
)

BASE_URL = (
    "https://lilawildlife.blob.core.windows.net/"
    "lila-wildlife/caltech-unzipped/cct_images"
)


df = pd.read_csv(
    METADATA_FILE
)

file_name = df.iloc[0]["file_name"]

url = (
    f"{BASE_URL}/{file_name}"
)

print("Testing:")
print(url)


response = requests.get(
    url,
    stream=True,
    timeout=30
)

print(
    "\nHTTP status:",
    response.status_code
)

print(
    "Content type:",
    response.headers.get(
        "Content-Type"
    )
)

print(
    "Content length:",
    response.headers.get(
        "Content-Length"
    )
)


if response.ok:

    output = Path(
        "data/raw/test_image.jpg"
    )

    with open(
        output,
        "wb"
    ) as f:

        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):

            if chunk:
                f.write(chunk)

    print(
        "\nTest image saved to:",
        output
    )

else:

    print(
        "\nDownload failed."
    )