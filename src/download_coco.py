import os
import requests
from tqdm import tqdm

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    chunk_size = 1024 * 1024  # 1MB
    with open(save_path, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=os.path.basename(save_path)) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    print(f"Downloaded: {save_path}")

base_url = "http://images.cocodataset.org/"
files = {
    "train2017.zip": "zips/train2017.zip",
    "val2017.zip": "zips/val2017.zip",
    "annotations_trainval2017.zip": "annotations/annotations_trainval2017.zip",
}

save_dir = "./coco_data"
os.makedirs(save_dir, exist_ok=True)

for filename, path in files.items():
    url = base_url + path
    save_path = os.path.join(save_dir, filename)
    download_file(url, save_path)