import os
import subprocess
from pathlib import Path
import argparse
from definitons import ROOT_DIR

KAGGLE_DATASET = ""

def download_kaggle_dataset(dataset: str, dest_dir: str):
    os.makedirs(dest_dir, exist_ok=True)
    print(f"Downloading {dataset} into {dest_dir}...")
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", dataset,
        "-p", dest_dir,
        "--unzip"
    ], check=True)
    print("Download complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    args=parser.parse_args()
    KAGGLE_DATASET = args.dataset
    DATA_DIR = Path(ROOT_DIR) / "data"
    download_kaggle_dataset(KAGGLE_DATASET, str(f"{DATA_DIR}/{KAGGLE_DATASET.split('/')[1]}"))
