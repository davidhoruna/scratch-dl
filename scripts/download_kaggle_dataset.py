import os
import subprocess
from pathlib import Path
import argparse
from definitons import ROOT_DIR

KAGGLE_DATASET = "lantian773030/pokemonclassification"

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
    DATA_DIR = Path(ROOT_DIR) / "data"
    download_kaggle_dataset(KAGGLE_DATASET, str(DATA_DIR))
