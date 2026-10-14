#!/usr/bin/env python3
"""
Helper script to download a Kaggle dataset using the Kaggle API.

Usage:
  1. Install the Kaggle package: pip install kaggle
  2. Place your Kaggle API token (kaggle.json) in ~/.kaggle/kaggle.json OR set
     environment variables KAGGLE_USERNAME and KAGGLE_KEY.
  3. Run this script, for example:
       python scripts/download_kaggle.py \
         --dataset abhi8923shriv/liver-disease-patient-dataset \
         --output data/liver

The script will download and unzip the dataset files into the output directory.
"""

from __future__ import annotations

import argparse
import os
import sys

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except Exception:  # ImportError or others
    print("The 'kaggle' package is required. Install it with: pip install kaggle")
    sys.exit(1)


def download_dataset(dataset: str, output_dir: str) -> None:
    api = KaggleApi()
    api.authenticate()

    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading dataset '{dataset}' into '{output_dir}'...")
    # dataset_download_files will download a zip by default; use unzip=True to extract
    api.dataset_download_files(dataset, path=output_dir, unzip=True, quiet=False)
    print("Download complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Kaggle dataset using the Kaggle API")
    parser.add_argument(
        "--dataset",
        required=False,
        default="abhi8923shriv/liver-disease-patient-dataset",
        help="Kaggle dataset slug in the form owner/dataset-name",
    )
    parser.add_argument(
        "--output",
        required=False,
        default="data",
        help="Output directory to save/unzip dataset files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        download_dataset(args.dataset, args.output)
    except Exception as exc:
        print(f"Error downloading dataset: {exc}")
        sys.exit(2)


if __name__ == "__main__":
    main()
