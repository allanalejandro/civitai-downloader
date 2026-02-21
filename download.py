#!/usr/bin/env python3
import os
import sys
import argparse
import time
import zipfile
from pathlib import Path
import requests

CHUNK_SIZE = 1024 * 1024  # 1MB chunks
TOKEN_FILE = Path.home() / ".civitai" / "config"
CIVITAI_BASE_URL = "https://civitai.com/api/download/models"
USER_AGENT = "civitai-downloader/1.0"


def get_args():
    parser = argparse.ArgumentParser(description="CivitAI Model Downloader")
    parser.add_argument(
        "model_ids",
        type=str,
        help="Comma-separated model IDs (e.g. 12345,67890)",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Directory to download models into",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Do not extract ZIP files after download",
    )
    return parser.parse_args()


def get_token():
    # 1️⃣ Environment variable
    token = os.getenv("CIVITAI_TOKEN")
    if token:
        return token

    # 2️⃣ Stored config file
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text().strip()

    return None


def store_token(token: str):
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(token)


def prompt_for_token():
    token = input("Please enter your CivitAI API token: ").strip()
    store_token(token)
    return token


def download_file(model_id: str, output_path: str, token: str, skip_extract=False):
    url = f"{CIVITAI_BASE_URL}/{model_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": USER_AGENT,
    }

    print(f"\n=== Downloading model ID: {model_id} ===")

    try:
        with requests.get(url, headers=headers, stream=True, allow_redirects=True) as r:
            r.raise_for_status()

            # Determine filename
            filename = None
            content_disposition = r.headers.get("Content-Disposition")
            if content_disposition and "filename=" in content_disposition:
                filename = content_disposition.split("filename=")[1].strip('"')

            if not filename:
                filename = r.url.split("/")[-1].split("?")[0]

            output_file = os.path.join(output_path, filename)

            total_size = int(r.headers.get("Content-Length", 0))
            downloaded = 0
            start_time = time.time()

            with open(output_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size:
                            progress = downloaded / total_size
                            percent = progress * 100
                            elapsed = time.time() - start_time
                            speed = (downloaded / (1024 ** 2)) / elapsed if elapsed > 0 else 0
                            sys.stdout.write(
                                f"\rDownloading: {filename} [{percent:.2f}%] - {speed:.2f} MB/s"
                            )
                            sys.stdout.flush()

            sys.stdout.write("\n")
            print(f"Download completed: {filename}")

            # Simple ZIP extraction (same behavior as old script)
            if output_file.endswith(".zip") and not skip_extract:
                print("Extracting ZIP archive...")
                try:
                    with zipfile.ZipFile(output_file, "r") as zip_ref:
                        zip_ref.extractall(os.path.dirname(output_file))
                    print("Extraction completed.")
                except Exception as e:
                    print(f"ERROR extracting ZIP: {e}")

    except requests.HTTPError as e:
        print(f"HTTP error for model {model_id}: {e}")
    except Exception as e:
        print(f"Unexpected error for model {model_id}: {e}")


def main():
    args = get_args()

    token = get_token()
    if not token:
        token = prompt_for_token()

    os.makedirs(args.output_path, exist_ok=True)

    model_ids = [mid.strip() for mid in args.model_ids.split(",") if mid.strip()]
    for model_id in model_ids:
        download_file(model_id, args.output_path, token, skip_extract=args.no_extract)


if __name__ == "__main__":
    main()
