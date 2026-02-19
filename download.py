#!/usr/bin/env python3
import os
import sys
import argparse
import time
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote
import requests


CHUNK_SIZE = 1638400
TOKEN_FILE = Path.home() / '.civitai' / 'config'
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
DEFAULT_ENV_NAME = os.getenv("CIVITAI_TOKEN_NAME", "CIVITAI_TOKEN")
CIVITAI_BASE_URL = 'https://civitai.com/api/download/models'


def get_args():
    parser = argparse.ArgumentParser(
        description='CivitAI Downloader',
    )

    parser.add_argument(
        'model_id',
        type=str,
        help='CivitAI Model ID or comma-separated list (eg: 46846 or 1535275,1188894,1122976)'        
    )

    parser.add_argument(
        'output_path',
        type=str,
        help='Output path, eg: /workspace/stable-diffusion-webui/models/Stable-diffusion'
    )

    return parser.parse_args()


def get_token():
    token = os.getenv(DEFAULT_ENV_NAME, None)
    if token:
        return token
    try:
        with open(TOKEN_FILE, 'r') as file:
            token = file.read().strip()
            return token
    except Exception as e:
        return None


def store_token(token: str):
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(TOKEN_FILE, 'w') as file:
        file.write(token)


def prompt_for_civitai_token():
    token = input('Please enter your CivitAI API token: ')
    store_token(token)
    return token



def download_file(model_id: str, output_path: str, token: str):
    url = f"{CIVITAI_BASE_URL}/{model_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": USER_AGENT,
    }

    print(f"=== Downloading model ID: {model_id} ===")

    with requests.get(url, headers=headers, stream=True, allow_redirects=True) as r:
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            raise Exception(f"HTTP Error {r.status_code}: {r.reason}") from e

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
                        elapsed = time.time() - start_time
                        speed = (downloaded / (1024 ** 2)) / elapsed if elapsed > 0 else 0
                        sys.stdout.write(
                            f"\rDownloading: {filename} "
                            f"[{progress*100:.2f}%] - {speed:.2f} MB/s"
                        )
                        sys.stdout.flush()

        sys.stdout.write("\n")
        elapsed = time.time() - start_time
        minutes, seconds = divmod(int(elapsed), 60)
        print(f"Download completed: {filename}")
        print(f"Time taken: {minutes}m {seconds}s")

        # Extract ZIP files automatically
        if output_file.endswith(".zip"):
            print("Extracting ZIP archive...")
            try:
                with zipfile.ZipFile(output_file, "r") as zip_ref:
                    zip_ref.extractall(os.path.dirname(output_file))
            except Exception as e:
                print(f"ERROR: Failed to unzip file: {e}")




def main():
    args = get_args()
    token = get_token()

    if not token:
        token = prompt_for_civitai_token()

    # Split model IDs by comma and strip whitespace
    model_ids = [mid.strip() for mid in args.model_id.split(',') if mid.strip()]

    for model_id in model_ids:
        print(f'\n=== Downloading model ID: {model_id} ===')
        try:
            download_file(model_id, args.output_path, token)
        except Exception as e:
            print(f'ERROR downloading model {model_id}: {e}')


if __name__ == '__main__':
    main()
