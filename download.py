#!/usr/bin/env python3
import os
import sys
import argparse
import time
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote


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
    headers = {
        'Authorization': f'Bearer {token}',
        'User-Agent': USER_AGENT,
    }

    url = f'{CIVITAI_BASE_URL}/{model_id}'
    request = urllib.request.Request(url, headers=headers)

    # FIRST REQUEST (authenticated)
    response = urllib.request.urlopen(request)

    # Expect redirect
    if response.status not in (301, 302, 303, 307, 308):
        raise Exception(f'Unexpected response {response.status}')

    redirect_url = response.getheader('Location')
    if not redirect_url:
        raise Exception('No redirect URL found')

    # SECOND REQUEST (NO AUTH HEADER)
    download_request = urllib.request.Request(
        redirect_url,
        headers={'User-Agent': USER_AGENT}
    )

    response = urllib.request.urlopen(download_request)

    if response.status != 200:
        raise Exception(f'HTTP Error {response.status}')

    content_disposition = response.getheader('Content-Disposition')
    filename = None

    if content_disposition and 'filename=' in content_disposition:
        filename = content_disposition.split('filename=')[1].strip('"')

    if not filename:
        filename = redirect_url.split('/')[-1].split('?')[0]

    output_file = os.path.join(output_path, filename)

    total_size = response.getheader('Content-Length')
    total_size = int(total_size) if total_size else None

    downloaded = 0
    start_time = time.time()

    with open(output_file, 'wb') as f:
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break

            f.write(chunk)
            downloaded += len(chunk)

            if total_size:
                progress = downloaded / total_size
                elapsed = time.time() - start_time
                speed = (downloaded / (1024 ** 2)) / elapsed if elapsed > 0 else 0

                sys.stdout.write(
                    f'\rDownloading: {filename} '
                    f'[{progress*100:.2f}%] - {speed:.2f} MB/s'
                )
                sys.stdout.flush()

    sys.stdout.write('\n')

    print(f'Download completed: {filename}')



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
