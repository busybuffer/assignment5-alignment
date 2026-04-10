import requests
import gzip
import shutil
import os
from tqdm import tqdm

# --- Configuration ---
SFT_TRAIN_URL = "https://nlp.stanford.edu/data/nfliu/cs336-spring-2024/assignment5/safety_augmented_ultrachat_200k_single_turn/train.jsonl.gz"
SFT_TEST_URL = "https://nlp.stanford.edu/data/nfliu/cs336-spring-2024/assignment5/safety_augmented_ultrachat_200k_single_turn/test.jsonl.gz"
TARGET_DIR = "data/sft"

TRAIN_GZ_FILENAME = "train.jsonl.gz"
TRAIN_JSONL_FILENAME = "train.jsonl"
TEST_GZ_FILENAME = "test.jsonl.gz"
TEST_JSONL_FILENAME = "test.jsonl"

TRAIN_GZ_FILEPATH = os.path.join(TARGET_DIR, TRAIN_GZ_FILENAME)
TRAIN_JSONL_FILEPATH = os.path.join(TARGET_DIR, TRAIN_JSONL_FILENAME)
TEST_GZ_FILEPATH = os.path.join(TARGET_DIR, TEST_GZ_FILENAME)
TEST_JSONL_FILEPATH = os.path.join(TARGET_DIR, TEST_JSONL_FILENAME)

# --- Main Download and Unzip Logic ---
def download_file(url, destination):
    """Downloads a file from a URL to a destination, showing progress."""
    print(f"Downloading {url} to {destination}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte

        with open(destination, 'wb') as f, tqdm(
            desc=destination,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                bar.update(size)
        print(f"Download complete: {destination}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return False
    except Exception as e:
        print(f"An error occurred during download: {e}")
        return False

def unzip_gz_file(gz_path, output_path):
    """Decompresses a .gz file."""
    print(f"Decompressing {gz_path} to {output_path}...")
    try:
        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"Decompression complete: {output_path}")
        return True
    except FileNotFoundError:
        print(f"Error: Input file not found at {gz_path}")
        return False
    except gzip.BadGzipFile:
        print(f"Error: File at {gz_path} is not a valid Gzip file or is corrupted.")
        return False
    except Exception as e:
        print(f"An error occurred during decompression: {e}")
        return False

if __name__ == "__main__":
    # Ensure target directory exists
    os.makedirs(TARGET_DIR, exist_ok=True)

    # --- Download and Unzip Training Data ---
    print("--- Processing Training Data ---")
    if download_file(SFT_TRAIN_URL, TRAIN_GZ_FILEPATH):
        # Unzip the file
        if unzip_gz_file(TRAIN_GZ_FILEPATH, TRAIN_JSONL_FILEPATH):
             # Optional: Remove the .gz file after successful decompression
             try:
                 os.remove(TRAIN_GZ_FILEPATH)
                 print(f"Removed compressed file: {TRAIN_GZ_FILEPATH}")
             except OSError as e:
                 print(f"Error removing compressed file {TRAIN_GZ_FILEPATH}: {e}")

    print("\n--- Processing Test Data ---")
    # --- Download and Unzip Test Data ---
    if download_file(SFT_TEST_URL, TEST_GZ_FILEPATH):
        # Unzip the file
        if unzip_gz_file(TEST_GZ_FILEPATH, TEST_JSONL_FILEPATH):
             # Optional: Remove the .gz file after successful decompression
             try:
                 os.remove(TEST_GZ_FILEPATH)
                 print(f"Removed compressed file: {TEST_GZ_FILEPATH}")
             except OSError as e:
                 print(f"Error removing compressed file {TEST_GZ_FILEPATH}: {e}") 