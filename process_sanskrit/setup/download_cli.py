import os
import sys
import requests
import gzip
import shutil
import importlib.resources

# --- Configuration ---
# GitHub Release Info
REPO_OWNER = "Giacomo-De-Luca"
REPO_NAME = "Process-Sanskrit"
RELEASE_TAG = "v1.0.0" # Consider making this dynamic or configurable later
ASSET_NAME = "SQliteDB.sqlite.gz" # Make sure this is correct!
# Target Location within the package
TARGET_FOLDER_NAME = "resources"
# --- End Configuration ---

# Derive the final unzipped filename
if ASSET_NAME.endswith(".gz"):
    UNZIPPED_FILENAME = ASSET_NAME[:-3]
else:
    UNZIPPED_FILENAME = ASSET_NAME

DOWNLOAD_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/download/{RELEASE_TAG}/{ASSET_NAME}"

def download_and_unzip(target_dir, asset_name, download_url):
    """Downloads and unzips the asset into the target directory."""
    os.makedirs(target_dir, exist_ok=True)
    downloaded_gz_path = os.path.join(target_dir, asset_name)
    unzipped_file_path = os.path.join(target_dir, UNZIPPED_FILENAME)

    print(f"Target directory: {target_dir}")
    print(f"Download URL: {download_url}")
    print(f"Output file: {unzipped_file_path}")

    # Check if file already exists
    if os.path.exists(unzipped_file_path):
        print(f"File '{unzipped_file_path}' already exists. Skipping download.")
        return True # Indicate success or skipped

    try:
        print(f"Downloading '{asset_name}'...")
        # Use verify=True by default for security.
        with requests.get(download_url, stream=True, timeout=120, verify=True) as response:
            response.raise_for_status() # Raise an exception for bad status codes

            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            downloaded_size = 0

            print(f"Saving to '{downloaded_gz_path}'...")
            with open(downloaded_gz_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    progress = int(50 * downloaded_size / total_size) if total_size else 0
                    sys.stdout.write(f"\r[{'#' * progress}{'.' * (50 - progress)}] {downloaded_size / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB")
                    sys.stdout.flush()
            print("\nDownload complete.")

        if asset_name.endswith(".gz"):
            print(f"Unzipping '{downloaded_gz_path}' to '{unzipped_file_path}'...")
            with gzip.open(downloaded_gz_path, 'rb') as f_in:
                with open(unzipped_file_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print("Unzipping complete.")
            print(f"Cleaning up '{downloaded_gz_path}'...")
            os.remove(downloaded_gz_path)
            print("Cleanup complete.")
        else:
             # If not gzipped, the downloaded file is the final file.
             if unzipped_file_path != downloaded_gz_path:
                 shutil.move(downloaded_gz_path, unzipped_file_path)
                 print(f"Moved '{downloaded_gz_path}' to '{unzipped_file_path}'.")

        print(f"\nSuccess! Asset placed in '{unzipped_file_path}'.")
        return True

    except requests.exceptions.RequestException as e:
        print(f"\nError during download: {e}", file=sys.stderr)
    except Exception as e:
        print(f"\nAn error occurred during download/unzip: {e}", file=sys.stderr)
    finally:
        # Ensure partial downloads are cleaned up on error
        if os.path.exists(downloaded_gz_path) and not os.path.exists(unzipped_file_path):
             try:
                 os.remove(downloaded_gz_path)
                 print(f"Removed partially downloaded file: {downloaded_gz_path}")
             except OSError as rm_err:
                 print(f"Error removing file {downloaded_gz_path} on error: {rm_err}", file=sys.stderr)
    return False # Indicate failure


def update_database():
    """
    Command-line entry point function to download/update the database.
    Finds the installed package's resource directory.
    """
    print("Attempting to download/update the process-sanskrit database...")

    try:
        # Use importlib.resources to find the 'resources' directory within the installed package
        # This is the modern and reliable way.
        # 'process_sanskrit' should match the actual package name installed.
        resource_dir_ref = importlib.resources.files('process_sanskrit').joinpath(TARGET_FOLDER_NAME)

        # importlib.resources might return a Traversable object. Convert to string path.
        # Ensure the parent directory exists before trying to create the target.
        target_path = str(resource_dir_ref)
        print(f"Determined target resource directory: {target_path}")

        if not download_and_unzip(target_path, ASSET_NAME, DOWNLOAD_URL):
            print("\nDatabase download/update failed.", file=sys.stderr)
            sys.exit(1) # Exit with error code
        else:
            print("\nDatabase download/update process finished.")

    except ModuleNotFoundError:
         print(f"Error: Could not find the installed package 'process_sanskrit'. Is it installed correctly?", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    # Allow running this script directly for testing
    update_database()