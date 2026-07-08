import hashlib
import os
import shutil
from typing import BinaryIO
from app.core.logging import logger


def calculate_file_sha256(file_obj: BinaryIO) -> str:
    """Calculates SHA256 hash of a file object dynamically."""
    sha256_hash = hashlib.sha256()
    # Read in chunks of 4kb to handle large files gracefully
    file_obj.seek(0)
    for byte_block in iter(lambda: file_obj.read(4096), b""):
        sha256_hash.update(byte_block)
    file_obj.seek(0)  # Reset cursor
    return sha256_hash.hexdigest()


def save_file_safely(file_obj: BinaryIO, destination_path: str) -> None:
    """Saves a binary file to disk and ensures parent directories are built."""
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    with open(destination_path, "wb") as buffer:
        file_obj.seek(0)
        shutil.copyfileobj(file_obj, buffer)
    logger.info(f"File saved successfully to {destination_path}")


def remove_file_safely(file_path: str) -> None:
    """Safely deletes a file if it exists."""
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info(f"File deleted successfully from {file_path}")
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")
