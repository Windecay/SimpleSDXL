"""
Downloads utility module for SeedVR2
Handles model and VAE downloads from HuggingFace repositories with integrity validation
"""

import os
import hashlib
import json
import urllib.request
from typing import Optional
from tqdm import tqdm
import time

from .model_registry import MODEL_REGISTRY, DEFAULT_VAE
from .constants import (
    get_base_cache_dir,
    find_model_file,
    get_validation_cache_path,
    HUGGINGFACE_BASE_URL,
    DOWNLOAD_CHUNK_SIZE,
    DOWNLOAD_MAX_RETRIES,
    DOWNLOAD_RETRY_DELAY
)

def load_validation_cache():
    """Load validation cache"""
    cache_path = get_validation_cache_path()
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}


def save_validation_cache(cache):
    """Save validation cache"""
    cache_path = get_validation_cache_path()
    try:
        with open(cache_path, 'w') as f:
            json.dump(cache, f, indent=2)
    except:
        pass


def is_file_validated_cached(filepath: str) -> bool:
    """Check if file is validated using cache (fast)"""
    if not os.path.exists(filepath):
        return False
    
    cache = load_validation_cache()
    filename = os.path.basename(filepath)
    
    if filename in cache:
        cached = cache[filename]
        # Check if file hasn't changed since validation
        if (cached.get('size') == os.path.getsize(filepath) and 
            abs(cached.get('mtime', 0) - os.path.getmtime(filepath)) < 2):
            return True
    
    return False


def validate_file(filepath: str, expected_hash: Optional[str] = None) -> bool:
    """File validation with hash check and cache update"""
    if not os.path.exists(filepath):
        return False
    
    file_size = os.path.getsize(filepath)
    if file_size == 0:
        return False
    
    # Quick safetensors header check
    if filepath.endswith('.safetensors'):
        try:
            with open(filepath, 'rb') as f:
                header_size = int.from_bytes(f.read(8), 'little')
                if header_size <= 0 or header_size > file_size:
                    return False
        except:
            return False
    
    # Only calculate hash if we have an expected hash to compare
    if expected_hash:
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(DOWNLOAD_CHUNK_SIZE):
                sha256.update(chunk)
        actual_hash = sha256.hexdigest()
        
        if actual_hash != expected_hash:
            return False
        
        # Update cache with validated info
        cache = load_validation_cache()
        cache[os.path.basename(filepath)] = {
            "size": os.path.getsize(filepath),
            "mtime": os.path.getmtime(filepath),
            "hash": expected_hash
        }
        save_validation_cache(cache)
    
    return True


def download_with_resume(url: str, filepath: str, debug=None) -> bool:
    """Download with resume support and progress bar"""
    temp_file = f"{filepath}.download"
    existing_size = os.path.getsize(temp_file) if os.path.exists(temp_file) else 0
    
    headers = {'Range': f'bytes={existing_size}-'} if existing_size > 0 else {}
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            content_length = int(response.headers.get('Content-Length', 0))
            total_size = existing_size + content_length
            
            pbar = tqdm(total=total_size, initial=existing_size, unit='B', 
                       unit_scale=True, unit_divisor=1024, 
                       desc=os.path.basename(filepath))
            
            with open(temp_file, 'ab' if existing_size else 'wb') as f:
                while chunk := response.read(DOWNLOAD_CHUNK_SIZE):
                    f.write(chunk)
                    pbar.update(len(chunk))
            pbar.close()
        
        os.replace(temp_file, filepath)
        return True
        
    except Exception as e:
        if debug:
            debug.log(f"Download error: {e}", level="ERROR", category="download", force=True)
        return False


def download_weight(model: str, model_dir: Optional[str] = None, debug=None) -> bool:
    """Download SeedVR2 model and VAE with integrity checking"""
    cache_dir = model_dir or get_base_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    
    files_to_download = [
        (model, MODEL_REGISTRY.get(model)),
        (DEFAULT_VAE, MODEL_REGISTRY.get(DEFAULT_VAE))
    ]
    
    for filename, model_info in files_to_download:
        if not model_info:
            if debug:
                debug.log(f"{filename} not in registry, skipping validation", 
                         level="WARNING", category="setup")
            continue
        
        # Check if file exists in any registered path first
        existing_filepath = find_model_file(filename, fallback_dir=cache_dir)
        
        # Use existing file path if it exists, otherwise download to cache_dir
        if os.path.exists(existing_filepath):
            filepath = existing_filepath
        else:
            filepath = os.path.join(cache_dir, filename)
            
        expected_hash = model_info.sha256
        repo = model_info.repo
        
        # Quick cache check first
        if is_file_validated_cached(filepath):
            continue
        
        # File exists - validate it
        if os.path.exists(filepath):
            if debug:
                debug.log(f"Validating {filename}...", category="setup", force=True)
            
            if validate_file(filepath, expected_hash):
                continue
            else:
                # File is corrupted
                if debug:
                    debug.log(f"File corrupted: {filename}, re-downloading...", 
                            level="WARNING", category="download", force=True)
                os.remove(filepath)
                # Clear from cache
                cache = load_validation_cache()
                if filename in cache:
                    del cache[filename]
                    save_validation_cache(cache)
        
        # Download file
        url = HUGGINGFACE_BASE_URL.format(repo=repo, filename=filename)
        temp_file = f"{filepath}.download"
        
        if os.path.exists(temp_file) and debug:
            size_gb = os.path.getsize(temp_file) / (1024**3)
            debug.log(f"Resuming {filepath} from {size_gb:.2f}GB (source: {url})", 
                     category="download", force=True)
        elif debug:
            debug.log(f"Downloading {filepath} from {url}...", 
                     category="download", force=True)
        
        # Download with retries
        success = False
        for attempt in range(DOWNLOAD_MAX_RETRIES):
            if attempt > 0:
                time.sleep(DOWNLOAD_RETRY_DELAY * attempt)
                if debug:
                    debug.log(f"Retry {attempt}/{DOWNLOAD_MAX_RETRIES}", 
                             category="download", force=True)
            
            if download_with_resume(url, filepath, debug):
                # Validate downloaded file
                if validate_file(filepath, expected_hash):
                    if debug:
                        debug.log(f"Downloaded and validated: {filename}", 
                                 category="success", force=True)
                    success = True
                    break
                else:
                    # Remove corrupted download
                    for f in [filepath, temp_file]:
                        if os.path.exists(f):
                            os.remove(f)
                    if debug:
                        debug.log(f"Downloaded file failed validation, retrying...", 
                                 level="WARNING", category="download", force=True)
        
        if not success:
            if debug:
                debug.log(f"Failed to download {filename} after {DOWNLOAD_MAX_RETRIES} attempts", 
                         level="ERROR", category="download", force=True)
                debug.log(f"Manual download: https://huggingface.co/{repo}/blob/main/{filename}", 
                         category="info", force=True)
                debug.log(f"Save to: {filepath}", category="info", force=True)
            return False
    
    return True