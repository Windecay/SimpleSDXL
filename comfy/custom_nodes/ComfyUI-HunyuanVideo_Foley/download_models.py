"""
download_models.py
──────────────────
Automatic model download helper for comfyui-hunyuanvideo_foley.

When the Model Loader node finds that required files are missing, it calls
`ensure_models_downloaded()` which fetches the missing files from HuggingFace
using `huggingface_hub` (a transitive dependency of `transformers`).

Files are downloaded to:
    ComfyUI/models/hunyuan_foley/
    ├── hunyuanvideo_foley.pth        (or _xl.pth, whichever is selected)
    ├── synchformer_state_dict.pth
    ├── vae_128d_48k.pth
    ├── siglip2/
    │   ├── model.safetensors
    │   ├── config.json
    │   └── preprocessor_config.json
    └── clap/
        ├── pytorch_model.bin
        ├── config.json
        ├── merges.txt
        └── vocab.json
"""

import os
import logging

log = logging.getLogger("HunyuanFoley.Download")

# ---------------------------------------------------------------------------
# File manifest
# ---------------------------------------------------------------------------

# Files from the main Tencent repo that are ALWAYS required (regardless of
# which foley checkpoint is selected).
_TENCENT_REPO = "Tencent/HunyuanVideo-Foley"

_ALWAYS_REQUIRED = [
    # (hf_repo, hf_filename, local_relative_path)
    (_TENCENT_REPO, "synchformer_state_dict.pth", "synchformer_state_dict.pth"),
    (_TENCENT_REPO, "vae_128d_48k.pth",           "vae_128d_48k.pth"),
]

# The two large foley checkpoints are downloaded on-demand (only the selected one).
_FOLEY_CHECKPOINTS = {
    "hunyuanvideo_foley.pth": (
        _TENCENT_REPO, "hunyuanvideo_foley.pth", "hunyuanvideo_foley.pth"
    ),
    "hunyuanvideo_foley_xl.pth": (
        _TENCENT_REPO, "hunyuanvideo_foley_xl.pth", "hunyuanvideo_foley_xl.pth"
    ),
}

_SIGLIP2_REPO = "google/siglip2-base-patch16-512"
_SIGLIP2_FILES = [
    (_SIGLIP2_REPO, "model.safetensors",      "siglip2/model.safetensors"),
    (_SIGLIP2_REPO, "config.json",            "siglip2/config.json"),
    (_SIGLIP2_REPO, "preprocessor_config.json", "siglip2/preprocessor_config.json"),
]

_CLAP_REPO = "laion/larger_clap_general"
_CLAP_FILES = [
    (_CLAP_REPO, "pytorch_model.bin", "clap/pytorch_model.bin"),
    (_CLAP_REPO, "config.json",       "clap/config.json"),
    (_CLAP_REPO, "merges.txt",        "clap/merges.txt"),
    (_CLAP_REPO, "vocab.json",        "clap/vocab.json"),
]


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def _download_file(repo_id: str, filename: str, local_path: str) -> None:
    """Download a single file from HuggingFace Hub to `local_path`."""
    from huggingface_hub import hf_hub_download

    local_dir = os.path.dirname(local_path)
    os.makedirs(local_dir, exist_ok=True)
    log.info(f"Downloading {filename} from {repo_id} ...")
    # hf_hub_download with local_dir places the file at local_dir/filename
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
    )
    # Verify the file landed where expected
    if not os.path.isfile(local_path):
        # Edge case: some versions may nest the file – try to locate it
        alt = os.path.join(local_dir, filename)
        if os.path.isfile(alt) and alt != local_path:
            import shutil
            shutil.move(alt, local_path)
    log.info(f"  ✓ Saved to {local_path}")


def _ensure_file(repo_id: str, hf_filename: str, local_path: str) -> bool:
    """
    Check if `local_path` exists.  Download if missing.
    Returns True if the file is now available.
    """
    if os.path.isfile(local_path):
        return True
    try:
        _download_file(repo_id, hf_filename, local_path)
        return os.path.isfile(local_path)
    except Exception as exc:
        log.error(f"Failed to download {hf_filename} from {repo_id}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ensure_models_downloaded(model_dir: str, foley_checkpoint_name: str) -> bool:
    """
    Ensure all required model files are present in `model_dir`.

    Downloads any missing files from HuggingFace.  The selected
    `foley_checkpoint_name` determines which large checkpoint is fetched.

    Returns True if all required files are present (or were successfully
    downloaded), False if one or more downloads failed.
    """
    os.makedirs(model_dir, exist_ok=True)

    # Build the full manifest for this run
    manifest = list(_ALWAYS_REQUIRED)

    # Add the selected foley checkpoint
    if foley_checkpoint_name in _FOLEY_CHECKPOINTS:
        manifest.append(_FOLEY_CHECKPOINTS[foley_checkpoint_name])
    else:
        log.warning(f"Unknown foley checkpoint '{foley_checkpoint_name}'; skipping checkpoint download.")

    manifest.extend(_SIGLIP2_FILES)
    manifest.extend(_CLAP_FILES)

    # Check which files are missing
    missing = [
        (repo, fname, os.path.join(model_dir, rel))
        for repo, fname, rel in manifest
        if not os.path.isfile(os.path.join(model_dir, rel))
    ]

    if not missing:
        return True  # Everything already present

    log.info(
        f"[Hunyuan-Foley] {len(missing)} model file(s) not found — downloading from HuggingFace..."
    )
    log.info("  This may take a while for large files. Please be patient.")

    all_ok = True
    for repo_id, hf_filename, local_path in missing:
        ok = _ensure_file(repo_id, hf_filename, local_path)
        if not ok:
            all_ok = False

    if all_ok:
        log.info("[Hunyuan-Foley] All required model files are now present.")
    else:
        log.error(
            "[Hunyuan-Foley] Some files could not be downloaded. "
            "Check the errors above and try downloading them manually."
        )
    return all_ok


def ensure_vae_downloaded(model_dir: str) -> bool:
    """
    Ensure only the VAE file is present (used by LoadDACHunyuanVAE).
    Downloads from HuggingFace if missing.
    """
    vae_path = os.path.join(model_dir, "vae_128d_48k.pth")
    if os.path.isfile(vae_path):
        return True
    log.info("[Hunyuan-Foley] VAE file not found — downloading vae_128d_48k.pth ...")
    return _ensure_file(_TENCENT_REPO, "vae_128d_48k.pth", vae_path)
