import sys

folder_paths = None

try:
    import folder_paths as _folder_paths
except (ImportError, ModuleNotFoundError):
    try:
        from comfy.cmd import folder_paths as _folder_paths
    except (ImportError, ModuleNotFoundError):
        _folder_paths = sys.modules.get("folder_paths", None)

if _folder_paths is not None and hasattr(_folder_paths, "get_filename_list") and hasattr(_folder_paths, "get_full_path_or_raise"):
    folder_paths = _folder_paths
    get_filename_list = folder_paths.get_filename_list
    get_full_path_or_raise = folder_paths.get_full_path_or_raise
else:
    from comfy.model_downloader import get_filename_list, get_full_path_or_raise

__all__ = ["get_filename_list", "get_full_path_or_raise", "folder_paths"]
