import os
import sys

from aiohttp import web

import folder_paths
from server import PromptServer

routes = PromptServer.instance.routes

def _resolve_restart_script_path() -> str:
    argv0 = sys.argv[0] if sys.argv else ""
    if argv0:
        p = os.path.expandvars(os.path.expanduser(argv0))
        if os.path.isabs(p) and os.path.exists(p):
            return p

        cwd = os.getcwd()
        if os.path.exists(os.path.join(cwd, p)):
            return os.path.join(cwd, p)

        parts = p.replace("/", os.sep).split(os.sep)
        if parts and parts[0].lower() == os.path.basename(cwd).lower():
            candidate = os.path.join(cwd, *parts[1:])
            if os.path.exists(candidate):
                return candidate

    try:
        comfy_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        fallback_name = os.path.basename(argv0) if argv0 else "main.py"
        candidate = os.path.join(comfy_dir, fallback_name)
        if os.path.exists(candidate):
            return candidate
    except Exception:
        pass

    return argv0

@routes.get("/ttN/reboot")
def restart(self):
    try:
        sys.stdout.close_log()
    except Exception as e:
        pass

    print(f"\nRestarting...\n\n")
    script_path = _resolve_restart_script_path()
    return os.execv(sys.executable, [sys.executable, script_path] + (sys.argv[1:] if len(sys.argv) > 1 else []))

@routes.get("/ttN/models")
def get_models(self):
    ckpts = folder_paths.get_filename_list("checkpoints")
    return web.json_response(list(map(lambda a: os.path.splitext(a)[0], ckpts)))

@routes.get("/ttN/loras")
def get_loras(self):
    loras = folder_paths.get_filename_list("loras")
    return web.json_response(loras)
