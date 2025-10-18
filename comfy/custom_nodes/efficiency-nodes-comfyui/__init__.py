import os
import subprocess
import importlib.util
import folder_paths
import shutil
import sys
import traceback

from  .efficiency_nodes import NODE_CLASS_MAPPINGS
#from  .py.ttl_nn_latent_upscaler import NODE_CLASS_MAPPINGS
#from  .py.city96_latent_upscaler import NODE_CLASS_MAPPINGS


WEB_DIRECTORY = "js"

CC_VERSION = 2.0

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'CC_VERSION']
