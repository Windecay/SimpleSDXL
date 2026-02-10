from .PainterPrompt import NODE_CLASS_MAPPINGS as PainterPrompt_MAPPINGS
from .PainterPrompt import NODE_DISPLAY_NAME_MAPPINGS as PainterPrompt_NAMES

from .PainterI2V import NODE_CLASS_MAPPINGS as PainterI2V_MAPPINGS
from .PainterI2V import NODE_DISPLAY_NAME_MAPPINGS as PainterI2V_NAMES

from .PainterI2VAdvanced import NODE_CLASS_MAPPINGS as PainterI2VAdvanced_MAPPINGS
from .PainterI2VAdvanced import NODE_DISPLAY_NAME_MAPPINGS as PainterI2VAdvanced_NAMES

from .PainterAI2V import NODE_CLASS_MAPPINGS as PainterAI2V_MAPPINGS
from .PainterAI2V import NODE_DISPLAY_NAME_MAPPINGS as PainterAI2V_NAMES

from .PainterAV2V import NODE_CLASS_MAPPINGS as PainterAV2V_MAPPINGS
from .PainterAV2V import NODE_DISPLAY_NAME_MAPPINGS as PainterAV2V_NAMES

from .PainterSampler import NODE_CLASS_MAPPINGS as PainterSampler_MAPPINGS
from .PainterSampler import NODE_DISPLAY_NAME_MAPPINGS as PainterSampler_NAMES

from .PainterSamplerLTXV import NODE_CLASS_MAPPINGS as PainterSamplerLTXV_MAPPINGS
from .PainterSamplerLTXV import NODE_DISPLAY_NAME_MAPPINGS as PainterSamplerLTXV_NAMES

from .PainterLTX2V import NODE_CLASS_MAPPINGS as PainterLTX2V_MAPPINGS
from .PainterLTX2V import NODE_DISPLAY_NAME_MAPPINGS as PainterLTX2V_NAMES

from .PainterLTX2VPlus import NODE_CLASS_MAPPINGS as PainterLTX2VPlus_MAPPINGS
from .PainterLTX2VPlus import NODE_DISPLAY_NAME_MAPPINGS as PainterLTX2VPlus_NAMES

from .PainterFLF2V import NODE_CLASS_MAPPINGS as PainterFLF2V_MAPPINGS
from .PainterFLF2V import NODE_DISPLAY_NAME_MAPPINGS as PainterFLF2V_NAMES

from .PainterMultiF2V import NODE_CLASS_MAPPINGS as PainterMultiF2V_MAPPINGS
from .PainterMultiF2V import NODE_DISPLAY_NAME_MAPPINGS as PainterMultiF2V_NAMES

from .PainterLongVideo import NODE_CLASS_MAPPINGS as PainterLongVideo_MAPPINGS
from .PainterLongVideo import NODE_DISPLAY_NAME_MAPPINGS as PainterLongVideo_NAMES

from .PainterSequentialF2V import NODE_CLASS_MAPPINGS as PainterSequentialF2V_MAPPINGS
from .PainterSequentialF2V import NODE_DISPLAY_NAME_MAPPINGS as PainterSequentialF2V_NAMES

from .PainterFluxImageEdit import NODE_CLASS_MAPPINGS as PainterFluxImageEdit_MAPPINGS
from .PainterFluxImageEdit import NODE_DISPLAY_NAME_MAPPINGS as PainterFluxImageEdit_NAMES

from .PainterQwenImageEditPlus import NODE_CLASS_MAPPINGS as PainterQwenImageEditPlus_MAPPINGS
from .PainterQwenImageEditPlus import NODE_DISPLAY_NAME_MAPPINGS as PainterQwenImageEditPlus_NAMES

from .PainterVRAM import NODE_CLASS_MAPPINGS as PainterVRAM_MAPPINGS
from .PainterVRAM import NODE_DISPLAY_NAME_MAPPINGS as PainterVRAM_NAMES

from .PainterVideoCombine import NODE_CLASS_MAPPINGS as PainterVideoCombine_MAPPINGS
from .PainterVideoCombine import NODE_DISPLAY_NAME_MAPPINGS as PainterVideoCombine_NAMES

from .PainterVideoUpscale import NODE_CLASS_MAPPINGS as PainterVideoUpscale_MAPPINGS
from .PainterVideoUpscale import NODE_DISPLAY_NAME_MAPPINGS as PainterVideoUpscale_NAMES

from .PainterVideoInfo import NODE_CLASS_MAPPINGS as PainterVideoInfo_MAPPINGS
from .PainterVideoInfo import NODE_DISPLAY_NAME_MAPPINGS as PainterVideoInfo_NAMES

from .PainterFrameCount import NODE_CLASS_MAPPINGS as PainterFrameCount_MAPPINGS
from .PainterFrameCount import NODE_DISPLAY_NAME_MAPPINGS as PainterFrameCount_NAMES

from .PainterImageLoad import NODE_CLASS_MAPPINGS as PainterImageLoad_MAPPINGS
from .PainterImageLoad import NODE_DISPLAY_NAME_MAPPINGS as PainterImageLoad_NAMES

from .PainterImageFromBatch import NODE_CLASS_MAPPINGS as PainterImageFromBatch_MAPPINGS
from .PainterImageFromBatch import NODE_DISPLAY_NAME_MAPPINGS as PainterImageFromBatch_NAMES

from .PainterCombineFromBatch import NODE_CLASS_MAPPINGS as PainterCombineFromBatch_MAPPINGS
from .PainterCombineFromBatch import NODE_DISPLAY_NAME_MAPPINGS as PainterCombineFromBatch_NAMES

from .PainterAudioLength import NODE_CLASS_MAPPINGS as PainterAudioLength_MAPPINGS
from .PainterAudioLength import NODE_DISPLAY_NAME_MAPPINGS as PainterAudioLength_NAMES

from .PainterAudioCut import NODE_CLASS_MAPPINGS as PainterAudioCut_MAPPINGS
from .PainterAudioCut import NODE_DISPLAY_NAME_MAPPINGS as PainterAudioCut_NAMES

from .PainterS2Vplus import NODE_CLASS_MAPPINGS as PainterS2Vplus_MAPPINGS
from .PainterS2Vplus import NODE_DISPLAY_NAME_MAPPINGS as PainterS2Vplus_NAMES

from .PainterHumoAV2V import NODE_CLASS_MAPPINGS as PainterHumoAV2V_MAPPINGS
from .PainterHumoAV2V import NODE_DISPLAY_NAME_MAPPINGS as PainterHumoAV2V_NAMES

from .PainterHumoI2V import NODE_CLASS_MAPPINGS as PainterHumoI2V_MAPPINGS
from .PainterHumoI2V import NODE_DISPLAY_NAME_MAPPINGS as PainterHumoI2V_NAMES


__version__ = "1.0.0"

WEB_DIRECTORY = "./web/js"

NODE_CLASS_MAPPINGS = {
    **PainterPrompt_MAPPINGS,
    **PainterI2V_MAPPINGS,
    **PainterI2VAdvanced_MAPPINGS,
    **PainterAI2V_MAPPINGS,
    **PainterAV2V_MAPPINGS,
    **PainterSampler_MAPPINGS,
    **PainterSamplerLTXV_MAPPINGS,
    **PainterLTX2V_MAPPINGS,
    **PainterLTX2VPlus_MAPPINGS,
    **PainterFLF2V_MAPPINGS,
    **PainterMultiF2V_MAPPINGS,
    **PainterLongVideo_MAPPINGS,
    **PainterSequentialF2V_MAPPINGS,
    **PainterFluxImageEdit_MAPPINGS,
    **PainterQwenImageEditPlus_MAPPINGS,
    **PainterVRAM_MAPPINGS,
    **PainterVideoCombine_MAPPINGS,
    **PainterVideoUpscale_MAPPINGS,
    **PainterVideoInfo_MAPPINGS,
    **PainterFrameCount_MAPPINGS,
    **PainterImageLoad_MAPPINGS,
    **PainterImageFromBatch_MAPPINGS,
    **PainterCombineFromBatch_MAPPINGS,
    **PainterAudioLength_MAPPINGS,
    **PainterAudioCut_MAPPINGS,
    **PainterS2Vplus_MAPPINGS,
    **PainterHumoAV2V_MAPPINGS,
    **PainterHumoI2V_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **PainterPrompt_NAMES,
    **PainterI2V_NAMES,
    **PainterI2VAdvanced_NAMES,
    **PainterAI2V_NAMES,
    **PainterAV2V_NAMES,
    **PainterSampler_NAMES,
    **PainterSamplerLTXV_NAMES,
    **PainterLTX2V_NAMES,
    **PainterLTX2VPlus_NAMES,
    **PainterFLF2V_NAMES,
    **PainterMultiF2V_NAMES,
    **PainterLongVideo_NAMES,
    **PainterSequentialF2V_NAMES,
    **PainterFluxImageEdit_NAMES,
    **PainterQwenImageEditPlus_NAMES,
    **PainterVRAM_NAMES,
    **PainterVideoCombine_NAMES,
    **PainterVideoUpscale_NAMES,
    **PainterVideoInfo_NAMES,
    **PainterFrameCount_NAMES,
    **PainterImageLoad_NAMES,
    **PainterImageFromBatch_NAMES,
    **PainterCombineFromBatch_NAMES,
    **PainterAudioLength_NAMES,
    **PainterAudioCut_NAMES,
    **PainterS2Vplus_NAMES,
    **PainterHumoAV2V_NAMES,
    **PainterHumoI2V_NAMES,
}


__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
    "__version__",
]

print(f"[PainterNodes] Loaded {len(NODE_CLASS_MAPPINGS)} nodes successfully!")
