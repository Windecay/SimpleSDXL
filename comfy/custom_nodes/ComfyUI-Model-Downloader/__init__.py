from .nodes.model_downloader import (
    DownloadCheckpoint,
    DownloadLora,
    DownloadVAE,
    DownloadUNET,
    DownloadControlNet
)
from .nodes.simple_batch_downloader import (SimpleBatchDownloader,SimpleModelDownloader)
                                            

NODE_CLASS_MAPPINGS = {
    "DownloadCheckpoint": DownloadCheckpoint,
    "DownloadLora": DownloadLora,
    "DownloadVAE": DownloadVAE,
    "DownloadUNET": DownloadUNET,
    "DownloadControlNet": DownloadControlNet,
    "SimpleBatchDownloader": SimpleBatchDownloader,
    "SimpleModelDownloader": SimpleModelDownloader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadCheckpoint": "(Down)load Checkpoint",
    "DownloadLora": "(Down)load LoRA",
    "DownloadVAE": "(Down)load VAE",
    "DownloadUNET": "(Down)load UNET",
    "DownloadControlNet": "(Down)load ControlNet",
    "SimpleBatchDownloader": "Simple Batch Downloader",
    "SimpleModelDownloader": "Simple Model Downloader"
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]