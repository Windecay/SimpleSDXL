# Nvidia_RTX_Nodes_ComfyUI

Contains RTX upscaling nodes for images and videos. Only works on Nvidia RTX GPUs.

Included nodes:

- RTX Video Super Resolution: keeps the original IMAGE output workflow for image batches and short clips.
- RTX Video Super Resolution Chunked: designed for long videos and large frame sequences. It processes frames in segments, limits GPU work by output pixel budget, and emits a compressed VIDEO result instead of a huge IMAGE tensor.
- RTX Video Super Resolution Chunked Image Sequence: compatibility node for legacy workflows that still require IMAGE sequence output after chunked processing.

Recommended long video workflow:

- LoadVideo
- RTX Video Super Resolution Chunked
- SaveVideo

The chunked node supports:

- adjustable chunk size
- memory or disk backed compressed cache
- near-lossless, balanced, or compact VIDEO compression
- optional direct audio passthrough or external audio input
- compatibility IMAGE sequence output for older merge-images-to-video nodes

Search RTX in the ComfyUI manager to install it.
