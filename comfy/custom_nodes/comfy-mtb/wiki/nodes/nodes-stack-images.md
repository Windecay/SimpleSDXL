# Stack Images

A simple way to stack images either horizontally or vertically. Stack image uses [dynamic inputs](web-dynamic-inputs).

It outputs RGBA tensors and supports RGB or RGBA as input (normalized to RGBA internally). If the image dimensions don't match they must at least match:
- in `width` when stacking vertically.
- in `height` when stacking horizontally.

Here is an example workflow using [Text To Image](nodes-text-to-image) (the text was generated using [Nous Hermes 2 Vision](https://huggingface.co/billborkowski/llava-NousResearch_Nous-Hermes-2-Vision-GGUF) thanks to the great [ComfyUI_VLM_nodes](https://github.com/gokayfem/ComfyUI_VLM_nodes) extension. For simplicity's sake, the workflow doesn't contain external nodes:

![stack_images](https://github.com/melMass/comfy_mtb/assets/7041726/a0c03621-3377-46cf-a6b4-f47e70c7d11f)
