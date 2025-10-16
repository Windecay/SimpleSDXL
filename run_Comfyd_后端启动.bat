@echo off
set HF_ENDPOINT=https://hf-mirror.com
.\python_embeded\python.exe -s SimpleSDXL\comfy\main_comfyd.py  --windows-standalone-build --output-directory ../../users/ComfyUI
echo All done.
pause