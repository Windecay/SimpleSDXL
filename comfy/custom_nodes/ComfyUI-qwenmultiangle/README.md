# ComfyUI-qwenmultiangle

A ComfyUI custom node for 3D camera angle control. Provides an interactive Three.js viewport to adjust camera angles and outputs formatted prompt strings for multi-angle image generation.
![img.png](img.png)
## Features

- **Interactive 3D Camera Control** - Drag handles in the Three.js viewport to adjust:
  - Horizontal angle (azimuth): 0° - 360°
  - Vertical angle (elevation): -30° to 90°
  - Zoom level: 0 - 10
- **Real-time Preview** - Connect an image input to see it displayed in the 3D scene
- **Prompt Output** - Outputs descriptive camera angle prompts like `"front view, eye level, medium shot (horizontal: 0, vertical: 0, zoom: 5.0)"`
- **Bidirectional Sync** - Slider widgets and 3D handles stay synchronized

## Installation

1. Navigate to your ComfyUI custom nodes folder:
   ```bash
   cd ComfyUI/custom_nodes
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/jtydhr88/ComfyUI-qwenmultiangle.git
   ```

3. Restart ComfyUI

## Usage

1. Add the **Qwen Multiangle Camera** node from the `image/multiangle` category
2. Optionally connect an IMAGE input to preview in the 3D scene
3. Adjust camera angles by:
   - Dragging the colored handles in the 3D viewport
   - Using the slider widgets
4. The node outputs a prompt string describing the camera angle

### Controls

| Handle | Color | Control |
|--------|-------|---------|
| Ring handle | Pink | Horizontal angle (azimuth) |
| Arc handle | Cyan | Vertical angle (elevation) |
| Line handle | Gold | Zoom/distance |

### Output Prompt Format

```
{direction}, {elevation}, {shot_type} (horizontal: {h}°, vertical: {v}°, zoom: {z})
```

Examples:
- `front view, eye level, medium shot (horizontal: 0, vertical: 0, zoom: 5.0)`
- `right side view, high angle, close-up (horizontal: 90, vertical: 45, zoom: 8.5)`
- `back view, bird's eye view, wide shot (horizontal: 180, vertical: 75, zoom: 1.0)`

## Credits

### Original Implementation

This ComfyUI node is based on [qwenmultiangle](https://github.com/jtydhr88/qwenmultiangle), a standalone web application for camera angle control.

The original project was inspired by:
- [multimodalart/qwen-image-multiple-angles-3d-camera](https://huggingface.co/spaces/multimodalart/qwen-image-multiple-angles-3d-camera) on Hugging Face Spaces
- [fal.ai - Qwen Image Edit 2511 Multiple Angles](https://fal.ai/models/fal-ai/qwen-image-edit-2511-multiple-angles/)

## License

MIT
