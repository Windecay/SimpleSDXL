此节点由抖音博主：绘画小子 制作。
Wan2.2 图生视频增强节点，专门针对4步LoRA（如 lightx2v）的慢动作问题进行优化。
🎯 解决的问题
✅ 减少慢动作拖影：提升运动幅度15-30%
✅ 保持画面亮度：增强算法不破坏亮度分布
✅ 单帧输入优化：专为单帧图生视频设计
✅ 即插即用：完全兼容原版Wan2.2工作流
📦 安装
方法1：ComfyUI Manager（推荐）
打开ComfyUI Manager
搜索 PainterI2V
点击安装
方法2：手动安装

# 进入ComfyUI的custom_nodes目录
cd ComfyUI/custom_nodes

# 克隆仓库
git clone https://github.com/princepainter/ComfyUI-PainterI2V.git

# 重启ComfyUI
🚀 使用方法
替换节点：在工作流中将 WanImageToVideo 替换为 PainterI2V
参数设置：
motion_amplitude: 1.15（推荐起始值）
其他参数与原版保持一致
场景参数推荐：
| 运动类型       | 推荐参数   | 示例提示词     |
| -------------- | ---------- | -------------- |
| 快速（跑步/跳跃） | 1.25-1.35  | "快速向前奔跑" |
| 正常（走路/挥手） | 1.10-1.20  | "流畅地行走"   |
| 慢动作特效     | 1.00~1.10    | "略微增强动态和运镜"     |

提示词优化：
明确描述运动节奏，如"快速奔跑"、"流畅行走"
避免模糊描述如"移动"、"走动"
📊 技术细节
| 参数值       | 运动提升 | 亮度变化 | 适用场景   |
| ------------ | -------- | -------- | ---------- |
| 1.0（原版）  | 0%       | 无       | 慢动作特效 |
| 1.15（默认） | +15%     | 无       | 通用场景   |
| 1.3          | +30%     | 无       | 体育运动   |
| 1.5          | +50%     | 无       | 极限运动   |
核心算法原理
亮度保护的运动缩放：放大运动向量前分离亮度均值
零latent初始化：严格保持4步LoRA的时序依赖链
参考帧增强：使用reference_latents保持主体一致性，不约束运动
🔧 进阶技巧
最佳效果：配合强运动提示词使用
运动过快：每次减少 motion_amplitude 0.05
仍然偏慢：可适当增大到1.4
亮度异常：确保 motion_amplitude ≥ 1.0（不建议<1.0）

🙏 致谢
Wan2.2 团队: 提供惊人的视频生成模型
ComfyUI 社区: 灵活的节点系统
问题反馈者: 帮助完善此节点
<div align="center">
如果这个项目对你有帮助，请给颗星 ⭐️ 支持一下！

🎨 ComfyUI-PainterI2V
<div align="center">
English | 中文
https://github.com/princepainter/ComfyUI-PainterI2V/releases
https://opensource.org/licenses/MIT
https://github.com/comfyanonymous/ComfyUI
</div>
<span id="english">
Wan2.2 Image-to-Video enhancement node that specifically fixes the slow-motion issue in 4-step LoRAs (e.g., lightx2v).
🎯 Problems Solved
✅ Reduces Slow-Motion Drag: Increases motion amplitude by 15-30%
✅ Maintains Brightness Stability: Enhancement algorithm preserves brightness distribution
✅ Optimized for Single Frame: Designed specifically for single-frame image-to-video workflows
✅ Plug & Play: Fully compatible with original Wan2.2 workflows
📦 Installation
Method 1: ComfyUI Manager (Recommended)
Open ComfyUI Manager
Search for PainterI2V
Click Install
Method 2: Manual Installation
bash
复制
# Navigate to ComfyUI's custom_nodes directory
cd ComfyUI/custom_nodes

# Clone the repository
git clone https://github.com/princepainter/ComfyUI-PainterI2V.git

# Restart ComfyUI
🚀 Usage
Replace Node: In your workflow, replace WanImageToVideo with PainterI2V
Parameter Settings:
motion_amplitude: 1.15 (Recommended starting value)
Keep other parameters identical to the original
Scene-Specific Settings:
| Motion Type               | Recommended Parameter | Example Prompt          |
|---------------------------|-----------------------|-------------------------|
| Fast (Running/Jumping)    | 1.25-1.35             | "Running forward quickly" |
| Normal (Walking/Waving)   | 1.10-1.20             | "Walking smoothly"      |
| Slow-motion Effect        | 1.00-1.10               | "Slightly enhance dynamics and camera movement"         |

Prompt Tips:
Clearly describe motion rhythm (e.g., "quickly running", "smoothly walking")
Avoid vague descriptions like "moving" or "walking"
📊 Technical Details
| Parameter Value | Motion Enhancement | Brightness Change | Applicable Scenario |
| --------------- | ------------------ | ----------------- | ------------------- |
| 1.0 (Original)  | 0%                 | None              | Slow-motion Effects |
| 1.15 (Default)  | +15%               | None              | General Scenarios   |
| 1.3             | +30%               | None              | Sports Events       |
| 1.5             | +50%               | None              | Extreme Sports      |
Core Algorithm
Brightness-Preserving Motion Scaling: Separates motion vectors from brightness mean before amplification
Zero Latent Initialization: Maintains 4-step LoRA's strict temporal dependency chain
Reference Frame Enhancement: Uses reference_latents for subject consistency without motion constraints
🔧 Advanced Tips
For Best Results: Combine with strong motion prompts
If Motion Too Fast: Decrease motion_amplitude by 0.05 increments
If Still Slow: Increase motion_amplitude up to 1.4 max
Brightness Issues: Ensure motion_amplitude ≥ 1.0 (values < 1.0 not recommended)

Example workflow (JSON)
Sample input/output
📄 License
MIT License







