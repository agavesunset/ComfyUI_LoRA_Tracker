ComfyUI LoRA Auto-Tracker
A powerful and intelligent custom node for ComfyUI that automatically tracks and stamps generation parameters onto your images. Perfect for LoRA testing, model comparison, and archiving.

一个强大且智能的 ComfyUI 自定义节点，能够自动追踪并将生成参数“烙印”在图片下方。非常适合用于 LoRA 测试、模型对比和作品归档。

(You can upload your image_81464d.png here as a preview / 你可以在这里上传你的效果图)

✨ Features (功能特性)
🕵️ Auto-Detection (全自动检测): No need to manually input Seed, Steps, CFG, Sampler, or Scheduler. The node reads the workflow history directly.

无需手动输入任何参数，节点直接读取工作流历史。

🧠 Smart Backtracking (智能溯源): Capable of tracing back through complex workflows, including Reroute, FluxGuidance, ReferenceLatent, and Conditioning nodes to find the original prompt and sampler.

能够穿透复杂的中间节点（如 FluxGuidance, Reroute 等）找到原始的 Prompt 和 Sampler。

📜 Smart Text Wrapping (智能文本换行): Long prompts are automatically wrapped to fit the image width. The footer height adjusts dynamically. No more truncated text!

长提示词会自动换行以适应图片宽度，底部黑边高度动态调整，不再有文字被切断的问题。

🌏 Multi-Language Support (多语言支持): Solves the "Tofu" (□□) problem for Chinese/Japanese/Korean characters by allowing custom font loading.

通过加载自定义字体，完美解决中文/日文等字符显示为方框的问题。

🤖 Flux & SDXL Ready: Correctly identifies model names (unet_name, ckpt_name) for both Standard SD and Flux workflows.

完美支持 Flux 和 SDXL，能够正确识别不同加载器的模型名称。

📦 Installation (安装)
Navigate to your ComfyUI custom nodes directory:

Bash
cd ComfyUI/custom_nodes/
Clone this repository:

Bash
git clone https://github.com/yourusername/ComfyUI-LoRA-Auto-Tracker.git
Important: Restart ComfyUI.

🛠️ Setup for Chinese/Custom Fonts (字体设置 - 重要)
To display Chinese characters or use a specific style, you must provide a .ttf font file. 为了显示中文或使用特定字体，你需要提供一个 .ttf 字体文件。

Find a font file (e.g., msyh.ttf, SimHei.ttf, or any font you like).

Rename it to font.ttf.

Place it inside the node folder:

Plaintext
ComfyUI/custom_nodes/ComfyUI-LoRA-Auto-Tracker/
├── __init__.py
├── lora_tracker.py
└── font.ttf  <-- Put your font here (放在这里)
Note: If font.ttf is missing, the node will use the system default font, which may not support non-English characters. 注意：如果没有 font.ttf，节点将回退到系统默认字体，可能无法显示中文。

🚀 Usage (使用方法)
Add Node: Right-click -> LoRA Testing/Utils -> LoRA Auto-Tracker (Smart Wrap).

Connect:

image: Connect your image source (usually from VAE Decode).

(Output): Connect to Save Image or Preview Image.

Optional:

custom_label: Add a custom note (e.g., "v1.0 Test") that will appear before the model name.

Supported Nodes for Traceback (支持的溯源节点)
Standard KSampler, KSampler Advanced

FluxGuidance

ReferenceLatent

Reroute nodes

PrimitiveNode (String)

LoraLoader (Standard & Custom)

CheckpointLoaderSimple, UNETLoader, DiffusionModelLoader
