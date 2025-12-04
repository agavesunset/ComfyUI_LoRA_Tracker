import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import textwrap

class LoRAParameterOverlay:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "custom_label": ("STRING", {"default": "", "multiline": False}),
            },
            "hidden": {
                "prompt": "PROMPT", 
                "unique_id": "UNIQUE_ID"
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "LoRA Testing/Utils"

    def run(self, image, custom_label, prompt, unique_id):
        batch_results = []
        
        # 1. 加载字体
        font_path = os.path.join(os.path.dirname(__file__), "font.ttf")
        if not os.path.exists(font_path):
            font = None # Fallback to default
        else:
            try:
                # 稍微调小一点字体以容纳更多内容，或者你可以保持 24
                font = ImageFont.truetype(font_path, 20)
            except:
                font = None

        for i in range(image.shape[0]):
            img_tensor = image[i]
            img_np = 255. * img_tensor.cpu().numpy()
            img_pil = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
            
            # --- 1. 自动溯源 KSampler 参数 ---
            sampler_info, ksampler_id = self.get_upstream_ksampler(prompt, unique_id)
            
            # --- 2. 自动溯源 LoRA 和 Checkpoint (增强版) ---
            detected_loras = []
            ckpt_name = "Unknown Model"
            if ksampler_id:
                detected_loras, ckpt_name = self.trace_models(prompt, ksampler_id)
            
            # --- 3. 构建显示文本 ---
            info_lines = []
            
            # [L1] 模型信息
            header = f"Model: {ckpt_name}"
            if custom_label: header = f"[{custom_label}] {header}"
            info_lines.append(header)

            # [L2] LoRA
            if detected_loras:
                info_lines.append(f"LoRA: {' | '.join(detected_loras)}")
            else:
                info_lines.append("LoRA: None")
            
            # [L3] 核心参数
            if sampler_info:
                seed = sampler_info.get('seed', '?')
                steps = sampler_info.get('steps', '?')
                cfg = sampler_info.get('cfg', '?')
                denoise = sampler_info.get('denoise', 1.0)
                
                line_params = f"Seed: {seed}  |  Steps: {steps}  |  CFG: {cfg}"
                if denoise < 1.0: line_params += f"  |  Denoise: {denoise}"
                info_lines.append(line_params)

                s_name = sampler_info.get('sampler_name', '?')
                sched_name = sampler_info.get('scheduler', '?')
                info_lines.append(f"Sampler: {s_name}  |  Scheduler: {sched_name}")
                
                # [Prompt] 不再截断，后续绘图时自动换行
                p_text = sampler_info.get('pos_text', '')
                n_text = sampler_info.get('neg_text', '')
                
                if p_text: info_lines.append(f"Pos: {p_text}")
                if n_text: info_lines.append(f"Neg: {n_text}")
            else:
                info_lines.append("MetaData Not Found (Link incomplete)")

            # --- 绘图 (支持自动换行) ---
            processed_img = self.draw_overlay_smart(img_pil, info_lines, font)
            
            img_out_np = np.array(processed_img).astype(np.float32) / 255.0
            img_out_tensor = torch.from_numpy(img_out_np)
            batch_results.append(img_out_tensor)

        return (torch.stack(batch_results),)

    def get_upstream_ksampler(self, prompt, unique_id):
        # ... (保持原有的 KSampler 查找逻辑不变) ...
        try:
            current_node_inputs = prompt[unique_id]['inputs']
            image_link = current_node_inputs.get('image')
            if not isinstance(image_link, list): return None, None
            source_node_id = image_link[0]
            steps = 0
            while steps < 50:
                if source_node_id not in prompt: break
                node_data = prompt[source_node_id]
                class_type = node_data.get('class_type', '')
                if "VAEDecode" in class_type:
                    samples_link = node_data['inputs'].get('samples')
                    if isinstance(samples_link, list): source_node_id = samples_link[0]
                    else: break
                elif "KSampler" in class_type or "Sampler" in class_type:
                    return self.extract_ksampler_info(prompt, source_node_id), source_node_id
                elif "Reroute" in class_type:
                    vals = list(node_data['inputs'].values())
                    if vals and isinstance(vals[0], list): source_node_id = vals[0][0]
                    else: break
                elif "Upscale" in class_type and "image" in node_data['inputs']: # 简单穿透放大
                     break 
                else: break
                steps += 1
        except: pass
        return None, None

    def trace_models(self, prompt, ksampler_id):
        found_loras = []
        ckpt_name = "Unknown"
        
        try:
            ksampler_inputs = prompt[ksampler_id]['inputs']
            model_link = ksampler_inputs.get('model')
            current_id = None
            if isinstance(model_link, list): current_id = model_link[0]
            
            steps = 0
            while current_id and steps < 30:
                if current_id not in prompt: break
                node = prompt[current_id]
                class_type = node.get('class_type', '')
                inputs = node.get('inputs', {})
                
                # 1. 记录 LoRA
                if "Lora" in class_type:
                    lora_name = inputs.get('lora_name')
                    if lora_name:
                        name_clean = str(lora_name).replace(".safetensors", "").replace(".pt", "")
                        found_loras.append(name_clean)
                
                # 2. 记录 Checkpoint / UNET (关键修改：兼容 Flux)
                # 检查所有可能的模型名称字段
                for key in ['ckpt_name', 'unet_name', 'model_name', 'diffusion_model']:
                    if key in inputs:
                        raw_name = str(inputs[key])
                        # 只取文件名，去掉路径
                        ckpt_name = raw_name.split('\\')[-1].split('/')[-1].replace(".safetensors", "")
                        return found_loras, ckpt_name # 找到底模通常就可以返回了
                
                # 向上回溯
                next_link = inputs.get('model')
                # Flux 可能会通过 'diffusion_model' 连接
                if not next_link: next_link = inputs.get('diffusion_model')
                
                if isinstance(next_link, list):
                    current_id = next_link[0]
                else:
                    break
                steps += 1
                
        except Exception as e:
            print(f"Trace Error: {e}")
            
        return found_loras, ckpt_name

    def extract_ksampler_info(self, prompt, sampler_id):
        # 保持不变，除了调用 upgraded trace_text
        data = prompt[sampler_id]['inputs']
        info = {
            "seed": data.get("seed") or data.get("noise_seed"),
            "steps": data.get("steps"),
            "cfg": data.get("cfg"),
            "sampler_name": data.get("sampler_name"),
            "scheduler": data.get("scheduler"),
            "denoise": data.get("denoise", 1.0),
            "pos_text": self.trace_text(prompt, data.get("positive")),
            "neg_text": self.trace_text(prompt, data.get("negative"))
        }
        return info

    def trace_text(self, prompt, link, depth=0):
        # 使用之前给你的最新版 trace_text，这里为了完整性再贴一次
        if depth > 10: return ""
        if not isinstance(link, list): return ""
        
        node_id = link[0]
        if node_id not in prompt: return ""
        node = prompt[node_id]
        class_type = node.get('class_type', '')
        inputs = node.get('inputs', {})

        if ("Text" in class_type or "Prompt" in class_type) and "Save" not in class_type:
            for key in ["text", "text_g", "text_l", "prompt", "string", "value"]:
                if key in inputs:
                    val = inputs[key]
                    if isinstance(val, str): return val
                    if isinstance(val, list): return self.trace_text(prompt, val, depth+1)
            return ""

        target_keys = ["conditioning", "conditioning_1", "clip_conditioning", "samples", "guidance"] # 加了 guidance
        for key in target_keys:
            if key in inputs:
                parent_link = inputs[key]
                found = self.trace_text(prompt, parent_link, depth+1)
                if found: return found

        if "Reroute" in class_type:
            first_val = list(inputs.values())
            if first_val and isinstance(first_val[0], list):
                return self.trace_text(prompt, first_val[0], depth+1)

        return ""

    def draw_overlay_smart(self, image, text_lines, font=None):
        """支持自动换行的绘图函数"""
        w, h = image.size
        
        # 1. 准备字体和测量
        if font is None:
            font = ImageFont.load_default()
            char_w, char_h = 7, 15 # 默认字体的估算
        else:
            # 获取字体度量
            try:
                # Pillow >= 10.0
                left, top, right, bottom = font.getbbox("A")
                char_h = bottom - top + 5 # 行间距
            except:
                char_h = 24 

        # 2. 处理自动换行
        # 预留左右边距
        margin = 15
        max_text_width = w - (margin * 2)
        
        final_lines = []
        for line in text_lines:
            # 如果是 Prompt 这种超长文本，进行 wrap
            # 简单粗暴的字符估算（Pillow 的 textlength 比较耗时，这里用字符数估算足矣，或者用 textwrap）
            # 为了更精准，我们使用 textwrap 配合一个估算的字符宽度
            
            # 估算一行能放多少字 (假设平均字符宽是字号的0.5倍，中文是1.0倍，取个折中0.6)
            # 如果加载了 font.ttf (20px), 字符宽约 12px
            avg_char_width = 12 if font else 7
            chars_per_line = max(10, int(max_text_width / avg_char_width))
            
            wrapped = textwrap.wrap(line, width=chars_per_line)
            final_lines.extend(wrapped)

        # 3. 计算新的高度
        footer_h = (len(final_lines) * int(char_h * 1.2)) + (margin * 2)
        new_h = h + footer_h
        
        new_img = Image.new("RGB", (w, new_h), (0, 0, 0))
        new_img.paste(image, (0, 0))
        
        draw = ImageDraw.Draw(new_img)
        
        # 4. 绘制
        y = h + margin
        for line in final_lines:
            draw.text((margin, y), str(line), fill=(255, 255, 255), font=font)
            y += int(char_h * 1.2)
            
        return new_img

NODE_CLASS_MAPPINGS = {
    "LoRAParameterOverlay": LoRAParameterOverlay
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoRAParameterOverlay": "LoRA Auto-Tracker (Smart Wrap)"
}