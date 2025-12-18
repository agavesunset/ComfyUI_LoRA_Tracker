from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def _basename_no_ext(path_like: Any) -> str:
    s = str(path_like)
    base = s.split("\\")[-1].split("/")[-1]
    for ext in (".safetensors", ".pt", ".pth", ".ckpt"):
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
            break
    return base


def _get_prompt_node(prompt: Dict[Any, Any], node_id: Any) -> Optional[Dict[str, Any]]:
    """Safely retrieve a node from a ComfyUI prompt dict (keys may be str or int)."""
    if node_id in prompt:
        return prompt[node_id]

    sid = str(node_id)
    if sid in prompt:
        return prompt[sid]

    try:
        iid = int(node_id)
    except Exception:
        return None
    return prompt.get(iid)


def _get_link_source_id(link: Any) -> Optional[Any]:
    """A link usually looks like [source_node_id, source_output_index, ...]."""
    if isinstance(link, list) and len(link) >= 1:
        return link[0]
    return None


def _is_link(v: Any) -> bool:
    return isinstance(v, list) and len(v) >= 1


def _first_link_in_inputs(inputs: Dict[str, Any]) -> Optional[Any]:
    for v in inputs.values():
        if _is_link(v):
            return v
    return None


def _or_unknown(v: Any, unknown: str = "?") -> Any:
    return unknown if v is None else v


_FONT_CACHE: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}


def _load_font(font_path: str, size: int) -> Optional[ImageFont.FreeTypeFont]:
    key = (font_path, int(size))
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]

    if not os.path.exists(font_path):
        return None

    try:
        font = ImageFont.truetype(font_path, int(size))
        _FONT_CACHE[key] = font
        return font
    except Exception:
        return None


def _measure_text(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, text: str) -> float:
    if hasattr(draw, "textlength"):
        return float(draw.textlength(text, font=font))
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return float(right - left)


def _wrap_by_chars(
    draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, text: str, max_width: float
) -> List[str]:
    out: List[str] = []
    cur = ""
    for ch in text:
        cand = cur + ch
        if cur and _measure_text(draw, font, cand) > max_width:
            out.append(cur)
            cur = ch
        else:
            cur = cand
    if cur:
        out.append(cur)
    return out or [""]


def _wrap_text(
    draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, text: str, max_width: float
) -> List[str]:
    """Greedy, width-based wrap (supports \n, English spaces, and CJK)."""
    lines: List[str] = []

    for para in str(text).splitlines():
        if para == "":
            lines.append("")
            continue

        if " " not in para:
            lines.extend(_wrap_by_chars(draw, font, para, max_width))
            continue

        tokens = para.split(" ")
        cur = ""
        for tok in tokens:
            cand = tok if not cur else f"{cur} {tok}"
            if _measure_text(draw, font, cand) <= max_width:
                cur = cand
                continue

            if cur:
                lines.append(cur)
                cur = ""

            if _measure_text(draw, font, tok) <= max_width:
                cur = tok
            else:
                lines.extend(_wrap_by_chars(draw, font, tok, max_width))
                cur = ""

        if cur:
            lines.append(cur)

    return lines or [""]


def _truncate(s: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(s) <= max_chars:
        return s
    if max_chars <= 1:
        return s[:max_chars]
    return s[: max_chars - 1] + "…"


@dataclass
class LoraEntry:
    name: str
    strength_model: Optional[float] = None
    strength_clip: Optional[float] = None

    def format(self) -> str:
        if self.strength_model is None and self.strength_clip is None:
            return self.name

        parts = []
        if self.strength_model is not None:
            parts.append(f"m={self.strength_model:g}")
        if self.strength_clip is not None:
            parts.append(f"c={self.strength_clip:g}")
        return f"{self.name} ({', '.join(parts)})"


class LoRAParameterOverlay_AS:
    """
    Overlay checkpoint / LoRA / sampler params / prompts below the image.

    Backward compatible node type key: LoRAParameterOverlay
    Display name: Lora_tracker_AS
    """

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "AgaveSunset/AS"

    _MAX_PROMPT_CHARS = 2000
    _MAX_LINES = 140

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "custom_label": ("STRING", {"default": "", "multiline": False}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    def run(self, image, custom_label: Optional[str] = None, prompt=None, unique_id=None):
        prompt = prompt or {}
        label = (custom_label or "").strip()

        font_path = os.path.join(os.path.dirname(__file__), "font.ttf")
        font = _load_font(font_path, size=20)

        sampler_info, sampler_id = self.get_upstream_sampler(prompt, unique_id)

        loras: List[LoraEntry] = []
        ckpt_name = "Unknown Model"
        if sampler_id is not None:
            loras, ckpt_name = self.trace_models(prompt, sampler_id)

        info_lines = self.build_info_lines(label, ckpt_name, loras, sampler_info)

        batch_results = []
        for i in range(image.shape[0]):
            img_tensor = image[i]
            img_np = 255.0 * img_tensor.cpu().numpy()
            img_pil = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))

            processed = self.draw_overlay(img_pil, info_lines, font=font)

            out_np = np.array(processed).astype(np.float32) / 255.0
            out_tensor = torch.from_numpy(out_np)
            batch_results.append(out_tensor)

        return (torch.stack(batch_results),)

    # -----------------------------
    # Graph tracing: find sampler
    # -----------------------------
    def get_upstream_sampler(self, prompt: Dict[Any, Any], unique_id: Any):
        """Trace image->VAE Decode->(KSampler or Sampler*) node."""
        try:
            current_node = _get_prompt_node(prompt, unique_id)
            if not current_node:
                return None, None

            current_inputs = current_node.get("inputs", {})
            image_link = current_inputs.get("image")
            source_node_id = _get_link_source_id(image_link)
            if source_node_id is None:
                return None, None

            steps = 0
            while steps < 120 and source_node_id is not None:
                node_data = _get_prompt_node(prompt, source_node_id)
                if not node_data:
                    break

                class_type = node_data.get("class_type", "")
                inputs = node_data.get("inputs", {})

                # Common image decode path
                if "VAEDecode" in class_type:
                    source_node_id = _get_link_source_id(inputs.get("samples"))
                    steps += 1
                    continue

                # Classic KSampler family
                if "KSampler" in class_type:
                    return self.extract_sampler_info(prompt, source_node_id), source_node_id

                # Newer sampler pipeline (SamplerCustomAdvanced, SamplerCustom, etc.)
                if "Sampler" in class_type:
                    return self.extract_sampler_info(prompt, source_node_id), source_node_id

                # Reroute
                if "Reroute" in class_type:
                    vals = list(inputs.values())
                    source_node_id = _get_link_source_id(vals[0]) if vals else None
                    steps += 1
                    continue

                # Upscale pass-through
                if "Upscale" in class_type and "image" in inputs:
                    source_node_id = _get_link_source_id(inputs.get("image"))
                    steps += 1
                    continue

                break

            return None, None
        except Exception as e:
            print(f"[LoRA_Tracker_AS] trace sampler failed: {e}")
            return None, None

    # -----------------------------
    # Model/LoRA tracing
    # -----------------------------
    def _get_model_link_from_inputs(self, inputs: Dict[str, Any]) -> Optional[Any]:
        for k in ("model", "diffusion_model", "unet", "unet_model", "model_"):
            v = inputs.get(k)
            if _is_link(v):
                return v
        return None

    def _get_guider_node(self, prompt: Dict[Any, Any], sampler_inputs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        guider_link = sampler_inputs.get("guider")
        guider_id = _get_link_source_id(guider_link)
        if guider_id is None:
            return None
        return _get_prompt_node(prompt, guider_id)

    def trace_models(self, prompt: Dict[Any, Any], sampler_id: Any) -> Tuple[List[LoraEntry], str]:
        """Trace from sampler to model chain, collecting LoRA + checkpoint name."""
        loras_downstream: List[LoraEntry] = []
        ckpt_name = "Unknown"

        try:
            sampler_node = _get_prompt_node(prompt, sampler_id)
            if not sampler_node:
                return [], ckpt_name

            s_inputs = sampler_node.get("inputs", {})

            # 1) classic sampler has model/diffusion_model directly
            model_link = self._get_model_link_from_inputs(s_inputs)

            # 2) custom samplers often carry model inside a guider node
            if model_link is None:
                guider_node = self._get_guider_node(prompt, s_inputs)
                if guider_node:
                    g_inputs = guider_node.get("inputs", {})
                    model_link = self._get_model_link_from_inputs(g_inputs)

            current_id = _get_link_source_id(model_link) if model_link is not None else None

            steps = 0
            while current_id is not None and steps < 120:
                node = _get_prompt_node(prompt, current_id)
                if not node:
                    break

                class_type = node.get("class_type", "")
                inputs = node.get("inputs", {})

                # LoRA loader patterns
                if "lora_name" in inputs or "Lora" in class_type or "LoRA" in class_type:
                    lora_name = inputs.get("lora_name")
                    if lora_name:
                        name = _basename_no_ext(lora_name)

                        sm = inputs.get("strength_model")
                        sc = inputs.get("strength_clip")
                        if sm is None and "strength" in inputs:
                            sm = inputs.get("strength")

                        def _to_float(x):
                            try:
                                return float(x)
                            except Exception:
                                return None

                        loras_downstream.append(
                            LoraEntry(
                                name=name,
                                strength_model=_to_float(sm),
                                strength_clip=_to_float(sc),
                            )
                        )

                # Checkpoint-ish names (different loaders use different keys)
                for key in ("ckpt_name", "unet_name", "model_name", "diffusion_model", "checkpoint", "ckpt"):
                    if key in inputs and isinstance(inputs[key], str) and inputs[key]:
                        ckpt_name = _basename_no_ext(inputs[key])
                        break

                next_link = self._get_model_link_from_inputs(inputs)
                if next_link is None:
                    break

                current_id = _get_link_source_id(next_link)
                steps += 1

        except Exception as e:
            print(f"[LoRA_Tracker_AS] trace models failed: {e}")

        # reverse so it reads in application order
        loras = list(reversed(loras_downstream))
        return loras, ckpt_name

    # -----------------------------
    # Sampler info extraction
    # -----------------------------
    def _trace_widget_value(
        self,
        prompt: Dict[Any, Any],
        start_link: Any,
        wanted_keys: Iterable[str],
        max_hops: int = 20,
    ) -> Optional[Any]:
        """Follow a link upstream until we find a node with one of wanted_keys as a direct widget value."""
        node_id = _get_link_source_id(start_link)
        hops = 0
        while node_id is not None and hops < max_hops:
            node = _get_prompt_node(prompt, node_id)
            if not node:
                return None

            inputs = node.get("inputs", {})

            for k in wanted_keys:
                if k in inputs and not _is_link(inputs[k]):
                    return inputs[k]

            # Reroute passthrough
            class_type = node.get("class_type", "")
            if "Reroute" in class_type:
                vals = list(inputs.values())
                node_id = _get_link_source_id(vals[0]) if vals else None
                hops += 1
                continue

            # Heuristic: follow the first link-shaped input
            nxt = _first_link_in_inputs(inputs)
            node_id = _get_link_source_id(nxt) if nxt is not None else None
            hops += 1

        return None

    def _extract_custom_sampler_info(self, prompt: Dict[Any, Any], sampler_id: Any) -> Dict[str, Any]:
        node = _get_prompt_node(prompt, sampler_id)
        if not node:
            return {}

        inputs = node.get("inputs", {})

        # seed: usually comes from a noise node
        seed = None
        if "noise" in inputs:
            seed = self._trace_widget_value(prompt, inputs.get("noise"), ("seed", "noise_seed"))

        # sampler_name: usually from a sampler-select node
        sampler_name = None
        if "sampler" in inputs:
            sampler_name = self._trace_widget_value(prompt, inputs.get("sampler"), ("sampler_name", "name"))

        # steps/scheduler/denoise: often from a scheduler/sigmas node
        steps = None
        scheduler = None
        denoise = None
        if "sigmas" in inputs:
            steps = self._trace_widget_value(prompt, inputs.get("sigmas"), ("steps", "num_steps"))
            scheduler = self._trace_widget_value(prompt, inputs.get("sigmas"), ("scheduler", "schedule", "scheduler_name"))
            denoise = self._trace_widget_value(prompt, inputs.get("sigmas"), ("denoise",))

        # cfg + prompts: often live in guider node
        cfg = None
        pos_text = ""
        neg_text = ""
        guider_node = self._get_guider_node(prompt, inputs)
        if guider_node:
            g_inputs = guider_node.get("inputs", {})
            cfg = g_inputs.get("cfg", None)
            if _is_link(cfg):
                cfg = self._trace_widget_value(prompt, cfg, ("cfg",))

            # Try common keys for pos/neg conditioning
            pos_link = g_inputs.get("positive") or g_inputs.get("pos") or g_inputs.get("conditioning")
            neg_link = g_inputs.get("negative") or g_inputs.get("neg")

            if pos_link:
                pos_text = self.trace_text(prompt, pos_link)
            if neg_link:
                neg_text = self.trace_text(prompt, neg_link)

        return {
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": denoise,
            "pos_text": pos_text,
            "neg_text": neg_text,
        }

    def extract_sampler_info(self, prompt: Dict[Any, Any], sampler_id: Any) -> Optional[Dict[str, Any]]:
        node = _get_prompt_node(prompt, sampler_id)
        if not node:
            return None

        data = node.get("inputs", {})

        # Classic KSampler has these widgets directly.
        has_classic = any(k in data for k in ("steps", "cfg", "sampler_name", "scheduler", "positive", "negative"))
        if has_classic:
            return {
                "seed": data.get("seed") or data.get("noise_seed"),
                "steps": data.get("steps"),
                "cfg": data.get("cfg"),
                "sampler_name": data.get("sampler_name"),
                "scheduler": data.get("scheduler"),
                "denoise": data.get("denoise", 1.0),
                "pos_text": self.trace_text(prompt, data.get("positive")),
                "neg_text": self.trace_text(prompt, data.get("negative")),
            }

        # Otherwise treat it as a custom sampler pipeline.
        return self._extract_custom_sampler_info(prompt, sampler_id)

    # -----------------------------
    # Text tracing (prompts)
    # -----------------------------
    def trace_text(self, prompt: Dict[Any, Any], link: Any, depth: int = 0) -> str:
        if depth > 20:
            return ""
        node_id = _get_link_source_id(link)
        if node_id is None:
            return ""

        node = _get_prompt_node(prompt, node_id)
        if not node:
            return ""

        inputs = node.get("inputs", {})

        # Broad match: any node that carries a string text-like input
        for key in ("text", "text_g", "text_l", "prompt", "string", "value"):
            if key in inputs and isinstance(inputs[key], str):
                return inputs[key]

        class_type = node.get("class_type", "")

        # Reroute passthrough
        if "Reroute" in class_type:
            vals = list(inputs.values())
            if vals:
                return self.trace_text(prompt, vals[0], depth + 1)
            return ""

        # Conditioning chains: try common keys
        for key in (
            "positive",
            "negative",
            "conditioning",
            "conditioning_1",
            "conditioning_2",
            "clip_conditioning",
            "guidance",
            "cond",
        ):
            if key in inputs and _is_link(inputs[key]):
                found = self.trace_text(prompt, inputs[key], depth + 1)
                if found:
                    return found

        # Fallback: follow first link input
        nxt = _first_link_in_inputs(inputs)
        if nxt is not None:
            return self.trace_text(prompt, nxt, depth + 1)

        return ""

    # -----------------------------
    # UI lines + rendering
    # -----------------------------
    def build_info_lines(
        self,
        label: str,
        ckpt_name: str,
        loras: List[LoraEntry],
        sampler_info: Optional[Dict[str, Any]],
    ) -> List[str]:
        lines: List[str] = []

        if label:
            lines.append(label)

        lines.append(f"Checkpoint: {ckpt_name}")

        if loras:
            lines.append("LoRA: " + "; ".join([l.format() for l in loras]))
        else:
            lines.append("LoRA: None")

        if not sampler_info:
            lines.append("MetaData Not Found (Link incomplete)")
            return lines

        seed = _or_unknown(sampler_info.get("seed"))
        steps = _or_unknown(sampler_info.get("steps"))
        cfg = _or_unknown(sampler_info.get("cfg"))
        denoise = sampler_info.get("denoise")

        line_params = f"Seed: {seed}  |  Steps: {steps}  |  CFG: {cfg}"
        try:
            if denoise is not None and float(denoise) < 1.0:
                line_params += f"  |  Denoise: {denoise}"
        except Exception:
            pass
        lines.append(line_params)

        s_name = _or_unknown(sampler_info.get("sampler_name"))
        sched_name = _or_unknown(sampler_info.get("scheduler"))
        lines.append(f"Sampler: {s_name}  |  Scheduler: {sched_name}")

        p_text = _truncate((sampler_info.get("pos_text") or "").strip(), self._MAX_PROMPT_CHARS)
        n_text = _truncate((sampler_info.get("neg_text") or "").strip(), self._MAX_PROMPT_CHARS)

        if p_text:
            lines.append(f"Pos: {p_text}")
        if n_text:
            lines.append(f"Neg: {n_text}")

        return lines

    def draw_overlay(self, image: Image.Image, text_lines: List[str], font=None) -> Image.Image:
        w, h = image.size
        margin = 15

        tmp = Image.new("RGB", (w, h), (0, 0, 0))
        draw_tmp = ImageDraw.Draw(tmp)

        if font is None:
            font = ImageFont.load_default()

        try:
            left, top, right, bottom = font.getbbox("Ag")
            line_h = int((bottom - top) * 1.25)
        except Exception:
            line_h = 18

        max_text_width = max(10, w - margin * 2)

        final_lines: List[str] = []
        for raw in text_lines:
            final_lines.extend(_wrap_text(draw_tmp, font, str(raw), max_text_width))

        if len(final_lines) > self._MAX_LINES:
            extra = len(final_lines) - self._MAX_LINES
            final_lines = final_lines[: self._MAX_LINES] + [f"…(truncated {extra} lines)"]

        footer_h = len(final_lines) * line_h + margin * 2
        new_h = h + footer_h

        new_img = Image.new("RGB", (w, new_h), (0, 0, 0))
        new_img.paste(image, (0, 0))

        draw = ImageDraw.Draw(new_img)
        y = h + margin
        for line in final_lines:
            draw.text((margin, y), line, fill=(255, 255, 255), font=font)
            y += line_h

        return new_img


NODE_CLASS_MAPPINGS = {
    "LoRAParameterOverlay": LoRAParameterOverlay_AS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoRAParameterOverlay": "Lora_tracker_AS",
}
