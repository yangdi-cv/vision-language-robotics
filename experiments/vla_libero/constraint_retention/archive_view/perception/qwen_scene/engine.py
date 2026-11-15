"""Local image understanding. This module has no robot control interface."""

import json
import hashlib
import importlib.metadata
import time
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
PROMPT = """你是桌面机械臂的视觉观察助手，只描述这张图中可见证据。
不要因为任务背景就假定物体存在；不确定写 null 或 unknown。不能从单帧确认抓取牢固、任务完成或三维坐标。
识别图中的物体、夹爪、桌面和目标可见性。目标是深蓝色塑胶壳，但它可能不在画面内。
只输出一个 JSON 对象，键如下：
scene_description: 中文字符串；visible_objects: 中文字符串数组；
target_visible: true/false/null；gripper_state: open/closed/unknown；
target_relation: 中文字符串或 null，仅描述图像中的相对位置，目标不可见时为 null；
grasp_evidence: 中文字符串，说明是否看见夹爪接触物体，不猜测稳定性；
uncertainties: 中文字符串数组。不要输出动作或运动指令。"""


def parse_result(raw):
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    required = {
        "scene_description",
        "visible_objects",
        "target_visible",
        "gripper_state",
        "target_relation",
        "grasp_evidence",
        "uncertainties",
    }
    if set(value) != required:
        raise ValueError("Unexpected or missing result fields")
    for key in ("scene_description", "grasp_evidence"):
        if not isinstance(value[key], str):
            raise ValueError(f"Invalid {key}")
    if value["target_relation"] is not None and not isinstance(
        value["target_relation"], str
    ):
        raise ValueError("Invalid target_relation")
    for key in ("visible_objects", "uncertainties"):
        if not isinstance(value[key], list) or not all(
            isinstance(x, str) for x in value[key]
        ):
            raise ValueError(f"Invalid {key}")
    if (
        value["target_visible"] is not None
        and type(value["target_visible"]) is not bool
    ):
        raise ValueError("Invalid target_visible")
    if value["gripper_state"] not in ("open", "closed", "unknown"):
        raise ValueError("Invalid gripper_state")
    return value


def prepare_image(image, rotation=0, pixels_already_rotated=False):
    """rotation is clockwise from sensor pixels; live producers may apply it first."""
    rotation = 0 if rotation is None else rotation
    if rotation not in (0, 90, 180, 270):
        raise ValueError("rotation must be 0/90/180/270 clockwise degrees")
    image = image.convert("RGB")
    if rotation and not pixels_already_rotated:
        image = image.rotate(-rotation, expand=True)
    return image, rotation


class SceneEngine:
    def __init__(self):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; check NVIDIA/PyTorch environment")
        self.torch = torch
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID, revision=REVISION, local_files_only=True
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            revision=REVISION,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            attn_implementation="sdpa",
        ).eval()
        self.load_seconds = time.perf_counter() - started

    def analyze(
        self,
        image,
        output_root,
        rotation=0,
        source="camera",
        language="zh",
        max_new_tokens=512,
        *,
        pixels_already_rotated=False,
    ):
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        directory = output_root / f"scene_{time.time_ns()}"
        directory.mkdir()
        image, rotation = prepare_image(image, rotation, pixels_already_rotated)
        image.save(directory / "input.png")
        prompt = PROMPT + (
            "\nAll descriptive strings and string arrays MUST be in English. Keep JSON keys and enum values unchanged."
            if language == "en"
            else ""
        )
        generated = self.generate(image, prompt, max_new_tokens=max_new_tokens)
        raw = generated["raw"]
        result = {
            "model": MODEL_ID,
            "revision": REVISION,
            "source": source,
            "language": language,
            "rotation_clockwise_degrees": rotation or 0,
            "rotation_applied_here": bool(rotation and not pixels_already_rotated),
            "image_size": list(image.size),
            "inference_seconds": generated["seconds"],
            "model_load_seconds": self.load_seconds,
            "output_tokens": generated["tokens"],
            "token_limit": max_new_tokens,
            "inference": generated["inference"],
            "raw_output": raw,
            "schema_valid": False,
            "scene": None,
        }
        try:
            result["scene"] = parse_result(raw)
            result["schema_valid"] = True
        except (ValueError, TypeError) as exc:
            result["validation_error"] = str(exc)
        (directory / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False)
        )
        (directory / "prompt.txt").write_text(prompt)
        return result, directory

    def generate(self, images, prompt, max_new_tokens=256):
        """Shared generation path for camera observation and LIBERO planning/checking."""
        images = images if isinstance(images, list) else [images]
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": image} for image in images],
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        metadata = {
            "model": MODEL_ID,
            "revision": REVISION,
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "engine_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "generation": {"max_new_tokens": max_new_tokens, "do_sample": False},
            "images": [
                {
                    "size": list(im.size),
                    "mode": im.mode,
                    "pixel_sha256": hashlib.sha256(im.tobytes()).hexdigest(),
                }
                for im in images
            ],
            "libraries": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers")
            },
        }
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda:0")
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        self.torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        raw = self.processor.decode(
            outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )
        return {
            "raw": raw,
            "seconds": elapsed,
            "inference": metadata,
            "device": str(next(self.model.parameters()).device),
            "allocated_mib": self.torch.cuda.memory_allocated() / 2**20,
            "reserved_mib": self.torch.cuda.memory_reserved() / 2**20,
            "tokens": int(outputs.shape[-1] - inputs["input_ids"].shape[-1]),
        }
