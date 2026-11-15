"""Fetch pinned base into the normal HF cache for the local-files-only inference code."""
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-VL-4B-Instruct',revision='ebb281ec70b05090aa6165b016eac8ec08e71b17',allow_patterns=['*.json','*.safetensors','*.txt','*.jinja'])
