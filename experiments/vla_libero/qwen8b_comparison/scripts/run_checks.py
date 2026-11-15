"""Sequential BF16 offline comparison; no simulator or robot commands."""

from pathlib import Path
import argparse, json, sys, time, types, hashlib, traceback

R = Path(__file__).resolve().parents[1]
ROOT = R.parents[2]
VLA = R.parent
sys.path[:0] = [str(VLA / "scripts"), str(ROOT / "perception/qwen_scene")]
import torch
from qwen_planner import Planner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["4b", "8b"], required=True)
    a = ap.parse_args()
    out = R / "results" / a.model
    out.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(20261112)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    metadata = {
        "name": a.model,
        "precision": "BF16",
        "device_map": "cuda:0 only",
        "attention": "sdpa",
        "planner_source_sha256": hashlib.sha256(
            (VLA / "scripts/qwen_planner.py").read_bytes()
        ).hexdigest(),
    }
    try:
        if a.model == "4b":
            from engine import SceneEngine

            engine = SceneEngine()
        else:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

            path = ROOT / "models/qwen3_vl_8b_instruct"
            processor = AutoProcessor.from_pretrained(path, local_files_only=True)
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                path,
                local_files_only=True,
                dtype=torch.bfloat16,
                device_map={"": "cuda:0"},
                attn_implementation="sdpa",
            ).eval()
            engine = types.SimpleNamespace(
                torch=torch, processor=processor, model=model
            )
        torch.cuda.synchronize()
        metadata["load_seconds"] = time.monotonic() - start
        metadata["parameters"] = sum(p.numel() for p in engine.model.parameters())
        metadata["parameter_dtypes"] = sorted(
            {str(p.dtype) for p in engine.model.parameters()}
        )
        metadata["all_parameters_on_cuda"] = all(
            p.device.type == "cuda" for p in engine.model.parameters()
        )
        metadata["image_processor_size"] = getattr(
            engine.processor.image_processor, "size", None
        )
        planner = Planner.__new__(Planner)
        planner.engine = engine
        cases = json.loads((R / "cases.json").read_text())["cases"]
        rows = []
        warm = planner.generate(
            [str(ROOT / x) for x in cases[0]["images"]],
            'Return only JSON {"ready":true}.',
            max_new_tokens=16,
        )
        metadata["warmup_seconds"] = warm["seconds"]
        for i, c in enumerate(cases):
            torch.manual_seed(20261112 + i)
            torch.cuda.reset_peak_memory_stats()
            t = time.monotonic()
            response = planner.check([str(ROOT / x) for x in c["images"]], c["subgoal"])
            elapsed = time.monotonic() - t
            row = {
                "case": c["id"],
                "expected_done": c["expected_done"],
                "historical_4b_done": c["historical_4b_done"],
                "response": response,
                "wall_seconds": elapsed,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                "matches_predicate": bool(
                    response.get("ok") and response.get("done") == c["expected_done"]
                ),
            }
            rows.append(row)
            (out / (c["id"] + ".json")).write_text(
                json.dumps(row, indent=2, ensure_ascii=False)
            )
            print(
                c["id"],
                response.get("done"),
                "expected",
                c["expected_done"],
                "wall",
                round(elapsed, 3),
                flush=True,
            )
        metadata.update(
            status="complete",
            n=len(rows),
            matched=sum(r["matches_predicate"] for r in rows),
            schema_valid=sum(bool(r["response"].get("ok")) for r in rows),
            peak_allocated_mib=max(r["peak_allocated_mib"] for r in rows),
            peak_reserved_mib=max(r["peak_reserved_mib"] for r in rows),
            median_wall_seconds=__import__("statistics").median(
                r["wall_seconds"] for r in rows
            ),
            rows=rows,
        )
    except Exception:
        metadata.update(status="error", error=traceback.format_exc())
        print(metadata["error"], flush=True)
    (out / "summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )
    print(
        json.dumps({k: v for k, v in metadata.items() if k != "rows"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
