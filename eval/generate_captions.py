#!/usr/bin/env python3
"""
Standalone caption generator for scan-to-map outputs.

Reads the crops manifest, loads the best crop image per component,
generates captions using a VLM, and writes component_captions.json.

Supports three backends:
  - "vllm"         : vLLM + Qwen2.5-VL-7B  (fastest, needs GPU)
  - "transformers"  : HuggingFace pipeline  (needs GPU)
  - "blip"          : BLIP-2 via transformers (smaller, can run on CPU)

Usage:
  conda activate scan-to-map-eval
  python generate_captions.py \
      --outputs-dir ../outputs/scannet_scene0000_00 \
      --backend vllm \
      --n-images 1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


CAPTION_PROMPT = (
    "These images show different views of the same object or region in a 3D scene. "
    "Analyze all the images together and provide a concise, descriptive caption "
    "that captures what this object or region is. Focus on:\n"
    "1. What the main object/region is\n"
    "2. Its key visual characteristics (color, shape, texture)\n"
    "3. Any notable features or context\n\n"
    "Keep the caption clear and factual, suitable for 3D semantic search."
)


def load_manifest(outputs_dir: Path) -> Dict[str, Any]:
    manifest_path = outputs_dir / "crops" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open("r") as f:
        return json.load(f)


def get_top_images(
    component_id: str, manifest: Dict[str, Any], n: int = 1
) -> List[Dict[str, Any]]:
    if component_id not in manifest:
        return []
    crops = manifest[component_id].get("crops", [])
    return sorted(crops, key=lambda c: c.get("visible_points", 0), reverse=True)[:n]


def resolve_crop_paths(
    component_id: str, top_images: List[Dict[str, Any]], outputs_dir: Path
) -> List[Path]:
    crops_dir = outputs_dir / "crops"
    paths = []
    for info in top_images:
        p = crops_dir / f"component_{component_id}" / info["crop_filename"]
        if p.exists():
            paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Backend: vLLM  (Qwen2.5-VL-7B-Instruct)
# ---------------------------------------------------------------------------
def caption_vllm(
    components: List[dict],
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    device: int = 0,
    max_tokens: int = 300,
) -> List[str]:
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(model_name)
    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        max_model_len=4096,
        limit_mm_per_prompt={"image": 10},
        gpu_memory_utilization=0.9,
    )
    sampling = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    vllm_inputs = []
    for comp in components:
        pil_images = [Image.open(p).convert("RGB") for p in comp["paths"]]
        placeholders = [{"type": "image", "image": img} for img in pil_images]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [*placeholders, {"type": "text", "text": CAPTION_PROMPT}]},
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        vllm_inputs.append({"prompt": prompt, "multi_modal_data": {"image": pil_images}})

    outputs = llm.generate(vllm_inputs, sampling_params=sampling)
    captions = []
    for out in outputs:
        text = out.outputs[0].text.strip().strip('"').strip("\\\"")
        text = text.replace("addCriterion:", "").replace("addCriterion", "").strip()
        captions.append(text)
    return captions


# ---------------------------------------------------------------------------
# Backend: HuggingFace Transformers  (Qwen2.5-VL-7B-Instruct)
# ---------------------------------------------------------------------------
def caption_transformers(
    components: List[dict],
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    device: int = 0,
    max_tokens: int = 300,
) -> List[str]:
    import torch
    from PIL import Image
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    print(f"Loading model {model_name} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=f"cuda:{device}"
    )
    processor = AutoProcessor.from_pretrained(model_name)

    captions = []
    for comp in components:
        pil_images = [Image.open(p).convert("RGB") for p in comp["paths"]]
        placeholders = [{"type": "image", "image": img} for img in pil_images]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [*placeholders, {"type": "text", "text": CAPTION_PROMPT}]},
        ]
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.0, do_sample=False)

        out_ids = generated[:, inputs["input_ids"].shape[1]:]
        caption = processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
        captions.append(caption)
        print(f"  Component {comp['id']}: {caption[:80]}...")

    return captions


# ---------------------------------------------------------------------------
# Backend: BLIP-2  (lighter, can run on CPU)
# ---------------------------------------------------------------------------
def caption_blip(
    components: List[dict],
    model_name: str = "Salesforce/blip2-opt-2.7b",
    device: int = 0,
    max_tokens: int = 150,
) -> List[str]:
    import torch
    from PIL import Image
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    dev = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev.startswith("cuda") else torch.float32
    print(f"Loading BLIP-2 ({model_name}) on {dev} ...")
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype).to(dev)

    prompt = "Question: What is this object or region? Describe it concisely. Answer:"

    captions = []
    for comp in components:
        img = Image.open(comp["paths"][0]).convert("RGB")
        inputs = processor(images=img, text=prompt, return_tensors="pt").to(dev, dtype)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=max_tokens)
        caption = processor.decode(generated[0], skip_special_tokens=True).strip()
        captions.append(caption)
        print(f"  Component {comp['id']}: {caption[:80]}...")

    return captions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate component_captions.json for scan-to-map outputs"
    )
    parser.add_argument(
        "--outputs-dir", type=Path, required=True,
        help="Path to the scene outputs directory (e.g. ../outputs/scannet_scene0000_00)",
    )
    parser.add_argument(
        "--backend", type=str, choices=["vllm", "transformers", "blip"], default="vllm",
        help="Captioning backend (default: vllm)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override the model name for the chosen backend",
    )
    parser.add_argument(
        "--n-images", type=int, default=1,
        help="Number of top crop images to use per component (default: 1)",
    )
    parser.add_argument(
        "--device", type=int, default=0,
        help="GPU device index (default: 0)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=300,
        help="Maximum tokens to generate per caption (default: 300)",
    )

    args = parser.parse_args()
    outputs_dir = args.outputs_dir.resolve()
    if not outputs_dir.is_dir():
        raise FileNotFoundError(f"Outputs directory not found: {outputs_dir}")

    manifest = load_manifest(outputs_dir)
    component_ids = sorted(manifest.keys(), key=int)
    print(f"Found {len(component_ids)} components in manifest")

    components = []
    for cid in component_ids:
        top = get_top_images(cid, manifest, n=args.n_images)
        paths = resolve_crop_paths(cid, top, outputs_dir)
        if not paths:
            print(f"  WARNING: no valid crop images for component {cid}, skipping")
            continue
        components.append({
            "id": cid,
            "paths": paths,
            "crop_filenames": [info["crop_filename"] for info in top if (outputs_dir / "crops" / f"component_{cid}" / info["crop_filename"]).exists()],
        })

    print(f"Captioning {len(components)} components with backend={args.backend}")

    default_models = {
        "vllm": "Qwen/Qwen2.5-VL-7B-Instruct",
        "transformers": "Qwen/Qwen2.5-VL-7B-Instruct",
        "blip": "Salesforce/blip2-opt-2.7b",
    }
    model_name = args.model or default_models[args.backend]

    start = time.time()
    if args.backend == "vllm":
        captions = caption_vllm(components, model_name=model_name, device=args.device, max_tokens=args.max_tokens)
    elif args.backend == "transformers":
        captions = caption_transformers(components, model_name=model_name, device=args.device, max_tokens=args.max_tokens)
    elif args.backend == "blip":
        captions = caption_blip(components, model_name=model_name, device=args.device, max_tokens=args.max_tokens)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    elapsed = time.time() - start

    results = []
    for comp, caption in zip(components, captions):
        results.append({
            "component_id": int(comp["id"]),
            "caption": caption,
            "num_images_used": len(comp["paths"]),
            "crop_filenames": comp["crop_filenames"],
        })

    output_path = outputs_dir / "component_captions.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} captions to {output_path}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/len(results):.1f}s per component)")
    for r in results:
        print(f"  Component {r['component_id']}: {r['caption'][:100]}{'...' if len(r['caption'])>100 else ''}")


if __name__ == "__main__":
    main()
