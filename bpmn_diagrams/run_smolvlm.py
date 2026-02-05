#!/usr/bin/env python3
"""
Process BPMN diagram images with SmolVLM2-2.2B-Instruct.
Converts each diagram to a text table description using the vision-language model.

Usage:
    python run_smolvlm.py [--input-dir DIR] [--output-dir DIR] [--prompt TEXT]
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("smolvlm_run.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)


def get_mps_memory_mb() -> float:
    """Return current MPS GPU memory allocated in MB."""
    if torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / (1024 * 1024)
    return 0.0


def log_gpu_memory(label: str = ""):
    mem = get_mps_memory_mb()
    log.info(f"[GPU] {label} — MPS allocated: {mem:.1f} MB")


def load_model(model_id: str, device: torch.device):
    log.info(f"Loading model: {model_id}")
    log_gpu_memory("before model load")

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()

    log_gpu_memory("after model load")
    log.info("Model loaded successfully")
    return processor, model


def process_image(
    image_path: Path,
    processor,
    model,
    device: torch.device,
    prompt: str,
) -> str:
    log.info(f"Processing: {image_path.name}")
    log_gpu_memory("before inference")
    t0 = time.time()

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    # Cast pixel values to float16 to match model dtype, then move to device
    for k, v in inputs.items():
        if v.dtype == torch.float32:
            inputs[k] = v.to(dtype=torch.float16, device=device)
        else:
            inputs[k] = v.to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024)

    # Decode only new tokens
    input_len = inputs["input_ids"].shape[1]
    output_text = processor.batch_decode(
        generated_ids[:, input_len:], skip_special_tokens=True
    )[0]

    elapsed = time.time() - t0
    log_gpu_memory("after inference")
    log.info(f"Done: {image_path.name} ({elapsed:.1f}s)")
    return output_text


def main():
    parser = argparse.ArgumentParser(description="Run SmolVLM2 on BPMN diagram images")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).parent / "png",
        help="Directory with PNG images (default: ./png)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "results",
        help="Directory for output text files (default: ./results)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Convert this diagram to a text table. Describe all elements, connections, and flow.",
        help="Prompt for the model",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        help="HuggingFace model ID",
    )
    args = parser.parse_args()

    # Validate input
    if not args.input_dir.exists():
        log.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)

    images = sorted(args.input_dir.glob("*.png"))
    if not images:
        log.error(f"No PNG files found in {args.input_dir}")
        sys.exit(1)

    log.info(f"Found {len(images)} images in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("Using Apple MPS GPU")
    else:
        device = torch.device("cpu")
        log.warning("MPS not available, falling back to CPU")

    # Load model
    processor, model = load_model(args.model_id, device)

    # Process images
    total_t0 = time.time()
    for i, img_path in enumerate(images, 1):
        log.info(f"=== Image {i}/{len(images)} ===")
        try:
            result = process_image(img_path, processor, model, device, args.prompt)
            out_file = args.output_dir / f"{img_path.stem}.txt"
            out_file.write_text(result, encoding="utf-8")
            log.info(f"Saved: {out_file}")
            print(f"\n--- {img_path.name} ---")
            print(result[:500])
            if len(result) > 500:
                print(f"... ({len(result)} chars total)")
            print()
        except Exception:
            log.exception(f"Failed to process {img_path.name}")

    total_elapsed = time.time() - total_t0
    log.info(f"All done. {len(images)} images processed in {total_elapsed:.1f}s")
    log_gpu_memory("final")


if __name__ == "__main__":
    main()
