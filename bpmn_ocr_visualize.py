"""
OCR visualization for BPMN diagrams.

Outputs:
- *_ocr_tesseract.png: Tesseract OCR boxes + text
- *_ocr_cv.png: Simple CV text-region detector boxes (no text)

Notes:
- CV detector is a lightweight heuristic (binarize + dilation + components).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

try:
    import pytesseract
    from pytesseract import Output as TesseractOutput
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

 


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    def clip(self, w: int, h: int) -> "Box":
        return Box(
            x1=max(0, min(self.x1, w - 1)),
            y1=max(0, min(self.y1, h - 1)),
            x2=max(0, min(self.x2, w - 1)),
            y2=max(0, min(self.y2, h - 1)),
        )


def _load_font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except Exception:
        return ImageFont.load_default()


def _draw_bold_text(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, font, fill):
    x, y = xy
    draw.text((x, y), text, fill=fill, font=font)
    draw.text((x + 1, y), text, fill=fill, font=font)
    draw.text((x, y + 1), text, fill=fill, font=font)
    draw.text((x + 1, y + 1), text, fill=fill, font=font)


def _wrap_text_to_width(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if draw.textbbox((0, 0), trial, font=font)[2] <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    # If a single word is too long, fall back to character splitting
    fixed: List[str] = []
    for line in lines:
        if draw.textbbox((0, 0), line, font=font)[2] <= max_width:
            fixed.append(line)
            continue
        chunk = ""
        for ch in line:
            trial = chunk + ch
            if draw.textbbox((0, 0), trial, font=font)[2] <= max_width:
                chunk = trial
            else:
                if chunk:
                    fixed.append(chunk)
                chunk = ch
        if chunk:
            fixed.append(chunk)
    return fixed


def _fit_text_in_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: Box,
    max_font_size: int = 14,
    min_font_size: int = 8,
    padding: int = 2,
) -> Tuple[List[str], ImageFont.ImageFont]:
    max_width = max(0, (box.x2 - box.x1) - padding * 2)
    max_height = max(0, (box.y2 - box.y1) - padding * 2)
    if max_width <= 0 or max_height <= 0:
        return [], _load_font(min_font_size)
    for size in range(max_font_size, min_font_size - 1, -1):
        font = _load_font(size)
        lines = _wrap_text_to_width(draw, text, font, max_width)
        if not lines:
            continue
        line_h = draw.textbbox((0, 0), "Ag", font=font)[3]
        total_h = line_h * len(lines)
        if total_h <= max_height:
            return lines, font
    # As a fallback, return a single truncated line with smallest font.
    font = _load_font(min_font_size)
    ellipsis = "…"
    line = text
    while line and draw.textbbox((0, 0), line + ellipsis, font=font)[2] > max_width:
        line = line[:-1]
    if line:
        return [line + ellipsis], font
    return [], font


def _draw_text_in_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: Box,
    fill=(0, 0, 0),
    padding: int = 2,
) -> None:
    lines, font = _fit_text_in_box(draw, text, box, padding=padding)
    if not lines:
        return
    x = box.x1 + padding
    y = box.y1 + padding
    line_h = draw.textbbox((0, 0), "Ag", font=font)[3]
    for i, line in enumerate(lines):
        _draw_bold_text(draw, (x, y + i * line_h), line, font, fill=fill)


def _iter_images(path: Path) -> Iterable[Path]:
    if path.is_dir():
        for img in sorted(path.glob("*.png")):
            if "_ocr_" in img.stem:
                continue
            yield img
    else:
        yield path


def _tesseract_boxes(image: Image.Image, lang: str, psm: str) -> List[Tuple[Box, str, int]]:
    if not HAS_TESSERACT:
        return []
    data = pytesseract.image_to_data(
        image,
        lang=lang,
        config=f"--psm {psm}",
        output_type=TesseractOutput.DICT,
    )
    results = []
    for i in range(len(data.get("text", []))):
        text = (data["text"][i] or "").strip()
        conf = int(float(data["conf"][i]))
        if not text or conf < 0:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        results.append((Box(x, y, x + w, y + h), text, conf))
    return results


def _draw_tesseract(image: Image.Image, boxes: List[Tuple[Box, str, int]]) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    for box, text, conf in boxes:
        draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=(0, 0, 0), width=2)
        label = text
        _draw_text_in_box(draw, label, box, fill=(0, 0, 0), padding=2)
    return out


def _tesseract_boxes_in_box(
    image: Image.Image,
    region: Box,
    lang: str,
    psm: str
) -> List[Tuple[Box, str, int]]:
    crop = image.crop((region.x1, region.y1, region.x2, region.y2))
    boxes = _tesseract_boxes(crop, lang=lang, psm=psm)
    results = []
    for b, text, conf in boxes:
        results.append((Box(b.x1 + region.x1, b.y1 + region.y1,
                            b.x2 + region.x1, b.y2 + region.y1), text, conf))
    return results


def _otsu_threshold(gray: np.ndarray) -> int:
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b = 0
    w_b = 0
    var_max = 0
    threshold = 128
    for i in range(256):
        w_b += hist[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += i * hist[i]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > var_max:
            var_max = var_between
            threshold = i
    return threshold


def _connected_components(binary: np.ndarray) -> List[Box]:
    h, w = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    boxes: List[Box] = []
    for y in range(h):
        for x in range(w):
            if binary[y, x] == 0 or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            min_x = max_x = x
            min_y = max_y = y
            while stack:
                cy, cx = stack.pop()
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            boxes.append(Box(min_x, min_y, max_x, max_y))
    return boxes


def _cv_text_regions(image: Image.Image) -> List[Box]:
    gray = np.array(image.convert("L"))
    thr = _otsu_threshold(gray)
    binary = (gray < thr).astype(np.uint8)  # text as 1

    # Downscale for speed
    scale = 2
    binary_small = binary[::scale, ::scale]
    img_small = Image.fromarray((binary_small * 255).astype(np.uint8))
    # Dilation to merge characters into words/lines
    img_small = img_small.filter(ImageFilter.MaxFilter(size=5))
    img_small = img_small.filter(ImageFilter.MaxFilter(size=5))
    binary_small = (np.array(img_small) > 0).astype(np.uint8)

    boxes = _connected_components(binary_small)
    h_small, w_small = binary_small.shape
    h, w = gray.shape

    # Scale boxes back and filter by size/aspect
    results: List[Box] = []
    for b in boxes:
        x1 = int(b.x1 * scale)
        y1 = int(b.y1 * scale)
        x2 = int(b.x2 * scale)
        y2 = int(b.y2 * scale)
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        area = bw * bh
        if area < 200 or area > (w * h * 0.2):
            continue
        if bw < 10 or bh < 6:
            continue
        results.append(Box(x1, y1, x2, y2).clip(w, h))
    return results


def _cv_text_regions_in_box(image: Image.Image, region: Box) -> List[Box]:
    crop = image.crop((region.x1, region.y1, region.x2, region.y2))
    boxes = _cv_text_regions(crop)
    results = []
    for b in boxes:
        results.append(Box(b.x1 + region.x1, b.y1 + region.y1,
                           b.x2 + region.x1, b.y2 + region.y1))
    return results


def _draw_cv(image: Image.Image, boxes: List[Box]) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    for box in boxes:
        draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=(0, 0, 0), width=2)
    return out


def _object_boxes_from_detector(image: Image.Image, model_object, device, threshold: float) -> List[Box]:
    from bpmn_detection_demo import detect_objects

    detections = detect_objects(image, model_object, device, threshold=threshold)
    boxes: List[Box] = []
    # Use only non-container elements (skip pool/lane)
    for det in detections:
        if det.get("label") in {"pool", "lane"}:
            continue
        x1, y1, x2, y2 = det["box"]
        boxes.append(Box(int(x1), int(y1), int(x2), int(y2)))
    return boxes


def process_image(path: Path, lang: str, psm: str, threshold: float,
                  model_object, device,
                  output_dir: Optional[Path] = None) -> None:
    image = Image.open(path).convert("RGB")
    obj_boxes = _object_boxes_from_detector(image, model_object, device, threshold=threshold)

    tesseract_boxes: List[Tuple[Box, str, int]] = []
    cv_boxes: List[Box] = []
    for region in obj_boxes:
        tesseract_boxes.extend(_tesseract_boxes_in_box(image, region, lang=lang, psm=psm))
        cv_boxes.extend(_cv_text_regions_in_box(image, region))

    tesseract_vis = _draw_tesseract(image, tesseract_boxes) if tesseract_boxes else None
    cv_vis = _draw_cv(image, cv_boxes)

    out_base = (output_dir or path.parent) / path.stem
    out_tess = out_base.with_name(f"{out_base.name}_ocr_tesseract.png")
    out_cv = out_base.with_name(f"{out_base.name}_ocr_cv.png")
    if tesseract_vis is not None:
        tesseract_vis.save(out_tess)
    cv_vis.save(out_cv)
    saved = [out_cv.name]
    if tesseract_vis is not None:
        saved.insert(0, out_tess.name)
    print(f"✓ Saved: {', '.join(saved)}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="OCR visualization (tesseract + CV baseline)")
    parser.add_argument("input", nargs="?", default="bpmn_diagrams/png", help="PNG file or directory")
    parser.add_argument("--lang", default="eng+rus", help="Tesseract language (e.g., eng, rus, eng+rus)")
    parser.add_argument("--psm", default="6", help="Tesseract PSM mode")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold for object boxes")
    parser.add_argument("--output-dir", default=None, help="Directory to save OCR визуализации")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    from bpmn_detection_demo import load_models
    model_object, _, device = load_models()

    for img_path in _iter_images(input_path):
        process_image(img_path, args.lang, args.psm, args.threshold,
                      model_object, device, output_dir=output_dir)


if __name__ == "__main__":
    main()
