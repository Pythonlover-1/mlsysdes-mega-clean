"""
BPMN Detection Demo
Uses ELCA-SA/BPMN_Detection model to detect BPMN elements in diagrams

This demo uses two separate models:
- Faster R-CNN for BPMN objects (tasks, gateways, events, etc.)
- Keypoint R-CNN for arrows/flows with keypoints
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, keypointrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor
from torchvision.transforms import functional as F
from huggingface_hub import hf_hub_download

# BPMN object classes (detected by model_object.pth)
OBJECT_DICT = {
    0: 'background',
    1: 'task',
    2: 'exclusiveGateway',
    3: 'event',
    4: 'parallelGateway',
    5: 'messageEvent',
    6: 'pool',
    7: 'lane',
    8: 'dataObject',
    9: 'dataStore',
    10: 'subProcess',
    11: 'eventBasedGateway',
    12: 'timerEvent',
}

# Arrow/flow classes (detected by model_arrow.pth)
ARROW_DICT = {
    0: 'background',
    1: 'sequenceFlow',
    2: 'dataAssociation',
    3: 'messageFlow',
}

# Color palette for different element types
COLORS = {
    'background': '#F5F5F5',
    'task': '#81C784', 'subProcess': '#66BB6A',
    'exclusiveGateway': '#FFB74D', 'parallelGateway': '#FFA726', 'eventBasedGateway': '#FF9800',
    'event': '#E57373', 'messageEvent': '#EF5350', 'timerEvent': '#F44336',
    'pool': '#E3F2FD', 'lane': '#BBDEFB',
    'dataObject': '#FFE082', 'dataStore': '#FFD54F',
    'sequenceFlow': '#64B5F6', 'dataAssociation': '#42A5F5', 'messageFlow': '#2196F3'
}


def get_faster_rcnn_model(num_classes):
    """
    Create Faster R-CNN model for object detection
    Based on ResNet-50 with FPN backbone
    """
    # Important: explicitly disable backbone weights download.
    # Even with weights=None, torchvision may still try to download ResNet-50 backbone
    # weights from download.pytorch.org (weights_backbone defaults to pretrained).
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_arrow_model(num_classes, num_keypoints=2):
    """
    Create Keypoint R-CNN model for arrow detection with keypoints
    Based on ResNet-50 with FPN backbone
    """
    # Important: explicitly disable backbone weights download (see note above).
    model = keypointrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(512, num_keypoints)
    return model


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def load_models():
    """
    Load BPMN detection models from Hugging Face
    Downloads and initializes both object and arrow detection models
    """
    print("Loading BPMN Detection models from Hugging Face...")
    print("  This may take a few minutes on first run (downloading ~400MB)")

    # Download model weights from HuggingFace Hub
    print("  → Downloading model_object.pth (166 MB)...")
    model_object_path = hf_hub_download(
        repo_id="ELCA-SA/BPMN_Detection",
        filename="model_object.pth"
    )

    print("  → Downloading model_arrow.pth (237 MB)...")
    model_arrow_path = hf_hub_download(
        repo_id="ELCA-SA/BPMN_Detection",
        filename="model_arrow.pth"
    )

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  → Using device: {device}")

    # Create model architectures
    print("  → Initializing model architectures...")
    model_object = get_faster_rcnn_model(len(OBJECT_DICT))
    model_arrow = get_arrow_model(len(ARROW_DICT), num_keypoints=2)

    # Load weights
    print("  → Loading weights...")
    model_object.load_state_dict(torch.load(model_object_path, map_location=device))
    model_arrow.load_state_dict(torch.load(model_arrow_path, map_location=device))

    # Set to evaluation mode
    model_object.eval()
    model_arrow.eval()

    # Move to device
    model_object.to(device)
    model_arrow.to(device)

    print(f"✓ Models loaded successfully")
    print(f"  - Object model: {len(OBJECT_DICT)} classes")
    print(f"  - Arrow model: {len(ARROW_DICT)} classes (with keypoints)")

    return model_object, model_arrow, device


def detect_objects(image: Image.Image, model, device, threshold: float = 0.5) -> List[Dict]:
    """
    Detect BPMN objects (tasks, gateways, events, etc.) in an image

    Args:
        image: PIL Image
        model: Object detection model
        device: torch device
        threshold: Confidence threshold (0-1)

    Returns:
        List of detections with boxes, labels, and scores
    """
    # Convert to tensor
    img_tensor = F.to_tensor(image).to(device)

    # Run inference
    with torch.no_grad():
        predictions = model([img_tensor])[0]

    # Filter by threshold and format results
    detections = []
    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        if score >= threshold:
            label_id = label.item()
            label_name = OBJECT_DICT.get(label_id, f"unknown_{label_id}")

            # Skip background class
            if label_name == 'background':
                continue

            detections.append({
                "type": "object",
                "label": label_name,
                "score": round(score.item(), 3),
                "box": [round(x, 2) for x in box.cpu().tolist()]
            })

    return detections


def detect_arrows(image: Image.Image, model, device, threshold: float = 0.5) -> List[Dict]:
    """
    Detect arrows/flows with keypoints in an image

    Args:
        image: PIL Image
        model: Arrow detection model
        device: torch device
        threshold: Confidence threshold (0-1)

    Returns:
        List of detections with boxes, labels, scores, and keypoints
    """
    # Convert to tensor
    img_tensor = F.to_tensor(image).to(device)

    # Run inference
    with torch.no_grad():
        predictions = model([img_tensor])[0]

    # Filter by threshold and format results
    detections = []
    for box, label, score, keypoints in zip(
        predictions['boxes'],
        predictions['labels'],
        predictions['scores'],
        predictions['keypoints']
    ):
        if score >= threshold:
            label_id = label.item()
            label_name = ARROW_DICT.get(label_id, f"unknown_{label_id}")

            # Skip background class
            if label_name == 'background':
                continue

            # Extract keypoints (x, y, visibility)
            kpts = keypoints.cpu().tolist()
            kpts_formatted = [[round(kpts[i][0], 2), round(kpts[i][1], 2)] for i in range(len(kpts))]

            detections.append({
                "type": "arrow",
                "label": label_name,
                "score": round(score.item(), 3),
                "box": [round(x, 2) for x in box.cpu().tolist()],
                "keypoints": kpts_formatted
            })

    return detections


def _draw_bold_text(draw: ImageDraw.ImageDraw, xy: Tuple[float, float], text: str, font, fill):
    """
    Simulate bold text by drawing it with small pixel offsets.
    """
    x, y = xy
    draw.text((x, y), text, fill=fill, font=font)
    draw.text((x + 1, y), text, fill=fill, font=font)
    draw.text((x, y + 1), text, fill=fill, font=font)
    draw.text((x + 1, y + 1), text, fill=fill, font=font)


def _draw_arrow_direction(
    draw: ImageDraw.ImageDraw,
    keypoints: List[List[float]],
    color: tuple,
    offset: Tuple[float, float] = (0.0, 0.0),
):
    """
    Draw direction based on keypoint order: K0 -> K1.
    """
    if len(keypoints) < 2:
        return
    ox, oy = offset
    x0, y0 = keypoints[0]
    x1, y1 = keypoints[1]
    x0, y0 = x0 + ox, y0 + oy
    x1, y1 = x1 + ox, y1 + oy
    draw.line([(x0, y0), (x1, y1)], fill=color, width=2)

    # Small arrowhead at K1 to show direction
    dx = x1 - x0
    dy = y1 - y0
    length = (dx * dx + dy * dy) ** 0.5
    if length == 0:
        return
    ux, uy = dx / length, dy / length
    size = 10
    left = (x1 - size * ux - size * uy * 0.6, y1 - size * uy + size * ux * 0.6)
    right = (x1 - size * ux + size * uy * 0.6, y1 - size * uy - size * ux * 0.6)
    draw.polygon([left, right, (x1, y1)], fill=color)


def _local_darkness(image, x: float, y: float, radius: int = 8,
                    bbox: Optional[List[float]] = None) -> float:
    """
    Estimate local darkness around a point.
    Returns a value in [0, 255], where higher means darker.
    """
    if image is None:
        return 0.0
    gray = image.convert("L")
    w, h = gray.size
    left = max(int(x) - radius, 0)
    right = min(int(x) + radius, w - 1)
    top = max(int(y) - radius, 0)
    bottom = min(int(y) + radius, h - 1)

    if bbox is not None:
        bx1, by1, bx2, by2 = bbox
        left = max(left, int(bx1))
        right = min(right, int(bx2))
        top = max(top, int(by1))
        bottom = min(bottom, int(by2))
    if right <= left or bottom <= top:
        return 0.0
    patch = gray.crop((left, top, right + 1, bottom + 1))
    pixels = list(patch.getdata())
    if not pixels:
        return 0.0
    avg = sum(pixels) / len(pixels)
    return 255.0 - avg


def _darkness_near_keypoint(
    image,
    kx: float,
    ky: float,
    bbox: Optional[List[float]],
    offset: float = 8.0,
) -> float:
    """
    Measure darkness slightly inside the arrow bbox from a keypoint (toward bbox center).
    """
    if image is None or not bbox:
        return 0.0
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    vx, vy = cx - kx, cy - ky
    length = (vx * vx + vy * vy) ** 0.5 or 1.0
    px = kx + offset * (vx / length)
    py = ky + offset * (vy / length)
    return _local_darkness(image, px, py, bbox=bbox)


def visualize_detections(
    image: Image.Image,
    detections: List[Dict],
    output_path: str,
):
    """
    Draw bounding boxes, labels, and keypoints on image

    Args:
        image: Original PIL Image
        detections: List of detection dictionaries
        output_path: Where to save the annotated image
    """
    draw = ImageDraw.Draw(image)

    # Try to use a nice font, fall back to default if not available
    try:
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        font_small = ImageFont.load_default()

    for det in detections:
        box = det["box"]
        label = det["label"]
        score = det["score"]
        det_type = det.get("type", "object")

        # Get color for this element type
        color_hex = COLORS.get(label, '#9E9E9E')
        color = hex_to_rgb(color_hex)

        # Draw bounding box
        width = 4 if det_type == "arrow" else 3
        draw.rectangle(box, outline=color, width=width)

        # Draw keypoints for arrows
        if det_type == "arrow" and "keypoints" in det:
            keypoints = det["keypoints"]
            bbox = det.get("box")
            d0 = _darkness_near_keypoint(image, keypoints[0][0], keypoints[0][1], bbox)
            d1 = _darkness_near_keypoint(image, keypoints[1][0], keypoints[1][1], bbox)

            for i, (kx, ky) in enumerate(keypoints):
                # Draw keypoint as circle
                radius = 6
                draw.ellipse(
                    [kx - radius, ky - radius, kx + radius, ky + radius],
                    fill=color,
                    outline="white",
                    width=2
                )
                # Draw darkness percentage for both keypoints
                darkness = d0 if i == 0 else d1
                darkness_pct = int(round((darkness / 255.0) * 100))
                _draw_bold_text(
                    draw,
                    (kx + 10, ky - 10),
                    f"{darkness_pct}%",
                    font_small,
                    fill=(0, 0, 0)
                )
            # Draw direction based on keypoint order (raw)
            _draw_arrow_direction(draw, det["keypoints"], color)

        # Draw label background
        text = f"{label} ({score:.2f})"
        bbox = draw.textbbox((box[0], box[1] - 22), text, font=font_small)
        draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=color)

        # Draw label text
        draw.text((box[0], box[1] - 22), text, fill="white", font=font_small)

    # Save annotated image
    image.save(output_path)
    print(f"  ✓ Saved annotated image to {output_path}")


def print_detection_summary(detections: List[Dict]):
    """Print a summary of detected elements"""
    if not detections:
        print("  No elements detected")
        return

    # Count by type
    counts = {}
    for det in detections:
        label = det["label"]
        counts[label] = counts.get(label, 0) + 1

    print(f"  Detected {len(detections)} elements:")
    for label, count in sorted(counts.items()):
        print(f"    - {label}: {count}")


def main():
    """Run BPMN detection demo on images in bpmn_diagrams/png"""

    print("=" * 80)
    print("BPMN Detection Demo")
    print("Model: ELCA-SA/BPMN_Detection")
    print("Two-stage detection: Objects (Faster R-CNN) + Arrows (Keypoint R-CNN)")
    print("=" * 80)
    print()

    # Load models
    model_object, model_arrow, device = load_models()
    print()

    # Find input images
    input_dir = Path("bpmn_diagrams/png")
    output_dir = Path("bpmn_detection_results")
    output_dir.mkdir(exist_ok=True)

    image_files = sorted(input_dir.glob("*.png"))

    if not image_files:
        print(f"No PNG files found in {input_dir}")
        return

    print(f"Found {len(image_files)} images to process")
    print()

    # Process each image
    for i, image_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processing: {image_path.name}")

        # Load image
        image = Image.open(image_path).convert("RGB")
        print(f"  Image size: {image.size[0]}x{image.size[1]}")

        # Detect objects
        print("  → Detecting objects...")
        object_detections = detect_objects(image, model_object, device, threshold=0.5)

        # Detect arrows
        print("  → Detecting arrows...")
        arrow_detections = detect_arrows(image, model_arrow, device, threshold=0.5)

        # Combine all detections
        all_detections = object_detections + arrow_detections

        # Print summary
        print_detection_summary(all_detections)

        # Visualize and save
        output_path = output_dir / f"detected_{image_path.name}"
        visualize_detections(
            image.copy(),
            all_detections,
            str(output_path)
        )

        print()

    print("=" * 80)
    print("Detection complete!")
    print(f"Results saved to: {output_dir}/")
    print()
    print("Legend:")
    print("  - Objects: tasks, gateways, events, pools, lanes, data stores")
    print("  - Arrows: sequence flows, data associations, message flows")
    print("  - Keypoints (K0, K1): start and end points of arrows")
    print("=" * 80)


if __name__ == "__main__":
    main()
