"""
BPMN Detection Service - Singleton Model Manager

Loads BPMN detection models (Faster R-CNN + Keypoint R-CNN) once at startup
and provides async inference with concurrency control.
"""
import asyncio
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn, keypointrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor
from torchvision.transforms import functional as F

from app.services.metrics import get_memory_mb

logger = logging.getLogger(__name__)

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


@dataclass
class DetectionMetrics:
    """Metrics collected during detection."""
    image_load_ms: float = 0.0
    image_size: Tuple[int, int] = (0, 0)
    image_bytes: int = 0

    object_detection_ms: float = 0.0
    objects_detected: int = 0

    arrow_dilation_ms: float = 0.0
    arrow_detection_ms: float = 0.0
    arrows_detected: int = 0

    total_ms: float = 0.0
    memory_before_mb: float = 0.0
    memory_after_mb: float = 0.0

    def log_summary(self):
        """Log detection metrics summary."""
        mem_delta = self.memory_after_mb - self.memory_before_mb
        sign = "+" if mem_delta > 0 else ""

        logger.info("=" * 60)
        logger.info("[DETECTION METRICS]")
        logger.info(f"  Image: {self.image_size[0]}x{self.image_size[1]} ({self.image_bytes / 1024:.1f} KB)")
        logger.info(f"  Load image:      {self.image_load_ms:>7.1f} ms")
        logger.info(f"  Object detection:{self.object_detection_ms:>7.1f} ms -> {self.objects_detected} objects")
        if self.arrow_dilation_ms > 0:
            logger.info(f"  Arrow dilation:  {self.arrow_dilation_ms:>7.1f} ms")
        logger.info(f"  Arrow detection: {self.arrow_detection_ms:>7.1f} ms -> {self.arrows_detected} arrows")
        logger.info(f"  TOTAL:           {self.total_ms:>7.1f} ms")
        logger.info(f"  Memory: {self.memory_before_mb:.1f} -> {self.memory_after_mb:.1f} MB ({sign}{mem_delta:.1f} MB)")
        logger.info("=" * 60)


def _get_faster_rcnn_model(num_classes: int):
    """Create Faster R-CNN model for object detection"""
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features  # type: ignore
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def _get_arrow_model(num_classes: int, num_keypoints: int = 2):
    """Create Keypoint R-CNN model for arrow detection with keypoints"""
    model = keypointrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features  # type: ignore
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(512, num_keypoints)
    return model


def _dilate_lines(image: Image.Image, dilate_px: int) -> Image.Image:
    """
    Dilate dark lines (accent pixels) in the image.

    This thickens all lines by dilate_px pixels in each direction,
    making thin arrows easier to detect.

    Args:
        image: PIL Image in RGB mode
        dilate_px: Number of pixels to expand lines (kernel radius)

    Returns:
        PIL Image with dilated lines
    """
    if dilate_px <= 0:
        return image

    # Convert to numpy array
    img_array = np.array(image)

    # Convert to grayscale for line detection
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Invert: lines become white (high values) for dilation
    # Threshold to get binary mask of dark pixels (lines)
    # Using adaptive threshold to handle varying backgrounds
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Create dilation kernel (square)
    # kernel_size = 2 * dilate_px + 1 makes the line expand by dilate_px on each side
    kernel_size = 2 * dilate_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Dilate the binary mask
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # Apply dilated mask back to original image
    # Where dilated mask is white (was a line), set pixel to dark
    result = img_array.copy()

    # Get the darkest color from original lines for filling
    line_mask = binary > 0
    if np.any(line_mask):
        # Use median color of original line pixels
        line_colors = img_array[line_mask]
        fill_color = np.median(line_colors, axis=0).astype(np.uint8)
    else:
        fill_color = np.array([0, 0, 0], dtype=np.uint8)

    # Apply: where dilated but not original, fill with line color
    new_line_pixels = (dilated > 0) & (binary == 0)
    result[new_line_pixels] = fill_color

    return Image.fromarray(result)


class BPMNModelManager:
    """
    Singleton manager for BPMN detection models.

    - Loads models once at startup
    - Provides thread-safe inference via ThreadPoolExecutor
    - Limits concurrent requests via asyncio.Semaphore
    """
    _instance: Optional["BPMNModelManager"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, max_workers: int = 4):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True

        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.semaphore = asyncio.Semaphore(max_workers)

        self.model_object = None
        self.model_arrow = None
        self.device = None
        self.models_loaded = False

    def load_models(self):
        """Load models at startup (called from FastAPI lifespan)"""
        if self.models_loaded:
            logger.info("Models already loaded, skipping")
            return

        logger.info("Loading BPMN Detection models from Hugging Face...")

        # Download model weights
        logger.info("Downloading model_object.pth (~166 MB)...")
        model_object_path = hf_hub_download(
            repo_id="ELCA-SA/BPMN_Detection",
            filename="model_object.pth"
        )

        logger.info("Downloading model_arrow.pth (~237 MB)...")
        model_arrow_path = hf_hub_download(
            repo_id="ELCA-SA/BPMN_Detection",
            filename="model_arrow.pth"
        )

        # Determine device (CPU only for this deployment)
        self.device = torch.device('cpu')
        logger.info(f"Using device: {self.device}")

        # Create model architectures
        logger.info("Initializing model architectures...")
        self.model_object = _get_faster_rcnn_model(len(OBJECT_DICT))
        self.model_arrow = _get_arrow_model(len(ARROW_DICT), num_keypoints=2)

        # Load weights
        logger.info("Loading weights...")
        self.model_object.load_state_dict(
            torch.load(model_object_path, map_location=self.device, weights_only=True)
        )
        self.model_arrow.load_state_dict(
            torch.load(model_arrow_path, map_location=self.device, weights_only=True)
        )

        # Set to evaluation mode
        self.model_object.eval()
        self.model_arrow.eval()

        self.models_loaded = True
        logger.info("Models loaded successfully")

    def _detect_objects_sync(
        self, image: Image.Image, threshold: float = 0.5
    ) -> Tuple[List[Dict], float]:
        """Detect BPMN objects (tasks, gateways, events, etc.). Returns (detections, duration_ms)."""
        start = time.perf_counter()

        img_tensor = F.to_tensor(image).to(self.device)

        with torch.no_grad():
            predictions = self.model_object([img_tensor])[0]

        detections = []
        for box, label, score in zip(
            predictions['boxes'], predictions['labels'], predictions['scores']
        ):
            if score >= threshold:
                label_id = label.item()
                label_name = OBJECT_DICT.get(label_id, f"unknown_{label_id}")

                if label_name == 'background':
                    continue

                detections.append({
                    "type": "object",
                    "label": label_name,
                    "score": round(score.item(), 3),
                    "box": [round(x, 2) for x in box.cpu().tolist()]
                })

        duration_ms = (time.perf_counter() - start) * 1000
        return detections, duration_ms

    def _detect_arrows_sync(
        self, image: Image.Image, threshold: float = 0.5, dilate_px: int = 0
    ) -> Tuple[List[Dict], float, float]:
        """Detect arrows/flows with keypoints. Returns (detections, detection_ms, dilation_ms)."""
        dilation_ms = 0.0

        # Apply line dilation if requested
        if dilate_px > 0:
            dilation_start = time.perf_counter()
            image = _dilate_lines(image, dilate_px)
            dilation_ms = (time.perf_counter() - dilation_start) * 1000

        start = time.perf_counter()
        img_tensor = F.to_tensor(image).to(self.device)

        with torch.no_grad():
            predictions = self.model_arrow([img_tensor])[0]

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

                if label_name == 'background':
                    continue

                kpts = keypoints.cpu().tolist()
                kpts_formatted = [
                    [round(kpts[i][0], 2), round(kpts[i][1], 2)]
                    for i in range(len(kpts))
                ]

                detections.append({
                    "type": "arrow",
                    "label": label_name,
                    "score": round(score.item(), 3),
                    "box": [round(x, 2) for x in box.cpu().tolist()],
                    "keypoints": kpts_formatted
                })

        duration_ms = (time.perf_counter() - start) * 1000
        return detections, duration_ms, dilation_ms

    def _detect_sync(
        self, image_bytes: bytes, threshold: float = 0.5, arrow_dilate: int = 0
    ) -> Tuple[List[Dict], Tuple[int, int], Image.Image, DetectionMetrics]:
        """
        Full detection pipeline (sync, runs in thread pool).
        Returns (detections, image_size, pil_image, metrics).
        """
        metrics = DetectionMetrics()
        metrics.memory_before_mb = get_memory_mb()
        metrics.image_bytes = len(image_bytes)
        total_start = time.perf_counter()

        # Load image
        load_start = time.perf_counter()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_size = image.size  # (width, height)
        metrics.image_load_ms = (time.perf_counter() - load_start) * 1000
        metrics.image_size = image_size

        # Object detection
        object_detections, metrics.object_detection_ms = self._detect_objects_sync(image, threshold)
        metrics.objects_detected = len(object_detections)

        # Arrow detection
        arrow_detections, metrics.arrow_detection_ms, metrics.arrow_dilation_ms = self._detect_arrows_sync(
            image, threshold, arrow_dilate
        )
        metrics.arrows_detected = len(arrow_detections)

        # Finalize metrics
        metrics.total_ms = (time.perf_counter() - total_start) * 1000
        metrics.memory_after_mb = get_memory_mb()

        all_detections = object_detections + arrow_detections
        return all_detections, image_size, image, metrics

    async def detect(
        self, image_bytes: bytes, threshold: float = 0.5, arrow_dilate: int = 0
    ) -> Tuple[List[Dict], Tuple[int, int], Image.Image, DetectionMetrics]:
        """
        Async detection with concurrency control.
        Returns (detections, image_size, pil_image, metrics).

        Args:
            image_bytes: Raw image bytes
            threshold: Detection confidence threshold
            arrow_dilate: Dilate lines by N pixels before arrow detection (0 = disabled)
        """
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        async with self.semaphore:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                lambda: self._detect_sync(image_bytes, threshold, arrow_dilate)
            )
            return result  # type: ignore


# Global singleton instance
model_manager = BPMNModelManager(max_workers=4)
