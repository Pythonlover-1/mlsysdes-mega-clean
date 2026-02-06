"""Pydantic schemas for BPMN detection API"""
from typing import List, Optional
from pydantic import BaseModel, Field


class ConversionParams(BaseModel):
    """Parameters for BPMN detection and graph conversion"""

    # Detection
    detection_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Min confidence for detections")
    arrow_dilate: int = Field(0, ge=0, le=10, description="Dilate lines by N pixels before arrow detection (0 = disabled)")

    # Graph building
    connection_threshold: float = Field(200.0, ge=0.0, description="Max distance to connect arrow to node")
    overlap_iou: float = Field(0.7, ge=0.0, le=1.0, description="IoU threshold for suppressing overlaps")
    arrow_fallback: bool = Field(True, description="Use bbox fallback when keypoints fail to connect")

    # Simple mode (for simple diagrams without pools/lanes)
    simple_mode: bool = Field(False, description="Enable simple mode for diagrams without pools/lanes")
    simple_pool_task_size_ratio: float = Field(0.7, ge=0.0, le=1.0, description="Size ratio for promoting pools to tasks")
    simple_merge_iou: float = Field(0.6, ge=0.0, le=1.0, description="IoU threshold for merging overlapping blocks")

    # OCR preprocessing
    ocr_upscale: int = Field(2, ge=1, le=4, description="Upscale factor for OCR crops")
    ocr_binarize: bool = Field(False, description="Binarize image before OCR")
    ocr_binarize_thr: int = Field(170, ge=0, le=255, description="Threshold for binarization")
    ocr_denoise: bool = Field(False, description="Apply median filter to reduce noise (can blur small text)")
    ocr_noisy_mode: bool = Field(False, description="Auto-detect noisy images and apply stronger cleanup")
    ocr_noisy_std: float = Field(35.0, ge=0.0, description="Std threshold for noisy image detection")

    # OCR crop settings
    ocr_inset_px: int = Field(2, ge=0, description="Inset padding in pixels to reduce border noise")
    ocr_inset_pct: float = Field(0.02, ge=0.0, le=0.5, description="Inset padding as percentage of box size")
    ocr_loose_scale: float = Field(0.0, ge=0.0, le=2.0, description="Scale factor for loose crop fallback (0.0 = disabled)")

    # OCR lane/pool
    ocr_lane_top_pct: float = Field(0.05, ge=0.05, le=1.0, description="After rotating lane 90° CW, keep top X% for OCR")

    # OCR Tesseract
    ocr_lang: str = Field("eng+rus", description="Tesseract language codes")
    ocr_psm_list: str = Field("6,7", description="Comma-separated PSM modes to try")
    ocr_oem: int = Field(1, ge=0, le=3, description="Tesseract OCR Engine Mode")
    ocr_tesseract_config: str = Field("-c preserve_interword_spaces=1", description="Extra Tesseract config")
    ocr_tesseract_whitelist: str = Field("", description="Explicit character whitelist for Tesseract")
    ocr_tesseract_whitelist_lang: bool = Field(False, description="Use language-based character whitelist (if explicit whitelist is empty)")

    # OCR postprocessing
    ocr_fix_typos: bool = Field(True, description="Fix mixed Cyrillic/Latin characters")
    ocr_spellcheck: bool = Field(True, description="Enable spell-check correction")
    ocr_spellcheck_topn: int = Field(20000, ge=1000, description="Wordlist size for spell-check")
    ocr_spellcheck_min_zipf: float = Field(3.0, ge=0.0, description="Min word frequency for correction")
    ocr_spellcheck_score: float = Field(88.0, ge=0.0, le=100.0, description="Min fuzzy-match score")
    ocr_user_vocab: str = Field("", description="Path to user vocabulary file")

    # Debug
    ocr_debug_dir: str = Field("", description="Directory to save debug crops (empty to disable)")


class BPMNNodeResponse(BaseModel):
    id: str
    label: str
    name: str
    score: float
    box: List[float]
    center: List[float]
    lane_id: Optional[str] = None


class BPMNEdgeResponse(BaseModel):
    id: str
    label: str
    name: str
    score: float
    source: Optional[str] = None
    target: Optional[str] = None
    keypoints: List[List[float]]


class BPMNLaneResponse(BaseModel):
    id: str
    label: str
    name: str
    score: float
    box: List[float]


class BPMNGraphResponse(BaseModel):
    nodes: List[BPMNNodeResponse]
    edges: List[BPMNEdgeResponse]
    lanes: List[BPMNLaneResponse]


class ConversionResponse(BaseModel):
    graph: BPMNGraphResponse
    csv: str
    filename: str
    image_size: List[int]
    processing_time_ms: int
