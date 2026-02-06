import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.models.schemas import ConversionParams, ConversionResponse
from app.services.bpmn_detector import model_manager
from app.services.bpmn_graph import GraphBuildParams, OCRParams, detections_to_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models at startup"""
    logger.info("Starting up - loading BPMN detection models...")
    model_manager.load_models()
    logger.info("Models loaded, ready to serve requests")
    yield
    logger.info("Shutting down")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_conversion_params(
    # Detection
    detection_threshold: Annotated[float, Query(ge=0.0, le=1.0, description="Min confidence for detections")] = 0.5,
    arrow_dilate: Annotated[int, Query(ge=0, le=10, description="Dilate lines by N pixels before arrow detection")] = 0,
    # Graph building
    connection_threshold: Annotated[float, Query(ge=0.0, description="Max distance to connect arrow to node")] = 200.0,
    overlap_iou: Annotated[float, Query(ge=0.0, le=1.0, description="IoU threshold for suppressing overlaps")] = 0.7,
    arrow_fallback: Annotated[bool, Query(description="Use bbox fallback when keypoints fail")] = True,
    # Simple mode (for diagrams without pools/lanes)
    simple_mode: Annotated[bool, Query(description="Enable simple mode for diagrams without pools/lanes")] = False,
    simple_pool_task_size_ratio: Annotated[float, Query(ge=0.0, le=1.0, description="Size ratio for promoting pools to tasks")] = 0.7,
    simple_merge_iou: Annotated[float, Query(ge=0.0, le=1.0, description="IoU for merging overlapping blocks")] = 0.6,
    # OCR preprocessing
    ocr_upscale: Annotated[int, Query(ge=1, le=4, description="Upscale factor for OCR crops")] = 2,
    ocr_binarize: Annotated[bool, Query(description="Binarize image before OCR")] = False,
    ocr_binarize_thr: Annotated[int, Query(ge=0, le=255, description="Threshold for binarization")] = 170,
    ocr_denoise: Annotated[bool, Query(description="Apply median filter to reduce noise")] = False,
    ocr_noisy_mode: Annotated[bool, Query(description="Auto-detect noisy images")] = False,
    ocr_noisy_std: Annotated[float, Query(ge=0.0, description="Std threshold for noisy detection")] = 35.0,
    # OCR crop settings
    ocr_inset_px: Annotated[int, Query(ge=0, description="Inset padding in pixels")] = 2,
    ocr_inset_pct: Annotated[float, Query(ge=0.0, le=0.5, description="Inset padding as percentage")] = 0.02,
    ocr_loose_scale: Annotated[float, Query(ge=0.0, le=2.0, description="Scale factor for loose crop (0.0 = disabled)")] = 0.0,
    # OCR lane/pool
    ocr_lane_top_pct: Annotated[float, Query(ge=0.05, le=1.0, description="After rotating lane 90° CW, keep top X%")] = 0.05,
    # OCR Tesseract
    ocr_lang: Annotated[str, Query(description="Tesseract language codes")] = "eng+rus",
    ocr_psm_list: Annotated[str, Query(description="Comma-separated PSM modes to try")] = "6,7",
    ocr_oem: Annotated[int, Query(ge=0, le=3, description="Tesseract OCR Engine Mode")] = 1,
    ocr_tesseract_config: Annotated[str, Query(description="Extra Tesseract config")] = "-c preserve_interword_spaces=1",
    ocr_tesseract_whitelist: Annotated[str, Query(description="Explicit character whitelist")] = "",
    ocr_tesseract_whitelist_lang: Annotated[bool, Query(description="Auto-generate whitelist from lang")] = False,
    # OCR postprocessing
    ocr_fix_typos: Annotated[bool, Query(description="Fix mixed Cyrillic/Latin characters")] = True,
    ocr_spellcheck: Annotated[bool, Query(description="Enable spell-check correction")] = True,
    ocr_spellcheck_topn: Annotated[int, Query(ge=1000, description="Wordlist size for spell-check")] = 20000,
    ocr_spellcheck_min_zipf: Annotated[float, Query(ge=0.0, description="Min word frequency for correction")] = 3.0,
    ocr_spellcheck_score: Annotated[float, Query(ge=0.0, le=100.0, description="Min fuzzy-match score")] = 88.0,
    ocr_user_vocab: Annotated[str, Query(description="Path to user vocabulary file")] = "",
    # Debug
    ocr_debug_dir: Annotated[str, Query(description="Directory to save debug crops")] = "",
) -> ConversionParams:
    return ConversionParams(
        detection_threshold=detection_threshold,
        arrow_dilate=arrow_dilate,
        connection_threshold=connection_threshold,
        overlap_iou=overlap_iou,
        arrow_fallback=arrow_fallback,
        simple_mode=simple_mode,
        simple_pool_task_size_ratio=simple_pool_task_size_ratio,
        simple_merge_iou=simple_merge_iou,
        ocr_upscale=ocr_upscale,
        ocr_binarize=ocr_binarize,
        ocr_binarize_thr=ocr_binarize_thr,
        ocr_denoise=ocr_denoise,
        ocr_noisy_mode=ocr_noisy_mode,
        ocr_noisy_std=ocr_noisy_std,
        ocr_inset_px=ocr_inset_px,
        ocr_inset_pct=ocr_inset_pct,
        ocr_loose_scale=ocr_loose_scale,
        ocr_lane_top_pct=ocr_lane_top_pct,
        ocr_lang=ocr_lang,
        ocr_psm_list=ocr_psm_list,
        ocr_oem=ocr_oem,
        ocr_tesseract_config=ocr_tesseract_config,
        ocr_tesseract_whitelist=ocr_tesseract_whitelist,
        ocr_tesseract_whitelist_lang=ocr_tesseract_whitelist_lang,
        ocr_fix_typos=ocr_fix_typos,
        ocr_spellcheck=ocr_spellcheck,
        ocr_spellcheck_topn=ocr_spellcheck_topn,
        ocr_spellcheck_min_zipf=ocr_spellcheck_min_zipf,
        ocr_spellcheck_score=ocr_spellcheck_score,
        ocr_user_vocab=ocr_user_vocab,
        ocr_debug_dir=ocr_debug_dir,
    )


@app.post("/api/convert", response_model=ConversionResponse)
async def convert(
    file: UploadFile = File(...),
    params: ConversionParams = Depends(get_conversion_params),
):
    """
    Convert a BPMN diagram image to a structured graph.

    Returns nodes, edges, lanes, and CSV representation.

    All parameters are optional and have sensible defaults.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    image_bytes = await file.read()
    start_time = time.time()

    try:
        # Run detection with configurable threshold and optional line dilation
        detections, image_size, pil_image, detection_metrics = await model_manager.detect(
            image_bytes,
            threshold=params.detection_threshold,
            arrow_dilate=params.arrow_dilate,
        )

        # Log detection metrics
        detection_metrics.log_summary()

        # Build OCR params
        ocr_params = OCRParams(
            upscale=params.ocr_upscale,
            binarize=params.ocr_binarize,
            binarize_thr=params.ocr_binarize_thr,
            denoise=params.ocr_denoise,
            noisy_mode=params.ocr_noisy_mode,
            noisy_std=params.ocr_noisy_std,
            inset_px=params.ocr_inset_px,
            inset_pct=params.ocr_inset_pct,
            loose_scale=params.ocr_loose_scale,
            lane_top_pct=params.ocr_lane_top_pct,
            lang=params.ocr_lang,
            psm_list=params.ocr_psm_list,
            oem=params.ocr_oem,
            tesseract_config=params.ocr_tesseract_config,
            tesseract_whitelist=params.ocr_tesseract_whitelist,
            tesseract_whitelist_lang=params.ocr_tesseract_whitelist_lang,
            fix_typos=params.ocr_fix_typos,
            spellcheck=params.ocr_spellcheck,
            spellcheck_topn=params.ocr_spellcheck_topn,
            spellcheck_min_zipf=params.ocr_spellcheck_min_zipf,
            spellcheck_score=params.ocr_spellcheck_score,
            user_vocab=params.ocr_user_vocab,
            debug_dir=params.ocr_debug_dir,
        )

        # Build graph params
        graph_params = GraphBuildParams(
            connection_threshold=params.connection_threshold,
            overlap_iou=params.overlap_iou,
            arrow_fallback=params.arrow_fallback,
            simple_mode=params.simple_mode,
            simple_pool_task_size_ratio=params.simple_pool_task_size_ratio,
            simple_merge_iou=params.simple_merge_iou,
            ocr=ocr_params,
        )

        # Convert to graph (logs its own metrics)
        graph = detections_to_graph(detections, image=pil_image, params=graph_params)
        graph.image_size = image_size

        processing_time_ms = int((time.time() - start_time) * 1000)

        # Log total pipeline summary
        logger.info("=" * 60)
        logger.info(f"[PIPELINE TOTAL] {processing_time_ms} ms")
        logger.info(f"  Detection:    {detection_metrics.total_ms:>7.1f} ms")
        logger.info(f"  Graph build:  {processing_time_ms - detection_metrics.total_ms:>7.1f} ms")
        logger.info(f"  Output: {len(graph.nodes)} nodes, {len(graph.edges)} edges, {len(graph.lanes)} lanes")
        logger.info("=" * 60)

        return ConversionResponse(
            graph=graph.to_dict(),
            csv=graph.to_csv(),
            filename=file.filename or "diagram.png",
            image_size=list(image_size),
            processing_time_ms=processing_time_ms,
        )

    except Exception as e:
        logger.exception("Detection failed")
        raise HTTPException(500, f"Detection error: {e}")


@app.get("/api/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": model_manager.models_loaded
    }
