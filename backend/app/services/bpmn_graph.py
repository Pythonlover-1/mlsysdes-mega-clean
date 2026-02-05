"""
BPMN Graph Conversion

Converts raw detection results to a structured graph with nodes, edges, and lanes.
Includes OCR for text extraction from detected elements.
"""
import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from PIL import Image, ImageFilter, ImageOps

if TYPE_CHECKING:
    from app.models.schemas import ConversionParams

# Optional dependencies
try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    from rapidfuzz import fuzz, process
    from wordfreq import top_n_list, zipf_frequency
    HAS_SPELL = True
except ImportError:
    HAS_SPELL = False


@dataclass
class BPMNLane:
    """Represents a BPMN lane/pool (container for grouping elements)"""
    id: str
    label: str
    name: str
    score: float
    box: List[float]

    def contains_point(self, x: float, y: float) -> bool:
        x1, y1, x2, y2 = self.box
        return x1 <= x <= x2 and y1 <= y <= y2

    def contains_box(self, other_box: List[float]) -> bool:
        ox1, oy1, ox2, oy2 = other_box
        center_x = (ox1 + ox2) / 2
        center_y = (oy1 + oy2) / 2
        return self.contains_point(center_x, center_y)


@dataclass
class BPMNNode:
    """Represents a BPMN element (task, gateway, event, etc.)"""
    id: str
    label: str
    name: str
    score: float
    box: List[float]
    center: Tuple[float, float] = field(default=(0, 0))
    lane_id: Optional[str] = None

    def __post_init__(self):
        x1, y1, x2, y2 = self.box
        self.center = ((x1 + x2) / 2, (y1 + y2) / 2)

    def distance_to_point(self, x: float, y: float) -> float:
        cx, cy = self.center
        return math.sqrt((x - cx) ** 2 + (y - cy) ** 2)


@dataclass
class BPMNEdge:
    """Represents a BPMN flow (sequenceFlow, messageFlow, dataAssociation)"""
    id: str
    label: str
    score: float
    box: List[float]
    keypoints: List[List[float]]
    source_id: Optional[str] = None
    target_id: Optional[str] = None


@dataclass
class BPMNGraph:
    """Graph representation of a BPMN diagram"""
    nodes: Dict[str, BPMNNode] = field(default_factory=dict)
    edges: List[BPMNEdge] = field(default_factory=list)
    lanes: Dict[str, BPMNLane] = field(default_factory=dict)
    image_size: Tuple[int, int] = (0, 0)

    def add_node(self, node: BPMNNode):
        self.nodes[node.id] = node

    def add_edge(self, edge: BPMNEdge):
        self.edges.append(edge)

    def add_lane(self, lane: BPMNLane):
        self.lanes[lane.id] = lane

    def to_dict(self) -> dict:
        return {
            "nodes": [
                {
                    "id": n.id,
                    "label": n.label,
                    "name": n.name,
                    "score": n.score,
                    "box": n.box,
                    "center": list(n.center),
                    "lane_id": n.lane_id
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "id": e.id,
                    "label": e.label,
                    "name": e.label,  # edges don't have custom names, use label
                    "score": e.score,
                    "source": e.source_id,
                    "target": e.target_id,
                    "keypoints": e.keypoints
                }
                for e in self.edges
            ],
            "lanes": [
                {
                    "id": l.id,
                    "label": l.label,
                    "name": l.name,
                    "score": l.score,
                    "box": l.box
                }
                for l in self.lanes.values()
            ]
        }

    def to_csv(self) -> str:
        """Export edges as CSV - only connected edges"""
        lines = [
            "source_id,source_label,source_name,source_lane,"
            "target_id,target_label,target_name,target_lane,edge_type,score"
        ]
        for edge in self.edges:
            if not edge.source_id or not edge.target_id:
                continue
            src = self.nodes.get(edge.source_id)
            tgt = self.nodes.get(edge.target_id)
            src_label = src.label if src else "unknown"
            tgt_label = tgt.label if tgt else "unknown"
            src_name = _csv_escape(src.name) if src else ""
            tgt_name = _csv_escape(tgt.name) if tgt else ""
            src_lane = src.lane_id if src and src.lane_id else ""
            tgt_lane = tgt.lane_id if tgt and tgt.lane_id else ""
            lines.append(
                f"{edge.source_id},{src_label},{src_name},{src_lane},"
                f"{edge.target_id},{tgt_label},{tgt_name},{tgt_lane},"
                f"{edge.label},{edge.score}"
            )
        return "\n".join(lines)


def _csv_escape(s: str) -> str:
    """Escape string for CSV (handle commas and quotes)"""
    if not s:
        return ""
    if "," in s or '"' in s or "\n" in s:
        return '"' + s.replace('"', '""') + '"'
    return s


# --- OCR helpers ---

_LATIN_TO_CYR = str.maketrans({
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М",
    "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У",
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у",
    "k": "к", "m": "м", "h": "н", "b": "в", "t": "т",
})

_CYR_TO_LATIN = str.maketrans({
    "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "К": "K", "М": "M",
    "О": "O", "Р": "P", "Т": "T", "Х": "X", "У": "Y",
    "а": "a", "с": "c", "е": "e", "о": "o", "р": "p", "х": "x", "у": "y",
    "к": "k", "м": "m", "н": "h", "в": "b", "т": "t",
})

_WORDLISTS: Dict[str, List[str]] = {}


def _script_ratio(word: str) -> Tuple[int, int]:
    lat = sum(1 for ch in word if "A" <= ch <= "Z" or "a" <= ch <= "z")
    cyr = sum(1 for ch in word if ("А" <= ch <= "я") or ch in {"Ё", "ё"})
    return lat, cyr


def _fix_word_by_script(word: str) -> str:
    if not word:
        return word
    lat, cyr = _script_ratio(word)
    if cyr > lat:
        w = word.translate(_LATIN_TO_CYR)
        w = w.replace("|", "л").replace("0", "о").replace("1", "l")
        return w
    if lat > cyr:
        w = word.translate(_CYR_TO_LATIN)
        w = w.replace("|", "l").replace("0", "O").replace("rn", "m")
        return w
    return word


def _get_wordlist(lang: str, topn: int = 20000) -> List[str]:
    cache_key = f"{lang}_{topn}"
    if cache_key in _WORDLISTS:
        return _WORDLISTS[cache_key]
    try:
        wl = top_n_list(lang, topn)
    except Exception:
        wl = []
    _WORDLISTS[cache_key] = wl
    return wl


def _match_case(src: str, tgt: str) -> str:
    if src.isupper():
        return tgt.upper()
    if src[:1].isupper():
        return tgt[:1].upper() + tgt[1:]
    return tgt


def _spell_correct_word(
    word: str,
    lang: str,
    min_zipf: float = 3.0,
    score_cutoff: float = 88.0,
    topn: int = 20000
) -> str:
    if not HAS_SPELL or len(word) < 4 or len(word) > 30:
        return word
    if any(ch.isdigit() for ch in word):
        return word
    low = word.lower()
    if zipf_frequency(low, lang) >= min_zipf:
        return word
    wl = _get_wordlist(lang, topn)
    if not wl:
        return word
    match = process.extractOne(low, wl, scorer=fuzz.ratio, score_cutoff=score_cutoff)
    if not match:
        return word
    cand = match[0]
    if zipf_frequency(cand, lang) < min_zipf:
        return word
    return _match_case(word, cand)


def _normalize_ocr_punct(text: str) -> str:
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = text.replace(""", '"').replace(""", '"').replace("'", "'").replace("`", "'")
    return " ".join(text.split()).strip()


def _fix_ocr_typos(
    text: str,
    spellcheck: bool = True,
    spellcheck_topn: int = 20000,
    spellcheck_min_zipf: float = 3.0,
    spellcheck_score: float = 88.0
) -> str:
    text = _normalize_ocr_punct(text)
    if not text:
        return text
    tokens = re.split(r"(\W+)", text, flags=re.UNICODE)
    fixed: List[str] = []
    for tok in tokens:
        if not tok or re.match(r"\W+", tok, flags=re.UNICODE):
            fixed.append(tok)
            continue
        w = _fix_word_by_script(tok)
        lat, cyr = _script_ratio(w)
        if HAS_SPELL and spellcheck:
            if cyr >= lat + 2:
                w = _spell_correct_word(w, "ru", spellcheck_min_zipf, spellcheck_score, spellcheck_topn)
            elif lat >= cyr + 2:
                w = _spell_correct_word(w, "en", spellcheck_min_zipf, spellcheck_score, spellcheck_topn)
        fixed.append(w)
    return "".join(fixed).strip()


def _preprocess_crop(
    crop: Image.Image,
    upscale: int = 2,
    binarize: bool = False,
    binarize_thr: int = 170,
    denoise: bool = True,
    noisy_mode: bool = False,
    noisy_std: float = 35.0
) -> Image.Image:
    gray = crop.convert("L")
    gray = ImageOps.autocontrast(gray)

    # Detect noisy/low-contrast crops and apply stronger cleanup
    is_noisy = False
    if noisy_mode:
        try:
            hist = gray.histogram()
            total = sum(hist) or 1
            mean = sum(i * h for i, h in enumerate(hist)) / total
            var = sum(((i - mean) ** 2) * h for i, h in enumerate(hist)) / total
            std = var ** 0.5
            is_noisy = std > noisy_std
        except Exception:
            is_noisy = False

    if denoise:
        if is_noisy:
            gray = gray.filter(ImageFilter.MedianFilter(size=5))
            gray = gray.filter(ImageFilter.GaussianBlur(radius=0.6))
        else:
            gray = gray.filter(ImageFilter.MedianFilter(size=3))

    gray = gray.filter(ImageFilter.SHARPEN)

    w, h = gray.size
    scale = max(1, min(upscale, 4))
    if binarize:
        thr = binarize_thr
        gray = gray.point(lambda p: 255 if p > thr else 0)
    return gray.resize((max(1, w * scale), max(1, h * scale)), Image.Resampling.LANCZOS)


def _tesseract_ocr(
    crop: Image.Image,
    lang: str = "eng+rus",
    psm_list: str = "6,7",
    oem: int = 1,
    tesseract_config: str = "-c preserve_interword_spaces=1",
    tesseract_whitelist: str = "",
    tesseract_whitelist_lang: bool = False
) -> Tuple[str, float]:
    if not HAS_TESSERACT:
        return "", 0.0
    psm_modes = psm_list.split(",")

    # Build whitelist: use explicit if provided, otherwise generate from lang if requested
    whitelist = tesseract_whitelist.strip()
    if not whitelist and tesseract_whitelist_lang:
        whitelist = (
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
            "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
            "0123456789 .,;:!?-()\"'"
        )

    best_text = ""
    best_conf = -1.0
    for psm in psm_modes:
        psm = psm.strip()
        if not psm:
            continue

        config = f"--oem {oem} --psm {psm}"
        if tesseract_config:
            config = f"{config} {tesseract_config}"
        if whitelist:
            config = f"{config} -c tessedit_char_whitelist={whitelist}"

        try:
            data = pytesseract.image_to_data(
                crop, lang=lang, config=config,
                output_type=pytesseract.Output.DICT
            )
            words = []
            confs = []
            for i in range(len(data.get("text", []))):
                text = (data["text"][i] or "").strip()
                if not text:
                    continue
                try:
                    conf = float(data["conf"][i])
                except Exception:
                    conf = -1.0
                if conf < 0:
                    continue
                words.append(text)
                confs.append(conf)
            text_out = " ".join(words).strip()
            avg_conf = sum(confs) / len(confs) if confs else 0.0
            if avg_conf > best_conf or (avg_conf == best_conf and len(text_out) > len(best_text)):
                best_text, best_conf = text_out, avg_conf
        except Exception:
            text = pytesseract.image_to_string(crop, lang=lang, config=config)
            text = " ".join(text.split()).strip()
            if len(text) > len(best_text):
                best_text, best_conf = text, best_conf
    return best_text, best_conf


@dataclass
class OCRParams:
    """OCR configuration parameters"""
    # Preprocessing
    upscale: int = 2
    binarize: bool = False
    binarize_thr: int = 170
    denoise: bool = True
    noisy_mode: bool = False
    noisy_std: float = 35.0

    # Crop settings
    inset_px: int = 2
    inset_pct: float = 0.02
    loose_scale: float = 0.0  # Disabled: was causing text cutoff

    # Tesseract
    lang: str = "eng+rus"
    psm_list: str = "6,7"
    oem: int = 1
    tesseract_config: str = "-c preserve_interword_spaces=1"
    tesseract_whitelist: str = ""  # Explicit whitelist
    tesseract_whitelist_lang: bool = False  # Auto-generate whitelist if explicit is empty

    # Postprocessing
    fix_typos: bool = True
    spellcheck: bool = True
    spellcheck_topn: int = 20000
    spellcheck_min_zipf: float = 3.0
    spellcheck_score: float = 88.0
    user_vocab: str = ""

    # Debug
    debug_dir: str = ""


def _ocr_text_in_box(
    image: Image.Image,
    box: List[float],
    ocr_params: Optional[OCRParams] = None
) -> str:
    """Run OCR on a cropped region"""
    if image is None:
        return ""
    if ocr_params is None:
        ocr_params = OCRParams()

    x1, y1, x2, y2 = box
    w, h = image.size

    # Expand crop to capture full text (padding instead of inset)
    pad_x = max(2, int((x2 - x1) * 0.04))
    pad_y = max(2, int((y2 - y1) * 0.04))
    left = max(int(x1) - pad_x, 0)
    top = max(int(y1) - pad_y, 0)
    right = min(int(x2) + pad_x, w - 1)
    bottom = min(int(y2) + pad_y, h - 1)

    if right <= left or bottom <= top:
        return ""

    crop = image.crop((left, top, right, bottom))
    crop = _preprocess_crop(
        crop,
        ocr_params.upscale,
        ocr_params.binarize,
        ocr_params.binarize_thr,
        ocr_params.denoise,
        ocr_params.noisy_mode,
        ocr_params.noisy_std
    )

    # Also try a looser crop to avoid cutting off characters
    loose_crop = None
    if ocr_params.loose_scale > 1.0:
        box_w = x2 - x1
        box_h = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        new_w = box_w * ocr_params.loose_scale
        new_h = box_h * ocr_params.loose_scale
        lx1 = max(int(cx - new_w / 2), 0)
        ly1 = max(int(cy - new_h / 2), 0)
        lx2 = min(int(cx + new_w / 2), w - 1)
        ly2 = min(int(cy + new_h / 2), h - 1)
        loose_crop = _preprocess_crop(
            image.crop((lx1, ly1, lx2, ly2)),
            ocr_params.upscale,
            ocr_params.binarize,
            ocr_params.binarize_thr,
            ocr_params.denoise,
            ocr_params.noisy_mode,
            ocr_params.noisy_std
        )

    # Save debug crops if requested
    if ocr_params.debug_dir:
        import os
        import uuid
        try:
            os.makedirs(ocr_params.debug_dir, exist_ok=True)
            uid = uuid.uuid4().hex[:8]
            crop.save(os.path.join(ocr_params.debug_dir, f"{uid}_crop.png"))
            if loose_crop:
                loose_crop.save(os.path.join(ocr_params.debug_dir, f"{uid}_loose.png"))
        except Exception:
            pass

    tess_text, tess_conf = _tesseract_ocr(
        crop,
        ocr_params.lang,
        ocr_params.psm_list,
        ocr_params.oem,
        ocr_params.tesseract_config,
        ocr_params.tesseract_whitelist,
        ocr_params.tesseract_whitelist_lang
    )

    # Try loose crop and use if better
    if loose_crop is not None:
        loose_text, loose_conf = _tesseract_ocr(
            loose_crop,
            ocr_params.lang,
            ocr_params.psm_list,
            ocr_params.oem,
            ocr_params.tesseract_config,
            ocr_params.tesseract_whitelist,
            ocr_params.tesseract_whitelist_lang
        )
        if loose_conf > tess_conf or (loose_conf == tess_conf and len(loose_text) > len(tess_text)):
            tess_text, tess_conf = loose_text, loose_conf

    if ocr_params.fix_typos:
        tess_text = _fix_ocr_typos(
            tess_text,
            ocr_params.spellcheck,
            ocr_params.spellcheck_topn,
            ocr_params.spellcheck_min_zipf,
            ocr_params.spellcheck_score
        )
    return tess_text


# --- Graph building ---

def _box_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def _suppress_overlaps(
    detections: List[Dict],
    iou_threshold: float,
    class_priority: Optional[Dict[str, int]] = None,
) -> List[Dict]:
    if iou_threshold <= 0:
        return detections
    priority = class_priority or {}

    def _rank(det: Dict) -> Tuple[int, float]:
        pr = priority.get(det.get("label"), 100)
        score = float(det.get("score", 0.0))
        return (pr, -score)

    candidates = sorted(detections, key=_rank)
    kept: List[Dict] = []
    for det in candidates:
        overlapped = False
        for k in kept:
            if _box_iou(det["box"], k["box"]) >= iou_threshold:
                overlapped = True
                break
        if not overlapped:
            kept.append(det)
    return kept


def _promote_similar_size_pools(
    objects: List[Dict],
    containers: List[Dict],
    size_ratio_threshold: float,
) -> List[Dict]:
    """
    Simple heuristic for non-BPMN diagrams:
    if a pool is similar in size to any task, treat it as a block.
    """
    if size_ratio_threshold <= 0 or not containers:
        return objects
    tasks = [o for o in objects if o.get("label") == "task"]
    if not tasks:
        return objects
    task_areas = []
    for t in tasks:
        x1, y1, x2, y2 = t["box"]
        task_areas.append(max(1.0, (x2 - x1)) * max(1.0, (y2 - y1)))
    promoted = list(objects)
    for cont in containers:
        if cont.get("label") != "pool":
            continue
        bx1, by1, bx2, by2 = cont["box"]
        area_b = max(1.0, (bx2 - bx1)) * max(1.0, (by2 - by1))
        for area_a in task_areas:
            ratio = min(area_a, area_b) / max(area_a, area_b)
            if ratio >= size_ratio_threshold:
                merged = dict(cont)
                merged["type"] = "object"
                merged["label"] = "block"
                promoted.append(merged)
                break
    # Deduplicate by box
    unique: Dict[tuple, Dict] = {}
    for obj in promoted:
        key = tuple(round(x, 2) for x in obj["box"])
        if key not in unique or float(obj.get("score", 0.0)) > float(unique[key].get("score", 0.0)):
            unique[key] = obj
    return list(unique.values())


def _merge_overlapping_blocks(
    objects: List[Dict],
    iou_threshold: float,
) -> List[Dict]:
    """
    Merge overlapping blocks by IoU, keeping the one with higher score.
    """
    if iou_threshold <= 0:
        return objects
    sorted_objs = sorted(objects, key=lambda d: float(d.get("score", 0.0)), reverse=True)
    kept: List[Dict] = []
    for obj in sorted_objs:
        overlapped = False
        for k in kept:
            if _box_iou(obj["box"], k["box"]) >= iou_threshold:
                overlapped = True
                break
        if not overlapped:
            kept.append(obj)
    return kept


def find_closest_node(
    point: List[float],
    nodes: Dict[str, BPMNNode],
    max_distance: float = 200
) -> Optional[str]:
    """Find node closest to a given point by center distance"""
    x, y = point
    closest_id = None
    closest_dist = float('inf')

    for node_id, node in nodes.items():
        dist = node.distance_to_point(x, y)
        if dist < closest_dist:
            closest_dist = dist
            closest_id = node_id

    if closest_dist <= max_distance:
        return closest_id
    return None


@dataclass
class GraphBuildParams:
    """Parameters for graph building"""
    connection_threshold: float = 200.0
    overlap_iou: float = 0.7
    arrow_fallback: bool = True

    # Simple mode (for diagrams without pools/lanes)
    simple_mode: bool = False
    simple_pool_task_size_ratio: float = 0.7
    simple_merge_iou: float = 0.6

    ocr: Optional[OCRParams] = None

    def __post_init__(self):
        if self.ocr is None:
            self.ocr = OCRParams()


def detections_to_graph(
    detections: List[Dict],
    image: Optional[Image.Image] = None,
    params: Optional[GraphBuildParams] = None
) -> BPMNGraph:
    """
    Convert raw detection results to a BPMNGraph.

    Args:
        detections: List of detection dictionaries from BPMN detection model
        image: PIL Image for OCR (optional)
        params: Graph building parameters (optional)

    Returns:
        BPMNGraph object with nodes, edges, and lanes
    """
    if params is None:
        params = GraphBuildParams()

    graph = BPMNGraph()

    CONTAINER_TYPES = {'lane', 'pool'}
    containers = [d for d in detections if d.get("type") == "object" and d.get("label") in CONTAINER_TYPES]
    arrows = [d for d in detections if d.get("type") == "arrow"]
    objects = [d for d in detections if d.get("type") == "object" and d.get("label") not in CONTAINER_TYPES]

    # Suppress overlaps
    class_priority = {
        "subProcess": 0, "task": 1,
        "event": 2, "messageEvent": 2, "timerEvent": 2,
        "exclusiveGateway": 3, "parallelGateway": 3, "eventBasedGateway": 3,
        "dataObject": 4, "dataStore": 4,
    }
    objects = _suppress_overlaps(objects, params.overlap_iou, class_priority=class_priority)

    # Simple mode: promote similar-size pools to blocks and merge overlapping
    if params.simple_mode:
        objects = _promote_similar_size_pools(
            objects, containers, params.simple_pool_task_size_ratio
        )
        objects = _merge_overlapping_blocks(objects, params.simple_merge_iou)

    # Create lanes/pools (skip in simple mode)
    if not params.simple_mode:
        for i, cont in enumerate(containers):
            name = f"{cont['label']}_{i}"
            if image is not None:
                ocr_text = _ocr_text_in_box(image, cont["box"], params.ocr)
                if ocr_text:
                    name = ocr_text
            lane = BPMNLane(
                id=f"lane_{i}",
                label=cont["label"],
                name=name,
                score=cont["score"],
                box=cont["box"]
            )
            graph.add_lane(lane)

    # Create nodes
    for i, obj in enumerate(objects):
        # In simple mode, rename labels to "block"
        if params.simple_mode:
            obj = dict(obj)
            obj["label"] = "block"

        name = f"{obj['label']}_{i}"
        if image is not None:
            ocr_text = _ocr_text_in_box(image, obj["box"], params.ocr)
            if ocr_text:
                name = ocr_text
        node = BPMNNode(
            id=f"node_{i}",
            label=obj["label"],
            name=name,
            score=obj["score"],
            box=obj["box"]
        )

        # Find containing lane (skip in simple mode)
        if not params.simple_mode:
            containing_lanes = []
            for lane_id, lane in graph.lanes.items():
                if lane.contains_box(node.box):
                    lx1, ly1, lx2, ly2 = lane.box
                    area = (lx2 - lx1) * (ly2 - ly1)
                    containing_lanes.append((lane_id, area))

            if containing_lanes:
                containing_lanes.sort(key=lambda x: x[1])
                node.lane_id = containing_lanes[0][0]

        graph.add_node(node)

    # Create edges
    for i, arrow in enumerate(arrows):
        # In simple mode, rename arrow labels to "arrow"
        if params.simple_mode:
            arrow = dict(arrow)
            arrow["label"] = "arrow"

        keypoints = arrow.get("keypoints", [])
        source_id = None
        target_id = None

        if len(keypoints) >= 2:
            start_point = keypoints[0]
            end_point = keypoints[1]
            source_id = find_closest_node(start_point, graph.nodes, params.connection_threshold)
            target_id = find_closest_node(end_point, graph.nodes, params.connection_threshold)

            # Skip self-loops
            if source_id == target_id:
                source_id = None
                target_id = None

        edge = BPMNEdge(
            id=f"edge_{i}",
            label=arrow["label"],
            score=arrow["score"],
            box=arrow["box"],
            keypoints=keypoints,
            source_id=source_id,
            target_id=target_id
        )
        graph.add_edge(edge)

    return graph
