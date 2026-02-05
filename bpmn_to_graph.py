"""
BPMN Detection to Graph Converter

Converts BPMN detection results into a graph structure.
Uses keypoints from arrows to determine connections between objects.

Output formats:
- NetworkX graph object
- JSON adjacency list
- DOT format (for Graphviz visualization)
- CSV edge list
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict
import math

# Optional: networkx for graph operations
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not installed. Some features disabled.")
    print("Install with: pip install networkx")

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    print("Warning: pytesseract not installed. OCR is disabled.")
    print("Install with: pip install pytesseract (and tesseract binary).")

from PIL import Image, ImageOps, ImageFilter

try:
    from wordfreq import zipf_frequency, top_n_list
    from rapidfuzz import process, fuzz
    HAS_SPELL = True
except ImportError:
    HAS_SPELL = False

DEFAULT_CONFIG = {
    "ARROW_FALLBACK": "1",
    "CONNECTION_THRESHOLD": "200",
    "OVERLAP_IOU": "0.7",
    "SIMPLE_POOL_TASK_SIZE_RATIO": "0.7",
    "SIMPLE_MERGE_IOU": "0.6",
    "OCR_LANG": "eng+rus",
    "OCR_OEM": "1",
    "OCR_PSM_LIST": "6,7",
    "OCR_UPSCALE": "2",
    "OCR_BINARIZE": "0",
    "OCR_DENOISE": "1",
    "OCR_INSET_PX": "2",
    "OCR_INSET_PCT": "0.02",
    "OCR_LOOSE_SCALE": "1.1",
    "OCR_TESSERACT_CONFIG": "-c preserve_interword_spaces=1",
    "OCR_TESSERACT_WHITELIST_LANG": "0",
    "OCR_FIX_TYPOS": "1",
    "OCR_SPELLCHECK": "1",
    "OCR_SPELLCHECK_TOPN": "20000",
    "OCR_SPELLCHECK_MIN_ZIPF": "3.0",
    "OCR_SPELLCHECK_SCORE": "88",
    "OCR_USER_VOCAB": "",
    "OCR_DEBUG_DIR": "",
}


def _get_env(name: str) -> str:
    return os.getenv(name, DEFAULT_CONFIG.get(name, ""))


@dataclass
class BPMNLane:
    """Represents a BPMN lane/pool (container for grouping elements)"""
    id: str
    label: str  # 'lane' or 'pool'
    name: str
    score: float
    box: List[float]  # [x1, y1, x2, y2]

    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is inside this lane's bounding box"""
        x1, y1, x2, y2 = self.box
        return x1 <= x <= x2 and y1 <= y <= y2

    def contains_box(self, other_box: List[float], threshold: float = 0.5) -> bool:
        """Check if another box is mostly inside this lane (by center point)"""
        ox1, oy1, ox2, oy2 = other_box
        center_x = (ox1 + ox2) / 2
        center_y = (oy1 + oy2) / 2
        return self.contains_point(center_x, center_y)


@dataclass
class BPMNNode:
    """Represents a BPMN element (task, gateway, event, etc.)"""
    id: str
    label: str  # Element type: task, exclusiveGateway, event, etc.
    name: str   # Display name (auto-generated or from OCR)
    score: float
    box: List[float]  # [x1, y1, x2, y2]
    center: Tuple[float, float] = field(default=(0, 0))
    lane_id: Optional[str] = None  # ID of containing lane/pool

    def __post_init__(self):
        # Calculate center point
        x1, y1, x2, y2 = self.box
        self.center = ((x1 + x2) / 2, (y1 + y2) / 2)

    def contains_point(self, x: float, y: float, margin: float = 10) -> bool:
        """Check if point is inside this node's bounding box (with margin)"""
        x1, y1, x2, y2 = self.box
        return (x1 - margin <= x <= x2 + margin and
                y1 - margin <= y <= y2 + margin)

    def distance_to_point(self, x: float, y: float) -> float:
        """Calculate distance from node center to a point"""
        cx, cy = self.center
        return math.sqrt((x - cx) ** 2 + (y - cy) ** 2)


@dataclass
class BPMNEdge:
    """Represents a BPMN flow (sequenceFlow, messageFlow, dataAssociation)"""
    id: str
    label: str  # Flow type
    score: float
    box: List[float]
    keypoints: List[List[float]]  # [[x_start, y_start], [x_end, y_end]]
    source_id: Optional[str] = None
    target_id: Optional[str] = None


@dataclass
class BPMNGraph:
    """Graph representation of a BPMN diagram"""
    nodes: Dict[str, BPMNNode] = field(default_factory=dict)
    edges: List[BPMNEdge] = field(default_factory=list)
    lanes: Dict[str, BPMNLane] = field(default_factory=dict)  # Lanes/pools as containers
    image_size: Tuple[int, int] = (0, 0)  # (width, height) of source image

    def add_node(self, node: BPMNNode):
        self.nodes[node.id] = node

    def add_edge(self, edge: BPMNEdge):
        self.edges.append(edge)

    def add_lane(self, lane: BPMNLane):
        self.lanes[lane.id] = lane

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
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

    def to_json(self, indent: int = 2) -> str:
        """Export as JSON string"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_csv(self) -> str:
        """Export edges as CSV (source,target,type) - only connected edges"""
        lines = [
            "source_id,source_label,source_name,source_lane,"
            "target_id,target_label,target_name,target_lane,edge_type,score"
        ]
        for edge in self.edges:
            # Skip unconnected edges
            if not edge.source_id or not edge.target_id:
                continue
            src = self.nodes.get(edge.source_id)
            tgt = self.nodes.get(edge.target_id)
            src_label = src.label if src else "unknown"
            tgt_label = tgt.label if tgt else "unknown"
            src_name = src.name if src else ""
            tgt_name = tgt.name if tgt else ""
            src_lane = src.lane_id if src and src.lane_id else ""
            tgt_lane = tgt.lane_id if tgt and tgt.lane_id else ""
            lines.append(
                f"{edge.source_id},{src_label},{src_name},{src_lane},"
                f"{edge.target_id},{tgt_label},{tgt_name},{tgt_lane},"
                f"{edge.label},{edge.score}"
            )
        return "\n".join(lines)

    def to_dot(self, use_positions: bool = True) -> str:
        """
        Export as DOT format for Graphviz

        Args:
            use_positions: If True, position nodes at their detected locations
        """
        image_width, image_height = self.image_size if self.image_size[0] > 0 else (1000, 500)

        lines = ["digraph BPMN {"]

        if use_positions:
            # Use neato layout with fixed positions
            lines.append("  layout=neato;")
            lines.append("  overlap=false;")
            lines.append("  splines=true;")
        else:
            lines.append("  rankdir=LR;")

        lines.append("  node [shape=box, fontsize=10];")
        lines.append("  edge [fontsize=8];")
        lines.append("")

        # Node shapes based on BPMN type
        shapes = {
            "task": "box",
            "block": "box",
            "subProcess": "box, style=rounded",
            "exclusiveGateway": "diamond",
            "parallelGateway": "diamond",
            "eventBasedGateway": "diamond",
            "event": "circle",
            "messageEvent": "doublecircle",
            "timerEvent": "circle",
            "dataObject": "note",
            "dataStore": "cylinder",
        }

        # Scale factor to convert pixels to points (72 DPI)
        # Увеличен в 2 раза для лучшей читаемости
        scale = 2.0 / 72.0

        def escape_label(text: str) -> str:
            return text.replace("\\", "\\\\").replace('"', '\\"')

        def _wrap_label(text: str, width_in: float, height_in: float, font_size: int) -> str:
            if not text:
                return ""
            # Rough estimation: average char width ~ 0.6 * fontsize (pt)
            char_w = 0.6 * font_size
            line_h = 1.2 * font_size
            max_chars = max(1, int((width_in * 72) / char_w))
            max_lines = max(1, int((height_in * 72) / line_h))

            words = text.split()
            lines: List[str] = []
            current = words[0] if words else ""
            for word in words[1:]:
                trial = f"{current} {word}"
                if len(trial) <= max_chars:
                    current = trial
                else:
                    lines.append(current)
                    current = word
            if current:
                lines.append(current)

            # If a single word is too long, split by chars
            fixed: List[str] = []
            for line in lines:
                if len(line) <= max_chars:
                    fixed.append(line)
                    continue
                chunk = ""
                for ch in line:
                    if len(chunk) + 1 <= max_chars:
                        chunk += ch
                    else:
                        fixed.append(chunk)
                        chunk = ch
                if chunk:
                    fixed.append(chunk)
            lines = fixed[:max_lines]

            # Truncate if still too long
            if len(lines) == max_lines and fixed and len(fixed) > max_lines:
                lines[-1] = (lines[-1][:-1] + "…") if len(lines[-1]) > 1 else "…"

            return "\\n".join(lines)

        # Define all nodes with positions
        for node in self.nodes.values():
            shape = shapes.get(node.label, "box")
            cx, cy = node.center

            # Flip Y coordinate (image Y goes down, Graphviz Y goes up)
            cy_flipped = image_height - cy

            # Convert to inches for Graphviz
            pos_x = cx * scale
            pos_y = cy_flipped * scale

            # Calculate node size based on bounding box
            x1, y1, x2, y2 = node.box
            width = (x2 - x1) * scale
            height = (y2 - y1) * scale

            label_text = node.name if node.name else node.label
            label_text = escape_label(label_text)
            # Fit label into node box
            font_size = 10
            wrapped = _wrap_label(label_text, width, height, font_size)
            label_full = wrapped if wrapped else label_text
            label_full = f"{label_full}\\n({node.label})"

            if use_positions:
                pos_attr = f'pos="{pos_x:.2f},{pos_y:.2f}!"'
                size_attr = f'width={width:.2f}, height={height:.2f}, fixedsize=true'
                lines.append(
                    f'  "{node.id}" [label="{label_full}", shape={shape}, fontsize={font_size}, {pos_attr}, {size_attr}];'
                )
            else:
                lines.append(f'  "{node.id}" [label="{label_full}", shape={shape}, fontsize={font_size}];')

        lines.append("")

        # Define edges
        edge_styles = {
            "sequenceFlow": "",
            "messageFlow": "style=dashed, color=blue",
            "dataAssociation": "style=dotted, color=gray",
        }

        for edge in self.edges:
            if edge.source_id and edge.target_id:
                style = edge_styles.get(edge.label, "")
                lines.append(f'  "{edge.source_id}" -> "{edge.target_id}" [{style}];')

        lines.append("}")
        return "\n".join(lines)

    def to_networkx(self):
        """Convert to NetworkX DiGraph (if available)"""
        if not HAS_NETWORKX:
            raise ImportError("networkx is required. Install with: pip install networkx")

        G = nx.DiGraph()

        # Add nodes with attributes (including lane assignment)
        for node in self.nodes.values():
            G.add_node(node.id,
                      label=node.label,
                      name=node.name,
                      score=node.score,
                      box=node.box,
                      center=node.center,
                      lane_id=node.lane_id)

        # Add edges with attributes
        for edge in self.edges:
            if edge.source_id and edge.target_id:
                G.add_edge(edge.source_id, edge.target_id,
                          edge_type=edge.label,
                          score=edge.score,
                          keypoints=edge.keypoints)

        return G


def _apply_edge_overrides(graph: BPMNGraph, overrides_path: str) -> None:
    """
    Apply manual edge overrides to the graph.
    JSON format:
    {
      "remove": [["node_1", "node_2"], ...],
      "add": [["node_3", "node_4", "sequenceFlow"], ...]
    }
    """
    if not overrides_path:
        return
    try:
        with open(overrides_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return
    remove = data.get("remove", []) or []
    add = data.get("add", []) or []

    # Remove edges
    if remove:
        to_remove = {(src, dst) for src, dst in remove if src and dst}
        graph.edges = [
            e for e in graph.edges
            if not (e.source_id and e.target_id and (e.source_id, e.target_id) in to_remove)
        ]

    # Add edges
    if add:
        existing = {(e.source_id, e.target_id, e.label) for e in graph.edges if e.source_id and e.target_id}
        for item in add:
            if len(item) < 2:
                continue
            src, dst = item[0], item[1]
            label = item[2] if len(item) >= 3 else "sequenceFlow"
            if not src or not dst:
                continue
            if (src, dst, label) in existing:
                continue
            graph.edges.append(
                BPMNEdge(
                    id=f"edge_manual_{len(graph.edges)}",
                    label=label,
                    score=1.0,
                    box=[0.0, 0.0, 0.0, 0.0],
                    keypoints=[],
                    source_id=src,
                    target_id=dst,
                )
            )


def _contains_cyrillic(text: str) -> bool:
    for ch in text:
        if "А" <= ch <= "я" or ch == "Ё" or ch == "ё":
            return True
    return False


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


def _normalize_ocr_punct(text: str) -> str:
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = text.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("`", "'")
    # Add spaces between Cyrillic and Latin/digits when glued
    text = re.sub(r"([А-Яа-я])([A-Za-z0-9])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z0-9])([А-Яа-я])", r"\1 \2", text)
    # Collapse extra whitespace
    return " ".join(text.split()).strip()


def _script_ratio(word: str) -> Tuple[int, int]:
    lat = sum(1 for ch in word if "A" <= ch <= "Z" or "a" <= ch <= "z")
    cyr = sum(1 for ch in word if ("А" <= ch <= "я") or ch in {"Ё", "ё"})
    return lat, cyr


def _fix_word_by_script(word: str) -> str:
    if not word:
        return word
    lat, cyr = _script_ratio(word)
    if cyr > lat:
        # Mostly Cyrillic: map similar Latin letters to Cyrillic
        w = word.translate(_LATIN_TO_CYR)
        w = w.replace("|", "л").replace("0", "о").replace("1", "l")
        return w
    if lat > cyr:
        # Mostly Latin: map similar Cyrillic letters to Latin
        w = word.translate(_CYR_TO_LATIN)
        w = w.replace("|", "l").replace("0", "O")
        # Common OCR merge: rn -> m
        w = w.replace("rn", "m")
        return w
    return word


_WORDLISTS = {}
_USER_VOCAB = None


def _get_wordlist(lang: str) -> List[str]:
    if lang in _WORDLISTS:
        return _WORDLISTS[lang]
    topn = int(_get_env("OCR_SPELLCHECK_TOPN") or "20000")
    try:
        wl = top_n_list(lang, topn)
    except Exception:
        wl = []
    # Merge user vocab if provided
    vocab_path = _get_env("OCR_USER_VOCAB").strip()
    if vocab_path:
        global _USER_VOCAB
        if _USER_VOCAB is None:
            try:
                with open(vocab_path, "r", encoding="utf-8") as fh:
                    _USER_VOCAB = [line.strip() for line in fh if line.strip()]
            except Exception:
                _USER_VOCAB = []
        if _USER_VOCAB:
            wl = list(dict.fromkeys(_USER_VOCAB + wl))
    _WORDLISTS[lang] = wl
    return wl


def _match_case(src: str, tgt: str) -> str:
    if src.isupper():
        return tgt.upper()
    if src[:1].isupper():
        return tgt[:1].upper() + tgt[1:]
    return tgt


def _spell_correct_word(word: str, lang: str) -> str:
    if not HAS_SPELL:
        return word
    if len(word) < 4:
        return word
    if len(word) > 30:
        return word
    if any(ch.isdigit() for ch in word):
        return word
    min_zipf = float(_get_env("OCR_SPELLCHECK_MIN_ZIPF") or "3.0")
    score_cutoff = float(_get_env("OCR_SPELLCHECK_SCORE") or "88")
    low = word.lower()
    if zipf_frequency(low, lang) >= min_zipf:
        return word
    wl = _get_wordlist(lang)
    if not wl:
        return word
    match = process.extractOne(low, wl, scorer=fuzz.ratio, score_cutoff=score_cutoff)
    if not match:
        return word
    cand = match[0]
    if zipf_frequency(cand, lang) < min_zipf:
        return word
    return _match_case(word, cand)


def _fix_ocr_typos(text: str) -> str:
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
        # Only correct when script is clearly dominant
        if HAS_SPELL and _get_env("OCR_SPELLCHECK").lower() not in {"0", "false", "no"}:
            if cyr >= lat + 2:
                w = _spell_correct_word(w, "ru")
            elif lat >= cyr + 2:
                w = _spell_correct_word(w, "en")
        fixed.append(w)
    return "".join(fixed).strip()


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
    unique = {}
    for obj in promoted:
        key = tuple(round(x, 2) for x in obj["box"])
        if key not in unique or float(obj.get("score", 0.0)) > float(unique[key].get("score", 0.0)):
            unique[key] = obj
    return list(unique.values())


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


def _preprocess_crop(crop: Image.Image) -> Image.Image:
    # Light preprocessing to improve OCR on small labels.
    gray = crop.convert("L")
    gray = ImageOps.autocontrast(gray)

    # Detect noisy/low-contrast crops and apply stronger cleanup.
    noisy_mode = False
    if _get_env("OCR_NOISY_MODE").lower() not in {"0", "false", "no"}:
        try:
            hist = gray.histogram()
            total = sum(hist) or 1
            mean = sum(i * h for i, h in enumerate(hist)) / total
            var = sum(((i - mean) ** 2) * h for i, h in enumerate(hist)) / total
            std = var ** 0.5
            noisy_std = float(_get_env("OCR_NOISY_STD") or "35")
            noisy_mode = std > noisy_std
        except Exception:
            noisy_mode = False

    if _get_env("OCR_DENOISE").lower() not in {"0", "false", "no"}:
        if noisy_mode:
            gray = gray.filter(ImageFilter.MedianFilter(size=5))
            gray = gray.filter(ImageFilter.GaussianBlur(radius=0.6))
        else:
            gray = gray.filter(ImageFilter.MedianFilter(size=3))

    gray = gray.filter(ImageFilter.SHARPEN)

    # Upscale for better OCR on tiny fonts
    w, h = gray.size
    scale = int(_get_env("OCR_UPSCALE") or "2")
    scale = max(1, min(scale, 4))
    if _get_env("OCR_BINARIZE").lower() in {"1", "true", "yes"}:
        # Simple binarization can help on low-contrast labels
        thr = int(_get_env("OCR_BINARIZE_THR") or "170")
        gray = gray.point(lambda p: 255 if p > thr else 0)
    return gray.resize((max(1, w * scale), max(1, h * scale)), Image.Resampling.LANCZOS)


def _tesseract_ocr(crop: Image.Image) -> Tuple[str, float]:
    if not HAS_TESSERACT:
        return "", 0.0
    lang = _get_env("OCR_LANG") or "eng+rus"
    psm_list = (_get_env("OCR_PSM_LIST") or "6,7").split(",")
    oem = _get_env("OCR_OEM") or "1"
    base_cfg = _get_env("OCR_TESSERACT_CONFIG") or "-c preserve_interword_spaces=1"
    whitelist = os.getenv("OCR_TESSERACT_WHITELIST", "").strip()
    if not whitelist and _get_env("OCR_TESSERACT_WHITELIST_LANG") in {"1", "true", "yes"}:
        whitelist = (
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
            "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
            " "
        )
    best_text = ""
    best_conf = -1.0
    for psm in psm_list:
        psm = psm.strip()
        if not psm:
            continue
        try:
            cfg = f"--oem {oem} --psm {psm} {base_cfg}".strip()
            if whitelist:
                cfg += f" -c tessedit_char_whitelist={whitelist}"
            data = pytesseract.image_to_data(
                crop, lang=lang, config=cfg, output_type=pytesseract.Output.DICT
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
            cfg = f"--oem {oem} --psm {psm} {base_cfg}".strip()
            if whitelist:
                cfg += f" -c tessedit_char_whitelist={whitelist}"
            text = pytesseract.image_to_string(crop, lang=lang, config=cfg)
            text = " ".join(text.split()).strip()
            if len(text) > len(best_text):
                best_text, best_conf = text, best_conf
    return best_text, best_conf


def _ocr_text_in_box(image, box: List[float]) -> str:
    """
    Run OCR on a cropped region (bbox). Returns cleaned text or empty string.
    """
    if image is None:
        return ""
    x1, y1, x2, y2 = box
    w, h = image.size
    # Inset crop to reduce border noise
    inset_px = int(_get_env("OCR_INSET_PX") or "2")
    inset_pct = float(_get_env("OCR_INSET_PCT") or "0.02")
    inset_x = max(inset_px, int((x2 - x1) * inset_pct))
    inset_y = max(inset_px, int((y2 - y1) * inset_pct))
    left = max(int(x1) + inset_x, 0)
    top = max(int(y1) + inset_y, 0)
    right = min(int(x2) - inset_x, w - 1)
    bottom = min(int(y2) - inset_y, h - 1)
    if right <= left or bottom <= top:
        return ""
    crop = image.crop((left, top, right, bottom))
    crop = _preprocess_crop(crop)
    # Also try a looser crop to avoid cutting off characters
    loose_scale = float(_get_env("OCR_LOOSE_SCALE") or "1.1")
    if loose_scale > 1.0:
        box_w = x2 - x1
        box_h = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        new_w = box_w * loose_scale
        new_h = box_h * loose_scale
        lx1 = max(int(cx - new_w / 2), 0)
        ly1 = max(int(cy - new_h / 2), 0)
        lx2 = min(int(cx + new_w / 2), w - 1)
        ly2 = min(int(cy + new_h / 2), h - 1)
        loose_crop = _preprocess_crop(image.crop((lx1, ly1, lx2, ly2)))
    else:
        loose_crop = None
    debug_dir = _get_env("OCR_DEBUG_DIR").strip()
    if debug_dir:
        try:
            os.makedirs(debug_dir, exist_ok=True)
            idx = len([p for p in os.listdir(debug_dir) if p.endswith(".png")])
            crop.save(os.path.join(debug_dir, f"crop_{idx:04d}.png"))
        except Exception:
            pass

    tess_text, tess_conf = _tesseract_ocr(crop)
    if loose_crop is not None:
        loose_text, loose_conf = _tesseract_ocr(loose_crop)
        if loose_conf > tess_conf or (loose_conf == tess_conf and len(loose_text) > len(tess_text)):
            tess_text, tess_conf = loose_text, loose_conf
    if _get_env("OCR_FIX_TYPOS").lower() not in {"0", "false", "no"}:
        tess_text = _fix_ocr_typos(tess_text)
    return tess_text


def find_closest_node(point: List[float], nodes: Dict[str, BPMNNode],
                      max_distance: float = 200) -> Optional[str]:
    """
    Find the node closest to a given point by center distance.

    Args:
        point: [x, y] coordinates
        nodes: Dictionary of BPMNNode objects (must be objects, not arrows)
        max_distance: Maximum distance to consider a connection

    Returns:
        Node ID or None if no suitable node found
    """
    x, y = point

    # Always find closest node by center distance
    closest_id = None
    closest_dist = float('inf')

    for node_id, node in nodes.items():
        dist = node.distance_to_point(x, y)
        if dist < closest_dist:
            closest_dist = dist
            closest_id = node_id

    # Return only if within max distance
    if closest_dist <= max_distance:
        return closest_id
    return None


def _distance_point_to_box(point: List[float], box: List[float]) -> float:
    px, py = point
    x1, y1, x2, y2 = box
    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)
    return (dx * dx + dy * dy) ** 0.5


def _fallback_connect_by_bbox(
    arrow_box: List[float],
    nodes: Dict[str, BPMNNode],
    max_distance: float = 200,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Fallback link restoration based on arrow bbox geometry:
    - horizontal: connect nearest node on the left/right
    - vertical: connect nearest node on the top/bottom
    """
    if not nodes:
        return None, None
    ax1, ay1, ax2, ay2 = arrow_box
    aw = max(1.0, ax2 - ax1)
    ah = max(1.0, ay2 - ay1)
    horizontal = aw >= ah
    cx = (ax1 + ax2) / 2
    cy = (ay1 + ay2) / 2

    left_candidates = []
    right_candidates = []
    top_candidates = []
    bottom_candidates = []

    for node_id, node in nodes.items():
        nx, ny = node.center
        dist = _distance_point_to_box([nx, ny], arrow_box)
        if dist > max_distance:
            continue
        if nx <= cx:
            left_candidates.append((dist, node_id))
        if nx >= cx:
            right_candidates.append((dist, node_id))
        if ny <= cy:
            top_candidates.append((dist, node_id))
        if ny >= cy:
            bottom_candidates.append((dist, node_id))

    if horizontal:
        source = min(left_candidates, default=(None, None))[1]
        target = min(right_candidates, default=(None, None))[1]
    else:
        source = min(top_candidates, default=(None, None))[1]
        target = min(bottom_candidates, default=(None, None))[1]
    if source == target:
        return None, None
    return source, target


def detections_to_graph(detections: List[Dict],
                        connection_threshold: float = 200,
                        image=None,
                        simple_mode: bool = False) -> BPMNGraph:
    """
    Convert raw detection results to a BPMNGraph.

    Args:
        detections: List of detection dictionaries from BPMN detection model
        connection_threshold: Max distance to connect arrow endpoint to node

    Returns:
        BPMNGraph object with nodes and edges
    """
    graph = BPMNGraph()

    # Separate objects, lanes/pools, and arrows
    CONTAINER_TYPES = {'lane', 'pool'}
    containers = [d for d in detections if d.get("type") == "object" and d.get("label") in CONTAINER_TYPES]
    arrows = [d for d in detections if d.get("type") == "arrow"]

    objects = [d for d in detections if d.get("type") == "object" and d.get("label") not in CONTAINER_TYPES]
    # Suppress overlaps across classes (e.g., task vs subprocess)
    iou_thr = float(_get_env("OVERLAP_IOU") or "0.7")
    class_priority = {
        "subProcess": 0,
        "task": 1,
        "event": 2,
        "messageEvent": 2,
        "timerEvent": 2,
        "exclusiveGateway": 3,
        "parallelGateway": 3,
        "eventBasedGateway": 3,
        "dataObject": 4,
        "dataStore": 4,
    }
    objects = _suppress_overlaps(objects, iou_thr, class_priority=class_priority)
    if simple_mode:
        size_ratio = float(_get_env("SIMPLE_POOL_TASK_SIZE_RATIO") or "0.7")
        merge_iou = float(_get_env("SIMPLE_MERGE_IOU") or "0.6")
        objects = _promote_similar_size_pools(objects, containers, size_ratio)
        objects = _merge_overlapping_blocks(objects, merge_iou)

    # Create lanes/pools as containers (not graph nodes)
    if not simple_mode:
        for i, cont in enumerate(containers):
            name = f"{cont['label']}_{i}"
            if image is not None:
                ocr_text = _ocr_text_in_box(image, cont["box"])
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

    # Create nodes from objects (excluding lanes/pools)
    for i, obj in enumerate(objects):
        if simple_mode:
            obj = dict(obj)
            obj["label"] = "block"
        name = f"{obj['label']}_{i}"
        if image is not None:
            ocr_text = _ocr_text_in_box(image, obj["box"])
            if ocr_text:
                name = ocr_text
        node = BPMNNode(
            id=f"node_{i}",
            label=obj["label"],
            name=name,
            score=obj["score"],
            box=obj["box"]
        )

        # Find which lane this node belongs to (smallest containing lane)
        if not simple_mode:
            containing_lanes = []
            for lane_id, lane in graph.lanes.items():
                if lane.contains_box(node.box):
                    # Calculate lane area for finding smallest
                    lx1, ly1, lx2, ly2 = lane.box
                    area = (lx2 - lx1) * (ly2 - ly1)
                    containing_lanes.append((lane_id, area))

            # Assign to smallest containing lane (most specific)
            if containing_lanes:
                containing_lanes.sort(key=lambda x: x[1])  # Sort by area
                node.lane_id = containing_lanes[0][0]

        graph.add_node(node)

    # Create edges from arrows and match to nodes (not lanes!)
    for i, arrow in enumerate(arrows):
        keypoints = arrow.get("keypoints", [])
        if simple_mode:
            arrow = dict(arrow)
            arrow["label"] = "arrow"

        # Find source and target nodes
        source_id = None
        target_id = None

        if len(keypoints) >= 2:
            start_point = keypoints[0]
            end_point = keypoints[1]

            source_id = find_closest_node(start_point, graph.nodes, connection_threshold)
            target_id = find_closest_node(end_point, graph.nodes, connection_threshold)

            # Skip self-loops (both endpoints closest to same node)
            if source_id == target_id:
                source_id = None
                target_id = None

        # Fallback: restore links by arrow bbox if keypoints failed
        if (source_id is None or target_id is None) and _get_env("ARROW_FALLBACK") != "0":
            fb_source, fb_target = _fallback_connect_by_bbox(
                arrow["box"], graph.nodes, connection_threshold
            )
            source_id = source_id or fb_source
            target_id = target_id or fb_target

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


def run_detection_and_convert(image_path: str,
                              threshold: float = 0.5,
                              simple_mode: bool = False) -> BPMNGraph:
    """
    Run BPMN detection on an image and convert results to graph.

    Args:
        image_path: Path to input image
        threshold: Detection confidence threshold

    Returns:
        BPMNGraph object
    """
    # Import detection functions
    from bpmn_detection_demo import (
        load_models, detect_objects, detect_arrows
    )
    from PIL import Image

    # Load models (cached after first call)
    model_object, model_arrow, device = load_models()

    # Load image
    image = Image.open(image_path).convert("RGB")
    image_size = image.size  # (width, height)

    # Run detection
    object_detections = detect_objects(image, model_object, device, threshold)
    arrow_detections = detect_arrows(image, model_arrow, device, threshold)

    # Combine and convert to graph
    all_detections = object_detections + arrow_detections
    conn_thr = float(_get_env("CONNECTION_THRESHOLD") or "200")
    graph = detections_to_graph(
        all_detections,
        connection_threshold=conn_thr,
        image=image,
        simple_mode=simple_mode
    )
    overrides_path = os.getenv("EDGE_OVERRIDES", "").strip()
    if overrides_path:
        _apply_edge_overrides(graph, overrides_path)
    graph.image_size = image_size
    return graph


def main():
    """Demo: Convert BPMN detection results to graph"""
    import argparse

    parser = argparse.ArgumentParser(description="Convert BPMN detection to graph")
    parser.add_argument("image", nargs="?", help="Input image path")
    parser.add_argument("--format", "-f", choices=["json", "csv", "dot", "all"],
                       default="all", help="Output format")
    parser.add_argument("--render-png", action="store_true",
                       help="Render DOT to PNG (requires graphviz 'dot')")
    parser.add_argument("--output", "-o", help="Output file path (without extension)")
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                       help="Detection confidence threshold")
    parser.add_argument("--simple-diagram", action="store_true",
                       help="Simplify to blocks + arrows only (ignore pools/lanes)")
    args = parser.parse_args()

    # If no image specified, use first image from bpmn_diagrams/png
    if args.image:
        image_path = args.image
    else:
        png_dir = Path("bpmn_diagrams/png")
        images = sorted(png_dir.glob("*.png"))
        if not images:
            print("No images found. Please specify an image path.")
            return
        image_path = str(images[0])
        print(f"Using default image: {image_path}")

    print(f"\nProcessing: {image_path}")
    print("=" * 60)

    # Run detection and convert to graph
    graph = run_detection_and_convert(
        image_path,
        args.threshold,
        simple_mode=args.simple_diagram
    )

    # Print summary
    print(f"\nGraph Summary:")
    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Edges: {len(graph.edges)}")
    print(f"  Lanes/Pools: {len(graph.lanes)}")

    # Count connected vs unconnected edges
    connected = sum(1 for e in graph.edges if e.source_id and e.target_id)
    print(f"  Connected edges: {connected}/{len(graph.edges)}")

    # Count nodes by lane
    nodes_in_lanes = sum(1 for n in graph.nodes.values() if n.lane_id)
    print(f"  Nodes in lanes: {nodes_in_lanes}/{len(graph.nodes)}")

    # Node type distribution
    type_counts = {}
    for node in graph.nodes.values():
        type_counts[node.label] = type_counts.get(node.label, 0) + 1
    print(f"\n  Node types:")
    for label, count in sorted(type_counts.items()):
        print(f"    - {label}: {count}")

    # Edge type distribution
    edge_counts = {}
    for edge in graph.edges:
        edge_counts[edge.label] = edge_counts.get(edge.label, 0) + 1
    print(f"\n  Edge types:")
    for label, count in sorted(edge_counts.items()):
        print(f"    - {label}: {count}")

    # Output
    output_base = args.output or Path(image_path).stem + "_graph"
    output_dir = Path("bpmn_graphs")
    output_dir.mkdir(exist_ok=True)

    if args.format in ["json", "all"]:
        json_path = output_dir / f"{output_base}.json"
        json_path.write_text(graph.to_json())
        print(f"\n✓ JSON saved: {json_path}")

    if args.format in ["csv", "all"]:
        csv_path = output_dir / f"{output_base}.csv"
        csv_path.write_text(graph.to_csv())
        print(f"✓ CSV saved: {csv_path}")

    if args.format in ["dot", "all"]:
        dot_path = output_dir / f"{output_base}.dot"
        dot_path.write_text(graph.to_dot())
        print(f"✓ DOT saved: {dot_path}")
        dot_bin = shutil.which("dot")
        if args.render_png and dot_bin:
            png_path = output_dir / f"{output_base}_viz.png"
            try:
                subprocess.run(
                    [dot_bin, "-Tpng", str(dot_path), "-o", str(png_path)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                print(f"✓ Graphviz PNG saved: {png_path}")
            except subprocess.CalledProcessError as exc:
                print("Warning: Graphviz render failed.")
                print(exc.stderr.strip() or exc.stdout.strip())
        else:
            print(f"  (Visualize with: dot -Tpng {dot_path} -o {output_base}_viz.png)")

    # Print sample output
    print("\n" + "=" * 60)
    print("Sample JSON output:")
    print("-" * 60)
    sample = graph.to_dict()
    sample["nodes"] = sample["nodes"][:2]  # Show first 2 nodes
    sample["edges"] = sample["edges"][:2]  # Show first 2 edges
    print(json.dumps(sample, indent=2))

    print("\n" + "=" * 60)
    print("Sample CSV output:")
    print("-" * 60)
    csv_lines = graph.to_csv().split("\n")[:5]
    print("\n".join(csv_lines))

    if HAS_NETWORKX:
        print("\n" + "=" * 60)
        print("NetworkX graph info:")
        print("-" * 60)
        G = graph.to_networkx()
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")
        print(f"  Is DAG: {nx.is_directed_acyclic_graph(G)}")

        # Find start/end nodes (no incoming/outgoing edges)
        start_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
        end_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]
        print(f"  Start nodes (no incoming): {len(start_nodes)}")
        print(f"  End nodes (no outgoing): {len(end_nodes)}")


if __name__ == "__main__":
    main()
