// BPMN Graph types (from API response)
export interface BPMNNode {
  id: string;
  label: string;  // Element type: task, exclusiveGateway, event, etc.
  name: string;   // OCR text or auto-generated name
  score: number;
  box: number[];  // [x1, y1, x2, y2]
  center: number[];  // [cx, cy]
  lane_id: string | null;
}

export interface BPMNEdge {
  id: string;
  label: string;  // Flow type: sequenceFlow, messageFlow, dataAssociation
  score: number;
  source: string | null;
  target: string | null;
  keypoints: number[][];
}

export interface BPMNLane {
  id: string;
  label: string;
  name: string;
  score: number;
  box: number[];
}

export interface BPMNGraph {
  nodes: BPMNNode[];
  edges: BPMNEdge[];
  lanes: BPMNLane[];
}

// API response
export interface ConversionResponse {
  graph: BPMNGraph;
  csv: string;
  filename: string;
  image_size: number[];
  processing_time_ms: number;
}

// Stored conversion result (in history)
export interface ConversionResult {
  graph: BPMNGraph;
  csv: string;
  filename: string;
  imageDataUrl: string;
  timestamp: number;
  image_size: number[];
  processing_time_ms: number;
}

// Staged file before conversion
export interface StagedFile {
  file: File;
  dataUrl: string;
  name: string;
}

// Current document being edited
export interface EditorDocument {
  graph: BPMNGraph;
  imageDataUrl: string;
  imageSize: number[];
  filename: string;
  timestamp: number;
  processingTimeMs: number;
}

// Active editor tab
export type EditorTab = 'graph' | 'json' | 'table';

// Screen type
export type Screen = 'start' | 'editor';

// Conversion parameters (matches backend)
export interface ConversionParams {
  // Detection
  detection_threshold: number; // 0.0-1.0, default 0.5
  arrow_dilate: number; // 0-10, default 0 - dilate lines before arrow detection (px)

  // Graph building
  connection_threshold: number; // >=0, default 200
  overlap_iou: number; // 0.0-1.0, default 0.7
  arrow_fallback: boolean; // default true
  simple_mode: boolean; // default false - for diagrams without pools/lanes
  simple_pool_task_size_ratio: number; // 0.0-1.0, default 0.7
  simple_merge_iou: number; // 0.0-1.0, default 0.6

  // OCR preprocessing
  ocr_upscale: number; // 1-4, default 2
  ocr_binarize: boolean; // default false
  ocr_binarize_thr: number; // 0-255, default 170
  ocr_denoise: boolean; // default false (can blur small text)
  ocr_noisy_mode: boolean; // default false
  ocr_noisy_std: number; // >=0, default 35.0

  // OCR crop settings
  ocr_inset_px: number; // >=0, default 2
  ocr_inset_pct: number; // 0.0-0.5, default 0.02
  ocr_loose_scale: number; // 0.0-2.0, default 0.0 (0.0 = disabled, was causing text cutoff)

  // OCR lane/pool
  ocr_lane_top_pct: number; // 0.05-1.0, default 0.05 - after rotating lane 90° CW, keep top X%

  // OCR Tesseract
  ocr_lang: string; // default "eng+rus"
  ocr_psm_list: string; // default "6,7"
  ocr_oem: number; // 0-3, default 1
  ocr_tesseract_config: string; // default "-c preserve_interword_spaces=1"
  ocr_tesseract_whitelist_lang: boolean; // default false

  // OCR postprocessing
  ocr_fix_typos: boolean; // default true
  ocr_spellcheck: boolean; // default true
  ocr_spellcheck_topn: number; // >=1000, default 20000
  ocr_spellcheck_min_zipf: number; // >=0, default 3.0
  ocr_spellcheck_score: number; // 0-100, default 88.0
  ocr_user_vocab: string; // default ""
}

export const DEFAULT_CONVERSION_PARAMS: ConversionParams = {
  detection_threshold: 0.5,
  arrow_dilate: 0,
  connection_threshold: 200,
  overlap_iou: 0.7,
  arrow_fallback: true,
  simple_mode: false,
  simple_pool_task_size_ratio: 0.7,
  simple_merge_iou: 0.6,
  ocr_upscale: 2,
  ocr_binarize: false,
  ocr_binarize_thr: 170,
  ocr_denoise: false,
  ocr_noisy_mode: false,
  ocr_noisy_std: 35.0,
  ocr_inset_px: 2,
  ocr_inset_pct: 0.02,
  ocr_loose_scale: 0.0,
  ocr_lane_top_pct: 0.05,
  ocr_lang: 'eng+rus',
  ocr_psm_list: '6,7',
  ocr_oem: 1,
  ocr_tesseract_config: '-c preserve_interword_spaces=1',
  ocr_tesseract_whitelist_lang: false,
  ocr_fix_typos: true,
  ocr_spellcheck: true,
  ocr_spellcheck_topn: 20000,
  ocr_spellcheck_min_zipf: 3.0,
  ocr_spellcheck_score: 88.0,
  ocr_user_vocab: '',
};
