# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

BPMN Diagram-to-Graph conversion system for ML System Design track (Мегашкола 2026). Uses local ML models (no external API dependencies).

## Commands

### Web service (Docker)
```bash
docker compose up --build
# Frontend: http://localhost:3000, Backend: http://localhost:8000
```

First build downloads ~400MB of ML models.

### Frontend development
```bash
cd frontend && npm install
npm run dev      # Dev server with HMR, proxies /api to localhost:8000
npm run build    # tsc && vite build
```

### Backend development
```bash
cd backend && pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Note: Models are downloaded on first startup (~400MB from HuggingFace).

### Standalone scripts
```bash
pip install -r requirements_detection.txt
python bpmn_detection_demo.py      # Detect BPMN elements in images
python bpmn_to_graph.py            # Convert detections to graph
python batch_process_bpmn.py       # Batch process multiple images
```

No test suite or linter configured.

## Architecture

### Backend (FastAPI, Python 3.12)

ML-based BPMN detection pipeline using ELCA-SA/BPMN_Detection model from HuggingFace.

**Detection Pipeline:**
1. **Faster R-CNN** — objects (tasks, gateways, events, pools, lanes)
2. **Keypoint R-CNN** — arrows/flows with start/end keypoints
3. **Tesseract OCR** — text extraction with spell-checking (rapidfuzz + wordfreq)

**Key Files:**
- `backend/app/main.py` — FastAPI app with lifespan for model loading
- `backend/app/services/bpmn_detector.py` — Singleton model manager, async inference with ThreadPool
- `backend/app/services/bpmn_graph.py` — Detection-to-graph conversion, OCR processing
- `backend/app/models/schemas.py` — Pydantic schemas (13 configurable parameters)

**API:**
- `POST /api/convert` — Upload image, returns graph + CSV + metadata
- `GET /api/health` — Health check with model status

**Concurrency:** 4 parallel requests via asyncio.Semaphore + ThreadPoolExecutor

### Frontend (React 18, TypeScript, Vite, Tailwind, React Flow)

Two-screen SPA with Zustand state management.

**Screens:**
- `StartScreen` — Upload area, parameter configuration, history thumbnails
- `EditorScreen` — Three-panel layout with graph/JSON/table tabs

**Key Files:**
- `frontend/src/store/index.ts` — Zustand store (state, actions, history persistence)
- `frontend/src/screens/` — StartScreen.tsx, EditorScreen.tsx
- `frontend/src/components/editor/` — GraphEditor, JsonEditor, TableEditor, NodeDetailsPanel
- `frontend/src/components/start/` — UploadArea, ParametersPanel, StartHistoryList
- `frontend/src/utils/graphTransform.ts` — API response to React Flow format
- `frontend/src/types.ts` — TypeScript definitions

**State Flow:** File upload → Zustand stages file → POST /api/convert → Response stored in history (localStorage) → Navigate to EditorScreen

**Nginx** proxies `/api/*` to backend.

## Environment Variables

All optional with sensible defaults:
- `DETECTION_THRESHOLD` (default 0.5) — Confidence threshold
- `CONNECTION_THRESHOLD` (default 200) — Max distance to connect arrow to node
- `OCR_LANG` (default "eng+rus") — Tesseract languages

## Docker Notes

- Backend image includes pre-downloaded ML models (~400MB baked in)
- Memory limit: 4GB (models ~2GB + inference overhead)
- CPU-only PyTorch for smaller image size
- Tesseract with English and Russian language packs
