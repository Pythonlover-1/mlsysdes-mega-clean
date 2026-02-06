import { create } from 'zustand';
import type {
  Screen,
  EditorTab,
  EditorDocument,
  StagedFile,
  ConversionResult,
  ConversionResponse,
  BPMNGraph,
  ConversionParams,
} from '../types';
import { DEFAULT_CONVERSION_PARAMS } from '../types';

const HISTORY_KEY = 'csv-history';
const MAX_HISTORY = 10;

function isValidHistoryItem(item: unknown): item is ConversionResult {
  if (!item || typeof item !== 'object') return false;
  const obj = item as Record<string, unknown>;
  if (typeof obj.timestamp !== 'number') return false;
  if (typeof obj.filename !== 'string') return false;
  if (typeof obj.imageDataUrl !== 'string') return false;
  if (!obj.graph || typeof obj.graph !== 'object') return false;
  const graph = obj.graph as Record<string, unknown>;
  if (!Array.isArray(graph.nodes)) return false;
  if (!Array.isArray(graph.edges)) return false;
  return true;
}

function loadHistory(): ConversionResult[] {
  try {
    const items = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
    // Filter out invalid/corrupted items
    return Array.isArray(items) ? items.filter(isValidHistoryItem) : [];
  } catch {
    return [];
  }
}

function saveHistory(history: ConversionResult[]) {
  const items = history.slice(0, MAX_HISTORY);
  for (let i = items.length; i >= 1; i--) {
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, i)));
      return;
    } catch {
      // quota exceeded — try with fewer items
    }
  }
  // nothing fits — clear
  localStorage.removeItem(HISTORY_KEY);
}

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// Image node position stored per document (by timestamp)
type ImageNodePositions = Record<number, { x: number; y: number }>;

interface AppStore {
  // Screen routing
  screen: Screen;
  navigateTo: (screen: Screen) => void;

  // Staged file (before conversion)
  stagedFile: StagedFile | null;
  stageFile: (file: File) => Promise<void>;
  clearStagedFile: () => void;

  // Conversion
  isConverting: boolean;
  conversionError: string | null;
  conversionParams: ConversionParams;
  setConversionParams: (params: Partial<ConversionParams>) => void;
  resetConversionParams: () => void;
  startConversion: () => Promise<void>;

  // Current document (in editor)
  currentDocument: EditorDocument | null;
  updateGraph: (graph: BPMNGraph) => void;

  // History
  history: ConversionResult[];
  loadFromHistory: (item: ConversionResult) => void;
  clearHistory: () => void;

  // Editor state
  activeTab: EditorTab;
  setActiveTab: (tab: EditorTab) => void;
  selectedNodeId: string | null;
  setSelectedNodeId: (id: string | null) => void;

  // Image node position (per document timestamp)
  imageNodePositions: ImageNodePositions;
  setImageNodePosition: (timestamp: number, position: { x: number; y: number }) => void;
  getImageNodePosition: (timestamp: number) => { x: number; y: number } | null;

  // Back to start
  backToStart: () => void;
}

export const useAppStore = create<AppStore>((set, get) => ({
  // Screen routing
  screen: 'start',
  navigateTo: (screen) => set({ screen }),

  // Staged file
  stagedFile: null,
  stageFile: async (file) => {
    const dataUrl = await fileToDataUrl(file);
    set({
      stagedFile: { file, dataUrl, name: file.name },
      conversionError: null,
    });
  },
  clearStagedFile: () => set({ stagedFile: null }),

  // Conversion
  isConverting: false,
  conversionError: null,
  conversionParams: { ...DEFAULT_CONVERSION_PARAMS },
  setConversionParams: (params) =>
    set((state) => ({
      conversionParams: { ...state.conversionParams, ...params },
    })),
  resetConversionParams: () => set({ conversionParams: { ...DEFAULT_CONVERSION_PARAMS } }),
  startConversion: async () => {
    const { stagedFile, history, conversionParams } = get();
    if (!stagedFile) return;

    set({ isConverting: true, conversionError: null });

    const formData = new FormData();
    formData.append('file', stagedFile.file);

    // Build query params from conversionParams
    const queryParams = new URLSearchParams();
    for (const [key, value] of Object.entries(conversionParams)) {
      queryParams.append(key, String(value));
    }

    try {
      const res = await fetch(`/api/convert?${queryParams.toString()}`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.detail || `Error ${res.status}`);
      }

      const data: ConversionResponse = await res.json();
      const timestamp = Date.now();

      // Create history entry
      const historyEntry: ConversionResult = {
        graph: data.graph,
        csv: data.csv,
        filename: data.filename,
        imageDataUrl: stagedFile.dataUrl,
        timestamp,
        image_size: data.image_size,
        processing_time_ms: data.processing_time_ms,
      };

      // Update history
      const newHistory = [historyEntry, ...history];
      saveHistory(newHistory);

      // Create editor document
      const document: EditorDocument = {
        graph: JSON.parse(JSON.stringify(data.graph)), // Deep clone for editing
        imageDataUrl: stagedFile.dataUrl,
        imageSize: data.image_size,
        filename: data.filename,
        timestamp,
        processingTimeMs: data.processing_time_ms,
      };

      set({
        isConverting: false,
        currentDocument: document,
        history: newHistory,
        stagedFile: null,
        screen: 'editor',
        activeTab: 'graph',
        selectedNodeId: null,
      });
    } catch (e) {
      set({
        isConverting: false,
        conversionError: e instanceof Error ? e.message : 'Unknown error',
      });
    }
  },

  // Current document
  currentDocument: null,
  updateGraph: (graph) => {
    const { currentDocument } = get();
    if (!currentDocument) return;
    set({
      currentDocument: { ...currentDocument, graph },
    });
  },

  // History
  history: loadHistory(),
  loadFromHistory: (item) => {
    // Validate item before loading
    if (!item.graph?.nodes || !item.graph?.edges) {
      console.error('Invalid history item:', item);
      return;
    }

    // Ensure lanes array exists
    const graph = {
      ...item.graph,
      lanes: item.graph.lanes || [],
    };

    const document: EditorDocument = {
      graph: JSON.parse(JSON.stringify(graph)), // Deep clone
      imageDataUrl: item.imageDataUrl,
      imageSize: item.image_size || [800, 600],
      filename: item.filename,
      timestamp: item.timestamp,
      processingTimeMs: item.processing_time_ms || 0,
    };
    set({
      currentDocument: document,
      screen: 'editor',
      activeTab: 'graph',
      selectedNodeId: null,
    });
  },
  clearHistory: () => {
    localStorage.removeItem(HISTORY_KEY);
    set({ history: [] });
  },

  // Editor state
  activeTab: 'graph',
  setActiveTab: (tab) => set({ activeTab: tab }),
  selectedNodeId: null,
  setSelectedNodeId: (id) => set({ selectedNodeId: id }),

  // Image node positions (per document)
  imageNodePositions: {},
  setImageNodePosition: (timestamp, position) =>
    set((state) => ({
      imageNodePositions: {
        ...state.imageNodePositions,
        [timestamp]: position,
      },
    })),
  getImageNodePosition: (timestamp) => {
    const { imageNodePositions } = get();
    return imageNodePositions[timestamp] || null;
  },

  // Back to start
  backToStart: () =>
    set({
      screen: 'start',
      currentDocument: null,
      selectedNodeId: null,
      stagedFile: null,
    }),
}));
