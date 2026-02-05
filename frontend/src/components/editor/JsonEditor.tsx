import { useState, useCallback, useEffect, useRef } from 'react';
import { useAppStore } from '../../store';
import type { BPMNGraph } from '../../types';

export default function JsonEditor() {
  const currentDocument = useAppStore((s) => s.currentDocument);
  const updateGraph = useAppStore((s) => s.updateGraph);

  const [jsonText, setJsonText] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isEdited, setIsEdited] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Sync JSON text with graph
  useEffect(() => {
    if (currentDocument) {
      setJsonText(JSON.stringify(currentDocument.graph, null, 2));
      setError(null);
      setIsEdited(false);
    }
  }, [currentDocument]);

  const handleChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setJsonText(e.target.value);
    setIsEdited(true);
    setError(null);
  }, []);

  const applyChanges = useCallback(() => {
    try {
      const parsed: BPMNGraph = JSON.parse(jsonText);

      // Basic validation
      if (!Array.isArray(parsed.nodes) || !Array.isArray(parsed.edges) || !Array.isArray(parsed.lanes)) {
        throw new Error('JSON must have nodes, edges, and lanes arrays');
      }

      updateGraph(parsed);
      setError(null);
      setIsEdited(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Invalid JSON');
    }
  }, [jsonText, updateGraph]);

  const resetChanges = useCallback(() => {
    if (currentDocument) {
      setJsonText(JSON.stringify(currentDocument.graph, null, 2));
      setError(null);
      setIsEdited(false);
    }
  }, [currentDocument]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      // Ctrl/Cmd + Enter to apply
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        applyChanges();
      }
      // Escape to reset
      if (e.key === 'Escape' && isEdited) {
        e.preventDefault();
        resetChanges();
      }
    },
    [applyChanges, resetChanges, isEdited]
  );

  const downloadJson = useCallback(() => {
    if (!currentDocument) return;
    const blob = new Blob([jsonText], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = currentDocument.filename.replace(/\.\w+$/, '') + '.json';
    a.click();
    URL.revokeObjectURL(url);
  }, [jsonText, currentDocument]);

  if (!currentDocument) {
    return (
      <div className="w-full h-full flex items-center justify-center text-slate-500">
        No document loaded
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Toolbar */}
      <div className="flex items-center justify-between p-4 border-b border-slate-700/50 bg-slate-800/30">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-600 to-purple-600 flex items-center justify-center">
            <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
            </svg>
          </div>
          <span className="text-white font-semibold">JSON Editor</span>
          {isEdited && (
            <span className="px-2 py-0.5 text-xs rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/30">
              Изменено
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {isEdited && (
            <>
              <button
                onClick={resetChanges}
                className="px-3 py-1.5 text-sm text-slate-400 hover:text-white hover:bg-slate-700/50 rounded-lg transition-all"
              >
                Отменить
              </button>
              <button
                onClick={applyChanges}
                className="px-3 py-1.5 text-sm bg-gradient-to-r from-violet-500 to-purple-500 text-white rounded-lg hover:from-violet-400 hover:to-purple-400 transition-all"
              >
                Применить (Ctrl+Enter)
              </button>
            </>
          )}
          <button
            onClick={downloadJson}
            className="px-3 py-1.5 text-sm text-slate-400 hover:text-white hover:bg-slate-700/50 rounded-lg transition-all flex items-center gap-1.5"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            Скачать
          </button>
        </div>
      </div>

      {/* Error display */}
      {error && (
        <div className="mx-4 mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg flex items-center gap-2">
          <svg className="w-4 h-4 text-red-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span className="text-sm text-red-300">{error}</span>
        </div>
      )}

      {/* Editor */}
      <div className="flex-1 p-4 overflow-hidden">
        <textarea
          ref={textareaRef}
          value={jsonText}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          spellCheck={false}
          className={`w-full h-full p-4 rounded-xl font-mono text-sm resize-none outline-none transition-all
            bg-slate-900/50 text-slate-300 border
            ${error ? 'border-red-500/50' : isEdited ? 'border-amber-500/50' : 'border-slate-700/50'}
            focus:border-violet-500/50 focus:ring-1 focus:ring-violet-500/20`}
          placeholder="JSON graph data..."
        />
      </div>

      {/* Status bar */}
      <div className="px-4 py-2 border-t border-slate-700/50 bg-slate-800/30 text-xs text-slate-500 flex items-center justify-between">
        <span>
          {currentDocument.graph.nodes.length} nodes, {currentDocument.graph.edges.length} edges
        </span>
        <span>Ctrl+Enter для применения, Escape для отмены</span>
      </div>
    </div>
  );
}
