import { useState } from 'react';
import { useAppStore } from '../store';
import TabBar from '../components/editor/TabBar';
import HistoryPanel from '../components/editor/HistoryPanel';
import NodeDetailsPanel from '../components/editor/NodeDetailsPanel';
import GraphEditor from '../components/editor/GraphEditor';
import JsonEditor from '../components/editor/JsonEditor';
import TableEditor from '../components/editor/TableEditor';

export default function EditorScreen() {
  const activeTab = useAppStore((s) => s.activeTab);
  const currentDocument = useAppStore((s) => s.currentDocument);
  const backToStart = useAppStore((s) => s.backToStart);
  const [showHistory, setShowHistory] = useState(true);
  const [showDetails, setShowDetails] = useState(true);

  if (!currentDocument) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-center space-y-4">
          <p className="text-slate-500">No document loaded</p>
          <button
            onClick={backToStart}
            className="px-4 py-2 bg-white text-black rounded-lg font-medium hover:bg-gray-100 transition-colors"
          >
            Back to Start
          </button>
        </div>
      </div>
    );
  }

  const isGraph = activeTab === 'graph';

  if (!isGraph) {
    // Classic layout for JSON and Table tabs
    return (
      <div className="h-screen bg-black flex flex-col overflow-hidden">
        {/* Top Bar */}
        <div className="flex-shrink-0 p-4 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={backToStart}
              className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
              title="Назад к началу"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
            </button>
            <div>
              <h1 className="text-white font-semibold">{currentDocument.filename}</h1>
              <p className="text-xs text-slate-500">
                {currentDocument.graph.nodes.length} elements, {currentDocument.graph.edges.length} connections
              </p>
            </div>
          </div>
          <TabBar />
          <div className="text-right text-xs text-slate-500">
            <p>{currentDocument.processingTimeMs}ms</p>
            <p>
              {currentDocument.imageSize[0]} × {currentDocument.imageSize[1]}
            </p>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 p-4 overflow-hidden">
          <div className="h-full rounded-2xl border border-slate-700/50 bg-gradient-to-br from-slate-900 to-slate-800 overflow-hidden">
            {activeTab === 'json' && <JsonEditor />}
            {activeTab === 'table' && <TableEditor />}
          </div>
        </div>
      </div>
    );
  }

  // Fullscreen layout for Graph tab
  return (
    <div className="h-screen bg-black relative overflow-hidden">
      {/* Graph Editor - Full Screen */}
      <div className="absolute inset-0">
        <GraphEditor />
      </div>

      {/* Top Bar (overlay) */}
      <div className="absolute top-0 left-0 right-0 z-10 p-4 flex items-center justify-between pointer-events-none">
        {/* Back button and filename */}
        <div className="flex items-center gap-3 pointer-events-auto rounded-2xl border border-slate-700/50 bg-slate-900/80 backdrop-blur-sm px-3 py-2 shadow-xl">
          <button
            onClick={backToStart}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-700 transition-colors"
            title="Назад к началу"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
          </button>
          <div>
            <h1 className="text-white font-semibold text-sm">{currentDocument.filename}</h1>
            <p className="text-[10px] text-slate-500">
              {currentDocument.graph.nodes.length} elements, {currentDocument.graph.edges.length} connections
            </p>
          </div>
        </div>

        {/* Tab Bar - centered */}
        <div className="pointer-events-auto">
          <TabBar />
        </div>

        {/* Processing info */}
        <div className="pointer-events-auto rounded-2xl border border-slate-700/50 bg-slate-900/80 backdrop-blur-sm px-3 py-2 shadow-xl text-right text-xs text-slate-500">
          <p>{currentDocument.processingTimeMs}ms</p>
          <p>
            {currentDocument.imageSize[0]} × {currentDocument.imageSize[1]}
          </p>
        </div>
      </div>

      {/* Left Panel - History (slide) */}
      <div className="absolute top-20 z-10 transition-all duration-300 ease-in-out"
        style={{ left: showHistory ? '16px' : '-256px' }}
      >
        <div className="relative">
          <div className="w-64">
            <HistoryPanel />
          </div>
          <button
            onClick={() => setShowHistory((v) => !v)}
            className="absolute top-2 -right-6 px-1 py-3 rounded-r-lg border border-l-0 border-slate-700/50 bg-slate-900/90 backdrop-blur-sm text-slate-400 hover:text-white transition-colors shadow-lg"
            title={showHistory ? 'Скрыть историю' : 'Показать историю'}
          >
            <svg className={`w-4 h-4 transition-transform duration-300 ${showHistory ? '' : 'rotate-180'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </button>
        </div>
      </div>

      {/* Right Panel - Node Details (slide) */}
      <div className="absolute top-20 z-10 transition-all duration-300 ease-in-out"
        style={{ right: showDetails ? '16px' : '-320px' }}
      >
        <div className="relative">
          <div className="w-80">
            <NodeDetailsPanel />
          </div>
          <button
            onClick={() => setShowDetails((v) => !v)}
            className="absolute top-2 -left-6 px-1 py-3 rounded-l-lg border border-r-0 border-slate-700/50 bg-slate-900/90 backdrop-blur-sm text-slate-400 hover:text-white transition-colors shadow-lg"
            title={showDetails ? 'Скрыть детали' : 'Показать детали'}
          >
            <svg className={`w-4 h-4 transition-transform duration-300 ${showDetails ? '' : 'rotate-180'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}
