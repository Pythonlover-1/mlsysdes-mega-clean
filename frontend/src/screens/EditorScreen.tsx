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

  return (
    <div className="h-screen bg-black flex flex-col overflow-hidden">
      {/* Top Bar */}
      <div className="flex-shrink-0 p-4 border-b border-slate-800 flex items-center justify-between">
        {/* Back button and filename */}
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

        {/* Tab Bar - centered */}
        <TabBar />

        {/* Processing info */}
        <div className="text-right text-xs text-slate-500">
          <p>{currentDocument.processingTimeMs}ms</p>
          <p>
            {currentDocument.imageSize[0]} × {currentDocument.imageSize[1]}
          </p>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel - History */}
        <div className="flex-shrink-0 p-4">
          <HistoryPanel />
        </div>

        {/* Main Editor Area */}
        <div className="flex-1 p-4 overflow-hidden">
          <div className="h-full rounded-2xl border border-slate-700/50 bg-gradient-to-br from-slate-900 to-slate-800 overflow-hidden">
            {activeTab === 'graph' && <GraphEditor />}
            {activeTab === 'json' && <JsonEditor />}
            {activeTab === 'table' && <TableEditor />}
          </div>
        </div>

        {/* Right Panel - Node Details */}
        <div className="flex-shrink-0 p-4">
          <NodeDetailsPanel />
        </div>
      </div>
    </div>
  );
}
