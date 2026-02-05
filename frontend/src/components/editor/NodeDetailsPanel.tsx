import { useAppStore } from '../../store';
import FloatingPanel from '../shared/FloatingPanel';
import CroppedImage from './CroppedImage';
import type { BPMNNode, BPMNGraph } from '../../types';

function NodeDetails({ node, lanes, imageDataUrl }: { node: BPMNNode; lanes: BPMNGraph['lanes']; imageDataUrl: string }) {
  const lane = node.lane_id ? lanes.find((l) => l.id === node.lane_id) : null;

  const getLabelColor = (label: string) => {
    if (label.includes('task') || label.includes('subProcess')) return 'text-cyan-400 bg-cyan-500/20';
    if (label.includes('Gateway')) return 'text-amber-400 bg-amber-500/20';
    if (label.includes('Event')) return 'text-emerald-400 bg-emerald-500/20';
    if (label.includes('data')) return 'text-orange-400 bg-orange-500/20';
    return 'text-slate-400 bg-slate-500/20';
  };

  return (
    <div className="p-4 space-y-4">
      {/* Cropped Image */}
      <div>
        <label className="text-xs text-slate-500 uppercase tracking-wider mb-2 block">Фрагмент изображения</label>
        <CroppedImage imageDataUrl={imageDataUrl} box={node.box} />
      </div>

      {/* Type and ID */}
      <div className="flex items-center gap-3">
        <div className={`px-3 py-1.5 rounded-lg text-sm font-medium ${getLabelColor(node.label)}`}>{node.label}</div>
        <span className="text-xs text-slate-500">#{node.id}</span>
      </div>

      {/* Name */}
      <div>
        <label className="text-xs text-slate-500 uppercase tracking-wider">Название</label>
        <p className="text-white font-medium mt-1">{node.name || <span className="text-slate-600 italic">Без названия</span>}</p>
      </div>

      {/* Confidence */}
      <div>
        <label className="text-xs text-slate-500 uppercase tracking-wider">Уверенность</label>
        <div className="flex items-center gap-2 mt-1">
          <div className="flex-1 h-2 bg-slate-700 rounded-full overflow-hidden">
            <div className="h-full bg-gradient-to-r from-cyan-500 to-emerald-500 rounded-full transition-all" style={{ width: `${node.score * 100}%` }} />
          </div>
          <span className="text-sm text-slate-300 font-mono">{(node.score * 100).toFixed(1)}%</span>
        </div>
      </div>

      {/* Position & Size */}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="text-xs text-slate-500 uppercase tracking-wider">Центр</label>
          <p className="text-slate-300 text-sm font-mono mt-1">
            ({node.center[0].toFixed(0)}, {node.center[1].toFixed(0)})
          </p>
        </div>
        <div>
          <label className="text-xs text-slate-500 uppercase tracking-wider">Размер</label>
          <p className="text-slate-300 text-sm font-mono mt-1">
            {(node.box[2] - node.box[0]).toFixed(0)} × {(node.box[3] - node.box[1]).toFixed(0)}
          </p>
        </div>
      </div>

      {/* Bounding Box */}
      <div>
        <label className="text-xs text-slate-500 uppercase tracking-wider">Bounding Box</label>
        <p className="text-slate-400 text-xs font-mono mt-1 bg-slate-800/50 p-2 rounded">
          x1: {node.box[0].toFixed(0)}, y1: {node.box[1].toFixed(0)}
          <br />
          x2: {node.box[2].toFixed(0)}, y2: {node.box[3].toFixed(0)}
        </p>
      </div>

      {/* Lane */}
      {lane && (
        <div>
          <label className="text-xs text-slate-500 uppercase tracking-wider">Дорожка (Lane)</label>
          <div className="mt-1 p-2 bg-violet-500/10 border border-violet-500/30 rounded-lg">
            <p className="text-violet-300 text-sm font-medium">{lane.name || lane.label}</p>
            <p className="text-violet-400 text-xs">#{lane.id}</p>
          </div>
        </div>
      )}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="h-full flex items-center justify-center text-slate-500 text-sm p-4">
      <div className="text-center space-y-2">
        <svg className="w-12 h-12 mx-auto text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122"
          />
        </svg>
        <p>Выберите элемент на графе</p>
        <p className="text-xs text-slate-600">Кликните на любой узел для просмотра деталей</p>
      </div>
    </div>
  );
}

export default function NodeDetailsPanel() {
  const currentDocument = useAppStore((s) => s.currentDocument);
  const selectedNodeId = useAppStore((s) => s.selectedNodeId);

  const selectedNode = selectedNodeId ? currentDocument?.graph.nodes.find((n) => n.id === selectedNodeId) : null;

  return (
    <FloatingPanel
      title="Детали элемента"
      icon={
        <svg className="w-4 h-4 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      }
      className="w-80 max-h-[calc(100vh-180px)]"
    >
      {selectedNode && currentDocument ? (
        <NodeDetails node={selectedNode} lanes={currentDocument.graph.lanes} imageDataUrl={currentDocument.imageDataUrl} />
      ) : (
        <EmptyState />
      )}
    </FloatingPanel>
  );
}
