import { useMemo, useCallback, useState, useEffect } from 'react';
import {
  ReactFlow,
  Controls,
  Background,
  MiniMap,
  useNodesState,
  useEdgesState,
  BackgroundVariant,
  type NodeTypes,
  type Node,
  type Edge,
  Handle,
  Position,
  MarkerType,
  type Connection,
  addEdge,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { useAppStore } from '../../store';
import type { BPMNGraph, BPMNEdge } from '../../types';
import { type BpmnNodeData } from '../../utils/graphTransform';

// Node type based on BPMN element type
type BpmnNodeType = 'task' | 'gateway' | 'event' | 'data' | 'default' | 'image';

function getBpmnNodeType(label: string): BpmnNodeType {
  if (label === 'task' || label === 'subProcess') return 'task';
  if (label.includes('Gateway')) return 'gateway';
  if (label.includes('Event') || label === 'event') return 'event';
  if (label === 'dataObject' || label === 'dataStore') return 'data';
  return 'default';
}

// Editable label component
function EditableLabel({
  label,
  isEditing,
  onStartEdit,
  onFinishEdit,
  isHighlighted,
}: {
  label: string;
  isEditing: boolean;
  onStartEdit: () => void;
  onFinishEdit: (newLabel: string) => void;
  isHighlighted: boolean;
}) {
  const [editValue, setEditValue] = useState(label);

  useEffect(() => {
    setEditValue(label);
  }, [label]);

  if (isEditing) {
    return (
      <input
        type="text"
        value={editValue}
        onChange={(e) => setEditValue(e.target.value)}
        onBlur={() => onFinishEdit(editValue)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') onFinishEdit(editValue);
          if (e.key === 'Escape') onFinishEdit(label);
        }}
        autoFocus
        className="bg-slate-700/80 text-white text-sm font-semibold px-2 py-1 rounded border border-cyan-400 outline-none w-full text-center"
        onClick={(e) => e.stopPropagation()}
      />
    );
  }

  return (
    <div
      onDoubleClick={(e) => {
        e.stopPropagation();
        onStartEdit();
      }}
      className={`font-semibold text-sm leading-tight cursor-text ${isHighlighted ? 'text-white' : 'text-slate-200'}`}
      title="Двойной клик для редактирования"
    >
      {label}
    </div>
  );
}

// Image node for background
function ImageNode({ data }: { data: { imageUrl: string; width: number; height: number } }) {
  return (
    <div
      style={{ width: data.width, height: data.height }}
      className="pointer-events-none select-none opacity-40"
    >
      <img
        src={data.imageUrl}
        alt="Background"
        className="w-full h-full object-contain"
        draggable={false}
      />
    </div>
  );
}

// Task node
function TaskNode({ data, selected }: { data: BpmnNodeData; selected?: boolean }) {
  const [isEditing, setIsEditing] = useState(false);

  return (
    <div className="relative group">
      {selected && (
        <div className="absolute -inset-2 bg-gradient-to-r from-cyan-500 via-blue-500 to-purple-500 rounded-2xl blur-lg opacity-75 animate-pulse" />
      )}
      <div
        className={`relative px-5 py-4 rounded-2xl text-center transition-all duration-300 cursor-pointer backdrop-blur-sm min-w-[160px]
          ${
            selected
              ? 'bg-gradient-to-br from-slate-800 to-slate-900 border-2 border-cyan-400 shadow-2xl shadow-cyan-500/40 scale-110'
              : 'bg-gradient-to-br from-slate-800/90 to-slate-900/90 border-2 border-slate-600/60 hover:border-cyan-500/70 hover:shadow-xl hover:shadow-cyan-500/20 hover:scale-105'
          }`}
      >
        <Handle type="target" position={Position.Left} className="!w-3.5 !h-3.5 !bg-gradient-to-r !from-cyan-400 !to-blue-500 !border-2 !border-slate-900" />
        <div
          className={`absolute -top-3 -left-3 w-7 h-7 rounded-xl flex items-center justify-center text-sm
          ${selected ? 'bg-gradient-to-br from-cyan-400 to-blue-500 text-white' : 'bg-slate-700 text-cyan-400 border border-slate-600'}
          shadow-lg transition-all duration-300`}
        >
          ☐
        </div>
        <EditableLabel
          label={data.label}
          isEditing={isEditing}
          onStartEdit={() => setIsEditing(true)}
          onFinishEdit={(newLabel) => {
            setIsEditing(false);
            if (newLabel !== data.label && data.onLabelChange) {
              data.onLabelChange(newLabel);
            }
          }}
          isHighlighted={!!selected}
        />
        <Handle type="source" position={Position.Right} className="!w-3.5 !h-3.5 !bg-gradient-to-r !from-blue-500 !to-purple-500 !border-2 !border-slate-900" />
      </div>
    </div>
  );
}

// Gateway node
function GatewayNode({ data, selected }: { data: BpmnNodeData; selected?: boolean }) {
  const isExclusive = data.bpmnType.includes('exclusive');
  const isParallel = data.bpmnType.includes('parallel');
  const isInclusive = data.bpmnType.includes('inclusive');

  const gatewaySymbol = isExclusive ? '✕' : isParallel ? '+' : isInclusive ? '○' : '◆';

  const gradientColors = isExclusive
    ? 'from-amber-500 to-orange-500'
    : isParallel
      ? 'from-emerald-500 to-teal-500'
      : 'from-violet-500 to-purple-500';

  return (
    <div className="w-full h-full flex items-center justify-center">
      {selected && (
        <div className={`absolute w-20 h-20 bg-gradient-to-r ${gradientColors} rounded-xl rotate-45 blur-xl opacity-70 animate-pulse`} />
      )}
      <Handle type="target" position={Position.Left} className={`!w-3.5 !h-3.5 !bg-gradient-to-r ${gradientColors} !border-2 !border-slate-900`} />
      <div
        className={`relative w-16 h-16 rotate-45 flex items-center justify-center transition-all duration-300 cursor-pointer rounded-xl
          ${
            selected
              ? `bg-gradient-to-br ${gradientColors} shadow-2xl scale-115`
              : 'bg-gradient-to-br from-slate-700 to-slate-800 hover:from-slate-600 hover:to-slate-700 hover:scale-110'
          }
          border-2 ${selected ? 'border-white/60' : 'border-slate-500/60 hover:border-slate-400'}`}
        title={data.label}
      >
        <span className={`-rotate-45 text-2xl font-bold ${selected ? 'text-white drop-shadow-lg' : 'text-slate-300'}`}>{gatewaySymbol}</span>
      </div>
      <Handle type="source" position={Position.Right} className={`!w-3.5 !h-3.5 !bg-gradient-to-r ${gradientColors} !border-2 !border-slate-900`} />
    </div>
  );
}

// Event node
function EventNode({ data, selected }: { data: BpmnNodeData; selected?: boolean }) {
  const bpmnType = data.bpmnType.toLowerCase();
  const isStart = bpmnType.includes('start') || bpmnType === 'event';
  const isEnd = bpmnType.includes('end');
  const isMessage = bpmnType.includes('message');
  const isTimer = bpmnType.includes('timer');

  let gradientColors = 'from-sky-400 to-blue-500';
  let ringColor = 'ring-sky-400/60';
  let icon = '●';

  if (isStart) {
    gradientColors = 'from-emerald-400 to-green-500';
    ringColor = 'ring-emerald-400/60';
    icon = '▸';
  } else if (isEnd) {
    gradientColors = 'from-rose-400 to-red-500';
    ringColor = 'ring-rose-400/60';
    icon = '■';
  } else if (isMessage) {
    gradientColors = 'from-blue-400 to-indigo-500';
    ringColor = 'ring-blue-400/60';
    icon = '✉';
  } else if (isTimer) {
    gradientColors = 'from-violet-400 to-purple-500';
    ringColor = 'ring-violet-400/60';
    icon = '⏱';
  }

  return (
    <div className="w-full h-full flex items-center justify-center">
      {selected && <div className={`absolute w-20 h-20 bg-gradient-to-r ${gradientColors} rounded-full blur-xl opacity-70 animate-pulse`} />}
      <Handle type="target" position={Position.Left} className={`!w-3.5 !h-3.5 !bg-gradient-to-r ${gradientColors} !border-2 !border-slate-900`} />
      <div
        className={`relative w-16 h-16 rounded-full flex items-center justify-center transition-all duration-300 cursor-pointer
          ${isEnd ? 'ring-4' : 'ring-2'} ${ringColor}
          ${
            selected
              ? `bg-gradient-to-br ${gradientColors} shadow-2xl scale-115`
              : 'bg-gradient-to-br from-slate-700 to-slate-800 hover:from-slate-600 hover:to-slate-700 hover:scale-110'
          }`}
        title={data.label}
      >
        {isEnd && <div className={`absolute inset-2.5 rounded-full border-2 ${selected ? 'border-white/60' : 'border-slate-500/60'}`} />}
        <span className={`text-xl ${selected ? 'text-white drop-shadow-lg' : 'text-slate-300'}`}>{icon}</span>
      </div>
      <Handle type="source" position={Position.Right} className={`!w-3.5 !h-3.5 !bg-gradient-to-r ${gradientColors} !border-2 !border-slate-900`} />
    </div>
  );
}

// Data node
function DataNode({ data, selected }: { data: BpmnNodeData; selected?: boolean }) {
  const [isEditing, setIsEditing] = useState(false);

  return (
    <div className="relative group">
      {selected && (
        <div className="absolute -inset-2 bg-gradient-to-r from-amber-500 to-orange-500 blur-lg opacity-60 animate-pulse" style={{ clipPath: 'polygon(0 15%, 85% 0, 100% 0, 100% 85%, 15% 100%, 0 100%)' }} />
      )}
      <div
        className={`relative px-4 py-3 text-sm text-center transition-all duration-300 cursor-pointer min-w-[120px]
          ${
            selected
              ? 'bg-gradient-to-br from-amber-600 to-orange-700 border-2 border-amber-400 shadow-2xl scale-110'
              : 'bg-gradient-to-br from-slate-700 to-slate-800 border-2 border-slate-500/60 hover:border-amber-500/70 hover:scale-105'
          }`}
        style={{ clipPath: 'polygon(0 15%, 85% 0, 100% 0, 100% 85%, 15% 100%, 0 100%)' }}
      >
        <Handle type="target" position={Position.Left} className="!bg-amber-400 !w-3 !h-3 !border-2 !border-slate-900" />
        <div className="flex items-center justify-center gap-1.5">
          <span>📄</span>
          <EditableLabel
            label={data.label}
            isEditing={isEditing}
            onStartEdit={() => setIsEditing(true)}
            onFinishEdit={(newLabel) => {
              setIsEditing(false);
              if (newLabel !== data.label && data.onLabelChange) {
                data.onLabelChange(newLabel);
              }
            }}
            isHighlighted={!!selected}
          />
        </div>
        <Handle type="source" position={Position.Right} className="!bg-amber-400 !w-3 !h-3 !border-2 !border-slate-900" />
      </div>
    </div>
  );
}

// Default node
function DefaultNode({ data, selected }: { data: BpmnNodeData; selected?: boolean }) {
  const [isEditing, setIsEditing] = useState(false);

  return (
    <div className="relative group">
      {selected && <div className="absolute -inset-2 bg-gradient-to-r from-slate-400 to-slate-500 rounded-xl blur-lg opacity-60 animate-pulse" />}
      <div
        className={`relative px-4 py-3 rounded-xl text-sm text-center transition-all duration-300 cursor-pointer min-w-[120px]
          ${
            selected
              ? 'bg-gradient-to-br from-slate-600 to-slate-700 border-2 border-slate-400 shadow-2xl scale-110'
              : 'bg-gradient-to-br from-slate-700 to-slate-800 border-2 border-slate-500/60 hover:border-slate-400/70 hover:scale-105'
          }`}
      >
        <Handle type="target" position={Position.Left} className="!bg-slate-400 !w-3 !h-3 !border-2 !border-slate-900" />
        <EditableLabel
          label={data.label}
          isEditing={isEditing}
          onStartEdit={() => setIsEditing(true)}
          onFinishEdit={(newLabel) => {
            setIsEditing(false);
            if (newLabel !== data.label && data.onLabelChange) {
              data.onLabelChange(newLabel);
            }
          }}
          isHighlighted={!!selected}
        />
        <Handle type="source" position={Position.Right} className="!bg-slate-400 !w-3 !h-3 !border-2 !border-slate-900" />
      </div>
    </div>
  );
}

const nodeTypes: NodeTypes = {
  task: TaskNode,
  gateway: GatewayNode,
  event: EventNode,
  data: DataNode,
  default: DefaultNode,
  image: ImageNode,
};

// Transform graph to React Flow format
function transformToReactFlow(
  graph: BPMNGraph,
  imageSize: number[],
  imageUrl: string,
  onLabelChange: (nodeId: string, newLabel: string) => void,
  savedImagePosition: { x: number; y: number } | null
): { nodes: Node[]; edges: Edge[]; defaultImagePosition: { x: number; y: number } } {
  const [imgWidth, imgHeight] = imageSize;
  const maxWidth = 3000;
  const maxHeight = 1800;
  const scale = Math.min(maxWidth / imgWidth, maxHeight / imgHeight, 3.0);

  const scaledWidth = imgWidth * scale;
  const scaledHeight = imgHeight * scale;

  // First, create BPMN nodes to find the topmost position
  const bpmnNodes: Node[] = graph.nodes.map((node) => {
    const [cx, cy] = node.center;
    const [x1, y1, x2, y2] = node.box;
    const width = Math.max((x2 - x1) * scale, 120);
    const height = Math.max((y2 - y1) * scale, 60);

    return {
      id: node.id,
      type: getBpmnNodeType(node.label),
      position: { x: cx * scale - width / 2, y: cy * scale - height / 2 },
      data: {
        label: node.name || node.label,
        name: node.name,
        bpmnType: node.label,
        score: node.score,
        laneId: node.lane_id,
        onLabelChange: (newLabel: string) => onLabelChange(node.id, newLabel),
      },
      style: { width, height },
    };
  });

  // Calculate default image position: above the topmost node
  let defaultImagePosition = { x: 0, y: 0 };
  if (bpmnNodes.length > 0) {
    const topMostY = Math.min(...bpmnNodes.map((n) => n.position.y));
    const leftMostX = Math.min(...bpmnNodes.map((n) => n.position.x));
    // Place image 50px above the topmost node
    defaultImagePosition = {
      x: leftMostX,
      y: topMostY - scaledHeight - 50,
    };
  }

  // Use saved position if available, otherwise use calculated default
  const imagePosition = savedImagePosition || defaultImagePosition;

  // Image background node
  const imageNode: Node = {
    id: '__background_image__',
    type: 'image',
    position: imagePosition,
    data: { imageUrl, width: scaledWidth, height: scaledHeight },
    draggable: true,
    selectable: false,
    connectable: false,
    zIndex: -1000,
  };

  const edges: Edge[] = graph.edges
    .filter((edge) => edge.source && edge.target)
    .map((edge) => {
      let strokeColor = '#64748b';
      let strokeDasharray: string | undefined = undefined;

      if (edge.label === 'messageFlow') {
        strokeColor = '#3b82f6';
        strokeDasharray = '10,5';
      } else if (edge.label === 'dataAssociation') {
        strokeColor = '#f59e0b';
        strokeDasharray = '5,5';
      } else {
        strokeColor = '#06b6d4';
      }

      return {
        id: edge.id,
        source: edge.source!,
        target: edge.target!,
        style: { stroke: strokeColor, strokeWidth: 2.5, strokeDasharray },
        markerEnd: { type: MarkerType.ArrowClosed, color: strokeColor, width: 24, height: 24 },
        animated: edge.label === 'sequenceFlow',
      };
    });

  return { nodes: [imageNode, ...bpmnNodes], edges, defaultImagePosition };
}

export default function GraphEditor() {
  const currentDocument = useAppStore((s) => s.currentDocument);
  const updateGraph = useAppStore((s) => s.updateGraph);
  const selectedNodeId = useAppStore((s) => s.selectedNodeId);
  const setSelectedNodeId = useAppStore((s) => s.setSelectedNodeId);
  const getImageNodePosition = useAppStore((s) => s.getImageNodePosition);
  const setImageNodePosition = useAppStore((s) => s.setImageNodePosition);

  const handleLabelChange = useCallback(
    (nodeId: string, newLabel: string) => {
      if (!currentDocument) return;
      const newGraph = {
        ...currentDocument.graph,
        nodes: currentDocument.graph.nodes.map((n) => (n.id === nodeId ? { ...n, name: newLabel } : n)),
      };
      updateGraph(newGraph);
    },
    [currentDocument, updateGraph]
  );

  // Get saved image position for current document
  const savedImagePosition = useMemo(() => {
    if (!currentDocument) return null;
    return getImageNodePosition(currentDocument.timestamp);
  }, [currentDocument, getImageNodePosition]);

  const { nodes: transformedNodes, edges: transformedEdges } = useMemo(() => {
    if (!currentDocument) return { nodes: [], edges: [], defaultImagePosition: { x: 0, y: 0 } };
    return transformToReactFlow(
      currentDocument.graph,
      currentDocument.imageSize,
      currentDocument.imageDataUrl,
      handleLabelChange,
      savedImagePosition
    );
  }, [currentDocument, handleLabelChange, savedImagePosition]);

  const [nodes, setNodes, onNodesChange] = useNodesState(transformedNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(transformedEdges);

  // Sync when graph changes externally
  useEffect(() => {
    setNodes(transformedNodes);
  }, [transformedNodes, setNodes]);

  useEffect(() => {
    setEdges(transformedEdges);
  }, [transformedEdges, setEdges]);

  // Handle node position changes (for saving image node position)
  const handleNodesChange = useCallback(
    (changes: Parameters<typeof onNodesChange>[0]) => {
      onNodesChange(changes);

      // Check if image node position changed
      if (!currentDocument) return;
      for (const change of changes) {
        if (change.type === 'position' && change.id === '__background_image__' && change.position) {
          setImageNodePosition(currentDocument.timestamp, change.position);
        }
      }
    },
    [onNodesChange, currentDocument, setImageNodePosition]
  );

  // Mark selected node
  const nodesWithSelection = useMemo(
    () =>
      nodes.map((node) => ({
        ...node,
        selected: node.id === selectedNodeId,
      })),
    [nodes, selectedNodeId]
  );

  const handleNodeClick = useCallback(
    (_event: React.MouseEvent, node: Node) => {
      if (node.id === '__background_image__') return;
      setSelectedNodeId(selectedNodeId === node.id ? null : node.id);
    },
    [setSelectedNodeId, selectedNodeId]
  );

  const handlePaneClick = useCallback(() => {
    setSelectedNodeId(null);
  }, [setSelectedNodeId]);

  // Handle creating new edges
  const onConnect = useCallback(
    (params: Connection) => {
      if (!params.source || !params.target || !currentDocument) return;
      if (params.source === '__background_image__' || params.target === '__background_image__') return;

      const newEdge: BPMNEdge = {
        id: `edge_${Date.now()}`,
        label: 'sequenceFlow',
        score: 1.0,
        source: params.source,
        target: params.target,
        keypoints: [],
      };

      const newGraph = {
        ...currentDocument.graph,
        edges: [...currentDocument.graph.edges, newEdge],
      };
      updateGraph(newGraph);

      // Also add to local React Flow state for immediate feedback
      setEdges((eds) =>
        addEdge(
          {
            ...params,
            id: newEdge.id,
            style: { stroke: '#06b6d4', strokeWidth: 2.5 },
            markerEnd: { type: MarkerType.ArrowClosed, color: '#06b6d4', width: 24, height: 24 },
            animated: true,
          },
          eds
        )
      );
    },
    [currentDocument, updateGraph, setEdges]
  );

  if (!currentDocument) {
    return (
      <div className="w-full h-full flex items-center justify-center text-slate-500">
        No document loaded
      </div>
    );
  }

  return (
    <div className="w-full h-full rounded-2xl border border-slate-700/50 bg-gradient-to-br from-slate-900 via-slate-900 to-slate-800 overflow-hidden shadow-2xl">
      <ReactFlow
        nodes={nodesWithSelection}
        edges={edges}
        onNodesChange={handleNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        nodeTypes={nodeTypes}
        onNodeClick={handleNodeClick}
        onPaneClick={handlePaneClick}
        fitView
        fitViewOptions={{ padding: 0.3 }}
        minZoom={0.1}
        maxZoom={2}
        proOptions={{ hideAttribution: true }}
      >
        <Controls
          className="!bg-slate-800/90 !border-slate-600 !rounded-xl !shadow-xl [&>button]:!bg-slate-700 [&>button]:!border-slate-600 [&>button]:hover:!bg-slate-600 [&>button]:!text-slate-300"
          showInteractive={false}
        />
        <MiniMap
          className="!bg-slate-800/80 !border-slate-600 !rounded-xl"
          nodeColor={(node) => {
            if (node.id === '__background_image__') return 'transparent';
            if (node.id === selectedNodeId) return '#22d3ee';
            switch (node.type) {
              case 'task':
                return '#06b6d4';
              case 'gateway':
                return '#f59e0b';
              case 'event':
                return '#10b981';
              case 'data':
                return '#f97316';
              default:
                return '#64748b';
            }
          }}
          maskColor="rgba(15, 23, 42, 0.85)"
          pannable
          zoomable
        />
        <Background variant={BackgroundVariant.Dots} gap={40} size={2} color="#334155" />
      </ReactFlow>
    </div>
  );
}
