import type { Node, Edge } from "@xyflow/react";
import type { BPMNGraph, BPMNNode, BPMNEdge } from "../types";

// Node type based on BPMN element type
export type BpmnNodeType = "task" | "gateway" | "event" | "data" | "default";

export interface BpmnNodeData extends Record<string, unknown> {
  label: string;
  name: string;
  bpmnType: string;
  score: number;
  laneId: string | null;
  isHovered?: boolean;
  onLabelChange?: (newLabel: string) => void;
}

// Map BPMN label to node type for styling
function getBpmnNodeType(label: string): BpmnNodeType {
  if (label === "task" || label === "subProcess") return "task";
  if (label.includes("Gateway")) return "gateway";
  if (label.includes("Event") || label === "event") return "event";
  if (label === "dataObject" || label === "dataStore") return "data";
  return "default";
}

// Get edge style based on flow type
function getEdgeStyle(label: string): Partial<Edge> {
  switch (label) {
    case "messageFlow":
      return { style: { strokeDasharray: "5,5", stroke: "#2196F3" } };
    case "dataAssociation":
      return { style: { strokeDasharray: "3,3", stroke: "#9E9E9E" } };
    default:
      return { style: { stroke: "#333" } };
  }
}

// Transform API graph to React Flow format
export function transformToReactFlow(
  graph: BPMNGraph,
  imageSize: number[]
): { nodes: Node<BpmnNodeData>[]; edges: Edge[] } {
  const [imgWidth, imgHeight] = imageSize;

  // Calculate scale to fit in viewport (max 800x600)
  const maxWidth = 800;
  const maxHeight = 600;
  const scale = Math.min(maxWidth / imgWidth, maxHeight / imgHeight, 1);

  const nodes: Node<BpmnNodeData>[] = graph.nodes.map((node: BPMNNode) => {
    const [cx, cy] = node.center;
    const [x1, y1, x2, y2] = node.box;
    const width = (x2 - x1) * scale;
    const height = (y2 - y1) * scale;

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
      },
      style: {
        width,
        height,
      },
    };
  });

  const edges: Edge[] = graph.edges
    .filter((edge: BPMNEdge) => edge.source && edge.target)
    .map((edge: BPMNEdge) => ({
      id: edge.id,
      source: edge.source!,
      target: edge.target!,
      label: edge.label,
      ...getEdgeStyle(edge.label),
    }));

  return { nodes, edges };
}

// Find all paths from start events to end events
export function findProcessChains(graph: BPMNGraph): string[][] {
  const adjacency: Map<string, string[]> = new Map();
  const nodeMap: Map<string, BPMNNode> = new Map();

  // Build adjacency list and node map
  for (const node of graph.nodes) {
    nodeMap.set(node.id, node);
    adjacency.set(node.id, []);
  }

  for (const edge of graph.edges) {
    if (edge.source && edge.target) {
      const neighbors = adjacency.get(edge.source) || [];
      neighbors.push(edge.target);
      adjacency.set(edge.source, neighbors);
    }
  }

  // Find start nodes (events with no incoming edges or nodes with in-degree 0)
  const hasIncoming = new Set<string>();
  for (const edge of graph.edges) {
    if (edge.target) hasIncoming.add(edge.target);
  }

  const startNodes = graph.nodes.filter(
    (n) => !hasIncoming.has(n.id) || n.label === "event"
  );

  // Find end nodes (no outgoing edges)
  const endNodes = new Set(
    graph.nodes.filter((n) => (adjacency.get(n.id) || []).length === 0).map((n) => n.id)
  );

  // DFS to find all paths
  const paths: string[][] = [];
  const maxPaths = 20; // Limit number of paths

  function dfs(nodeId: string, path: string[], visited: Set<string>) {
    if (paths.length >= maxPaths) return;
    if (visited.has(nodeId)) return; // Avoid cycles

    const node = nodeMap.get(nodeId);
    if (!node) return;

    const newPath = [...path, formatNodeLabel(node)];
    const neighbors = adjacency.get(nodeId) || [];

    if (neighbors.length === 0 || endNodes.has(nodeId)) {
      if (newPath.length > 1) {
        paths.push(newPath);
      }
      return;
    }

    const newVisited = new Set(visited);
    newVisited.add(nodeId);

    for (const next of neighbors) {
      dfs(next, newPath, newVisited);
    }
  }

  // Start DFS from each start node
  for (const start of startNodes) {
    if (paths.length >= maxPaths) break;
    dfs(start.id, [], new Set());
  }

  return paths;
}

function formatNodeLabel(node: BPMNNode): string {
  const typeEmoji = getTypeEmoji(node.label);
  const name = node.name || node.label;
  return `${typeEmoji} ${name}`;
}

function getTypeEmoji(label: string): string {
  if (label === "event" || label.includes("Event")) return "⚪";
  if (label.includes("Gateway")) return "◇";
  if (label === "task" || label === "subProcess") return "▢";
  if (label === "dataObject" || label === "dataStore") return "📄";
  return "•";
}
