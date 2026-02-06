import { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import { useAppStore } from '../../store';
import type { BPMNGraph } from '../../types';

function graphToCsv(graph: BPMNGraph): string {
  const header = 'SOURCE_ID,SOURCE_LABEL,SOURCE_NAME,SOURCE_LANE,TARGET_ID,TARGET_LABEL,TARGET_NAME,TARGET_LANE';
  const nodeMap = new Map(graph.nodes.map((n) => [n.id, n]));

  const rows = graph.edges
    .filter((e) => e.source && e.target)
    .map((edge) => {
      const source = nodeMap.get(edge.source!);
      const target = nodeMap.get(edge.target!);
      return [
        edge.source,
        source?.label || '',
        source?.name || '',
        source?.lane_id || '',
        edge.target,
        target?.label || '',
        target?.name || '',
        target?.lane_id || '',
      ]
        .map((v) => `"${v}"`)
        .join(',');
    });

  return [header, ...rows].join('\n');
}

type ParsedStep = {
  name: string;
  roles: string[];
};

function buildParsedSteps(graph: BPMNGraph, includeRoles: boolean): ParsedStep[] {
  const nodeMap = new Map(graph.nodes.map((n) => [n.id, n]));
  const laneMap = new Map(graph.lanes.map((l) => [l.id, l]));
  const sequenceEdges = graph.edges.filter((e) => e.source && e.target && e.label === 'sequenceFlow');
  const edges = sequenceEdges.length > 0 ? sequenceEdges : graph.edges.filter((e) => e.source && e.target);

  const adjacency = new Map<string, string[]>();
  const indegree = new Map<string, number>();
  for (const node of graph.nodes) {
    adjacency.set(node.id, []);
    indegree.set(node.id, 0);
  }
  for (const edge of edges) {
    const source = edge.source!;
    const target = edge.target!;
    adjacency.get(source)?.push(target);
    indegree.set(target, (indegree.get(target) || 0) + 1);
  }

  const hasTaskNodes = graph.nodes.some((n) => n.label === 'task' || n.label === 'subProcess');
  const isGateway = (label: string) => label.toLowerCase().includes('gateway');
  const isData = (label: string) => label === 'dataObject' || label === 'dataStore';

  const includeNode = (node: (typeof graph.nodes)[number]) => {
    if (isGateway(node.label) || isData(node.label)) return false;
    if (hasTaskNodes) return node.label === 'task' || node.label === 'subProcess';
    return Boolean(node.name?.trim());
  };

  const queue = graph.nodes
    .filter((n) => (indegree.get(n.id) || 0) === 0)
    .sort((a, b) => (a.center?.[1] || 0) - (b.center?.[1] || 0) || (a.center?.[0] || 0) - (b.center?.[0] || 0))
    .map((n) => n.id);

  const orderedIds: string[] = [];
  const seen = new Set<string>();
  while (queue.length > 0) {
    const id = queue.shift()!;
    if (seen.has(id)) continue;
    seen.add(id);
    orderedIds.push(id);
    const neighbors = adjacency.get(id) || [];
    for (const next of neighbors) {
      indegree.set(next, (indegree.get(next) || 0) - 1);
      if ((indegree.get(next) || 0) === 0) {
        queue.push(next);
      }
    }
  }

  const steps = orderedIds
    .map((id) => nodeMap.get(id))
    .filter((n): n is NonNullable<typeof n> => Boolean(n))
    .filter(includeNode)
    .map((node) => {
      let roles: string[] = [];
      if (includeRoles) {
        const lane = node.lane_id ? laneMap.get(node.lane_id) : null;
        const roleRaw = (lane?.name || lane?.label || '').trim();
        roles = roleRaw
          ? roleRaw
              .split(/[;,/|]/g)
              .map((r) => r.trim())
              .filter(Boolean)
          : ['—'];
      }
      return {
        name: (node.name || node.label || '').trim() || '—',
        roles,
      };
    });

  return steps.length > 0 ? steps : [{ name: '—', roles: includeRoles ? ['—'] : [] }];
}

function createParsedDiagramHtml(steps: ParsedStep[], title: string, includeRoles: boolean): string {
  const escapeHtml = (value: string) =>
    value
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');

  const rows = steps
    .map((step, index) => {
      const safeName = escapeHtml(step.name);
      const safeRoles = step.roles.map((role) => escapeHtml(role));
      const rolesHtml = step.roles.length > 1
        ? `<ul class="roles">${safeRoles.map((role) => `<li>${role}</li>`).join('')}</ul>`
        : safeRoles[0];
      const roleCell = includeRoles ? `<td class="col-role">${rolesHtml || '—'}</td>` : '';
      return `
        <tr>
          <td class="col-num">${index + 1}</td>
          <td class="col-action">${safeName}</td>
          ${roleCell}
        </tr>
      `;
    })
    .join('');

  const roleHeader = includeRoles ? '<th>Роль</th>' : '';

  return `<!doctype html>
<html lang="ru">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>${title} — Разобранная диаграмма</title>
    <style>
      :root { color-scheme: dark; }
      body {
        margin: 0;
        padding: 24px;
        background: #000;
        color: #fff;
        font-family: "Inter", "Segoe UI", Arial, sans-serif;
      }
      .wrap {
        max-width: 1100px;
        margin: 0 auto;
      }
      h1 {
        margin: 0 0 16px;
        font-size: 20px;
        font-weight: 600;
        color: #e2e8f0;
      }
      table {
        width: 100%;
        border-collapse: collapse;
        background: #000;
        border: 2px solid #fff;
      }
      th, td {
        border: 2px solid #fff;
        padding: 12px 16px;
        vertical-align: top;
        font-size: 16px;
      }
      th {
        text-align: left;
        font-weight: 700;
      }
      .col-num {
        width: 64px;
        text-align: center;
        font-weight: 700;
      }
      .roles {
        margin: 0;
        padding-left: 18px;
      }
      .roles li {
        margin: 0 0 4px;
      }
    </style>
  </head>
  <body>
    <div class="wrap">
      <h1>Разобранная диаграмма</h1>
      <table>
        <thead>
          <tr>
            <th>№</th>
            <th>Наименование действия</th>
            ${roleHeader}
          </tr>
        </thead>
        <tbody>
          ${rows}
        </tbody>
      </table>
    </div>
  </body>
</html>`;
}

function parseCsv(csv: string): string[][] {
  const lines = csv.trim().split('\n');
  return lines.map((line) => {
    const cells: string[] = [];
    let current = '';
    let inQuotes = false;

    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      if (char === '"') {
        inQuotes = !inQuotes;
      } else if (char === ',' && !inQuotes) {
        cells.push(current.trim().replace(/^"|"$/g, ''));
        current = '';
      } else {
        current += char;
      }
    }
    cells.push(current.trim().replace(/^"|"$/g, ''));
    return cells;
  });
}

function EditableCell({
  value,
  onChange,
  isHighlighted,
  isFirstColumn,
}: {
  value: string;
  onChange: (newValue: string) => void;
  isHighlighted: boolean;
  isFirstColumn: boolean;
}) {
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState(value);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setEditValue(value);
  }, [value]);

  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [isEditing]);

  if (isEditing) {
    return (
      <input
        ref={inputRef}
        type="text"
        value={editValue}
        onChange={(e) => setEditValue(e.target.value)}
        onBlur={() => {
          setIsEditing(false);
          if (editValue !== value) {
            onChange(editValue);
          }
        }}
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            setIsEditing(false);
            if (editValue !== value) {
              onChange(editValue);
            }
          }
          if (e.key === 'Escape') {
            setIsEditing(false);
            setEditValue(value);
          }
        }}
        className="bg-slate-700 text-white px-2 py-1 rounded border border-cyan-400 outline-none w-full text-sm"
      />
    );
  }

  return (
    <span
      onDoubleClick={() => setIsEditing(true)}
      className={`cursor-text hover:bg-slate-700/50 px-1 py-0.5 rounded transition-colors block
        ${isHighlighted ? 'text-white font-medium' : ''}
        ${isFirstColumn && !isHighlighted ? 'font-medium text-slate-200' : ''}
        ${!isFirstColumn && !isHighlighted ? 'text-slate-400' : ''}`}
      title="Двойной клик для редактирования"
    >
      {value || <span className="text-slate-600 italic">пусто</span>}
    </span>
  );
}

export default function TableEditor() {
  const currentDocument = useAppStore((s) => s.currentDocument);
  const updateGraph = useAppStore((s) => s.updateGraph);
  const selectedNodeId = useAppStore((s) => s.selectedNodeId);
  const setSelectedNodeId = useAppStore((s) => s.setSelectedNodeId);

  const tableRef = useRef<HTMLDivElement>(null);

  const csv = useMemo(() => (currentDocument ? graphToCsv(currentDocument.graph) : ''), [currentDocument]);
  const rows = useMemo(() => parseCsv(csv), [csv]);
  const [header, ...body] = rows;

  // Get highlighted rows based on selected node
  const highlightedRows = useMemo(() => {
    if (!selectedNodeId || !currentDocument) return new Set<number>();

    const node = currentDocument.graph.nodes.find((n) => n.id === selectedNodeId);
    const nodeLabel = node?.name || node?.label || null;

    const related = new Set<number>();
    for (let i = 1; i < rows.length; i++) {
      const row = rows[i];
      if (row.some((cell) => cell === selectedNodeId || (nodeLabel && cell.toLowerCase().includes(nodeLabel.toLowerCase())))) {
        related.add(i - 1);
      }
    }
    return related;
  }, [selectedNodeId, currentDocument, rows]);

  // Scroll to highlighted row
  useEffect(() => {
    if (highlightedRows.size > 0 && tableRef.current) {
      const firstIndex = Math.min(...highlightedRows);
      const row = tableRef.current.querySelector(`[data-row-index="${firstIndex}"]`);
      if (row) {
        row.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }
  }, [highlightedRows]);

  const handleCellChange = useCallback(
    (rowIndex: number, colIndex: number, newValue: string) => {
      if (!currentDocument) return;

      const edge = currentDocument.graph.edges[rowIndex];
      if (!edge) return;

      const newGraph = { ...currentDocument.graph };

      // Column 2 = SOURCE_NAME, Column 6 = TARGET_NAME
      if (colIndex === 2) {
        const sourceNode = newGraph.nodes.find((n) => n.id === edge.source);
        if (sourceNode) {
          newGraph.nodes = newGraph.nodes.map((n) => (n.id === sourceNode.id ? { ...n, name: newValue } : n));
        }
      } else if (colIndex === 6) {
        const targetNode = newGraph.nodes.find((n) => n.id === edge.target);
        if (targetNode) {
          newGraph.nodes = newGraph.nodes.map((n) => (n.id === targetNode.id ? { ...n, name: newValue } : n));
        }
      }

      updateGraph(newGraph);
    },
    [currentDocument, updateGraph]
  );

  const handleRowClick = useCallback(
    (rowIndex: number) => {
      if (!currentDocument) return;
      const edge = currentDocument.graph.edges[rowIndex];
      if (edge?.source) {
        setSelectedNodeId(selectedNodeId === edge.source ? null : edge.source);
      }
    },
    [currentDocument, setSelectedNodeId, selectedNodeId]
  );

  const downloadCsv = useCallback(() => {
    if (!currentDocument) return;
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = currentDocument.filename.replace(/\.\w+$/, '') + '.csv';
    a.click();
    URL.revokeObjectURL(url);
  }, [csv, currentDocument]);

  const downloadParsedDiagram = useCallback(() => {
    if (!currentDocument) return;
    const includeRoles = (currentDocument.graph.lanes || []).length > 0;
    const steps = buildParsedSteps(currentDocument.graph, includeRoles);
    const html = createParsedDiagramHtml(steps, currentDocument.filename.replace(/\.\w+$/, ''), includeRoles);
    const blob = new Blob([html], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = currentDocument.filename.replace(/\.\w+$/, '') + '-parsed.html';
    a.click();
    URL.revokeObjectURL(url);
  }, [currentDocument]);

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
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-emerald-600 to-teal-600 flex items-center justify-center">
            <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
          </div>
          <span className="text-white font-semibold">Table Editor</span>
          <span className="text-sm text-slate-500">({body.length} rows)</span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={downloadParsedDiagram}
            className="px-3 py-1.5 text-sm bg-gradient-to-r from-indigo-500 to-purple-500 text-white rounded-lg hover:from-indigo-400 hover:to-purple-400 transition-all flex items-center gap-1.5"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            Скачать разбор (HTML)
          </button>
          <button
            onClick={downloadCsv}
            className="px-3 py-1.5 text-sm bg-gradient-to-r from-emerald-500 to-teal-500 text-white rounded-lg hover:from-emerald-400 hover:to-teal-400 transition-all flex items-center gap-1.5"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            Скачать CSV
          </button>
        </div>
      </div>

      {/* Table */}
      <div ref={tableRef} className="flex-1 overflow-auto">
        <table className="min-w-full text-sm">
          <thead className="bg-slate-800/80 sticky top-0 z-10 backdrop-blur-sm">
            <tr>
              {header?.map((h, i) => (
                <th key={i} className="px-5 py-4 text-left font-bold text-slate-300 border-b border-slate-700 uppercase tracking-wider text-xs">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/50">
            {body.map((row, ri) => {
              const isHighlighted = highlightedRows.has(ri);
              return (
                <tr
                  key={ri}
                  data-row-index={ri}
                  onClick={() => handleRowClick(ri)}
                  className={`transition-all duration-300 group cursor-pointer
                    ${
                      isHighlighted
                        ? 'bg-gradient-to-r from-indigo-500/20 via-purple-500/20 to-indigo-500/20 border-l-4 border-l-indigo-400'
                        : 'hover:bg-slate-800/50 border-l-4 border-l-transparent'
                    }`}
                >
                  {row.map((cell, ci) => (
                    <td key={ci} className="px-5 py-3">
                      <EditableCell
                        value={cell}
                        onChange={(newValue) => handleCellChange(ri, ci, newValue)}
                        isHighlighted={isHighlighted}
                        isFirstColumn={ci === 0}
                      />
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Status bar */}
      <div className="px-4 py-2 border-t border-slate-700/50 bg-slate-800/30 text-xs text-slate-500 text-center">
        Двойной клик на ячейку для редактирования | Клик на строку для выбора узла
      </div>
    </div>
  );
}
