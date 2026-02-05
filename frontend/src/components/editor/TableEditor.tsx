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
