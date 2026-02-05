import { useAppStore } from '../../store';
import FloatingPanel from '../shared/FloatingPanel';

export default function HistoryPanel() {
  const history = useAppStore((s) => s.history);
  const loadFromHistory = useAppStore((s) => s.loadFromHistory);
  const currentDocument = useAppStore((s) => s.currentDocument);

  if (history.length === 0) return null;

  return (
    <FloatingPanel
      title="История"
      icon={
        <svg className="w-4 h-4 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      }
      className="w-64 max-h-[calc(100vh-180px)]"
    >
      <ul className="p-2 space-y-1">
        {history.map((item, i) => {
          const isCurrent = currentDocument?.timestamp === item.timestamp;
          return (
            <li
              key={item.timestamp + '-' + i}
              onClick={() => loadFromHistory(item)}
              className={`group p-3 rounded-lg cursor-pointer transition-all flex items-center gap-3 ${
                isCurrent
                  ? 'bg-gradient-to-r from-cyan-500/20 to-blue-500/20 border border-cyan-500/30'
                  : 'hover:bg-slate-700/50 border border-transparent'
              }`}
            >
              {/* Thumbnail */}
              <div className="w-10 h-10 rounded-lg overflow-hidden flex-shrink-0 bg-slate-800">
                <img
                  src={item.imageDataUrl}
                  alt=""
                  className="w-full h-full object-cover"
                />
              </div>

              <div className="flex-1 min-w-0">
                <p className={`text-xs font-medium truncate ${isCurrent ? 'text-cyan-300' : 'text-white'}`}>
                  {item.filename}
                </p>
                <p className="text-[10px] text-slate-500 mt-0.5">
                  {new Date(item.timestamp).toLocaleString('ru-RU', {
                    day: '2-digit',
                    month: 'short',
                    hour: '2-digit',
                    minute: '2-digit',
                  })}
                </p>
              </div>

              {isCurrent && (
                <div className="w-2 h-2 rounded-full bg-cyan-400 flex-shrink-0" />
              )}
            </li>
          );
        })}
      </ul>
    </FloatingPanel>
  );
}
