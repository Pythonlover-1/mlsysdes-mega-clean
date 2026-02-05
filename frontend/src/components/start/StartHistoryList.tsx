import { useAppStore } from '../../store';

export default function StartHistoryList() {
  const history = useAppStore((s) => s.history);
  const loadFromHistory = useAppStore((s) => s.loadFromHistory);
  const clearHistory = useAppStore((s) => s.clearHistory);

  if (history.length === 0) return null;

  return (
    <div className="space-y-4 p-6 bg-black rounded-2xl border border-gray-700">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gray-800 flex items-center justify-center">
            <svg className="w-4 h-4 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <h2 className="font-bold text-white text-lg">История</h2>
        </div>
        <button
          onClick={clearHistory}
          className="px-4 py-2 text-sm text-red-400 hover:text-red-300 hover:bg-red-500/10 rounded-lg transition-all duration-200 border border-red-500/20 hover:border-red-500/40 focus:outline-none focus:ring-2 focus:ring-gray-500"
        >
          Очистить
        </button>
      </div>
      <ul className="space-y-2">
        {history.map((item, i) => (
          <li
            key={item.timestamp + '-' + i}
            onClick={() => loadFromHistory(item)}
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                loadFromHistory(item);
              }
            }}
            className="group p-4 rounded-xl cursor-pointer transition-all duration-200 flex items-center justify-between focus:outline-none focus:ring-2 focus:ring-gray-500 bg-black border-2 border-gray-800 hover:border-gray-500 hover:bg-gray-900"
          >
            <div className="flex items-center gap-3 flex-1 min-w-0">
              <div className="w-10 h-10 rounded-lg bg-gray-800 group-hover:bg-white/10 flex items-center justify-center flex-shrink-0 transition-colors">
                <svg className="w-5 h-5 text-gray-400 group-hover:text-white transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-white truncate group-hover:text-gray-200 transition-colors">
                  {item.filename}
                </p>
                <p className="text-xs text-gray-500 mt-0.5">
                  {new Date(item.timestamp).toLocaleString('ru-RU', {
                    day: '2-digit',
                    month: 'short',
                    hour: '2-digit',
                    minute: '2-digit',
                  })}
                  {' - '}
                  {item.graph?.nodes?.length ?? 0} элементов
                </p>
              </div>
            </div>
            <svg
              className="w-5 h-5 text-gray-600 group-hover:text-white transition-all group-hover:translate-x-1 flex-shrink-0"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </li>
        ))}
      </ul>
    </div>
  );
}
