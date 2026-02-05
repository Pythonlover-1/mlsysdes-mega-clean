import { useAppStore } from '../../store';
import type { EditorTab } from '../../types';

const tabs: { id: EditorTab; label: string; icon: JSX.Element; gradient: string }[] = [
  {
    id: 'graph',
    label: 'Граф',
    icon: (
      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
      </svg>
    ),
    gradient: 'from-cyan-500 to-blue-500 shadow-cyan-500/30',
  },
  {
    id: 'json',
    label: 'JSON',
    icon: (
      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
      </svg>
    ),
    gradient: 'from-violet-500 to-purple-500 shadow-violet-500/30',
  },
  {
    id: 'table',
    label: 'Таблица',
    icon: (
      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
      </svg>
    ),
    gradient: 'from-emerald-500 to-teal-500 shadow-emerald-500/30',
  },
];

export default function TabBar() {
  const activeTab = useAppStore((s) => s.activeTab);
  const setActiveTab = useAppStore((s) => s.setActiveTab);

  return (
    <div className="flex gap-2 p-1.5 bg-slate-800/80 rounded-xl border border-slate-700/50 backdrop-blur-sm">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => setActiveTab(tab.id)}
          className={`px-5 py-2.5 rounded-lg text-sm font-medium transition-all ${
            activeTab === tab.id
              ? `bg-gradient-to-r ${tab.gradient} text-white shadow-lg`
              : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
          }`}
        >
          <span className="flex items-center gap-2">
            {tab.icon}
            {tab.label}
          </span>
        </button>
      ))}
    </div>
  );
}
