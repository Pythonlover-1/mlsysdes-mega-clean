import { ReactNode } from 'react';

interface Props {
  title: string;
  icon?: ReactNode;
  children: ReactNode;
  className?: string;
}

export default function FloatingPanel({ title, icon, children, className = '' }: Props) {
  return (
    <div
      className={`rounded-2xl border border-slate-700/50 bg-gradient-to-br from-slate-900/95 to-slate-800/95 shadow-xl backdrop-blur-sm overflow-hidden ${className}`}
    >
      <div className="px-4 py-3 border-b border-slate-700/50 bg-slate-800/50">
        <h3 className="text-sm font-semibold text-white flex items-center gap-2">
          {icon}
          {title}
        </h3>
      </div>
      <div className="overflow-auto">{children}</div>
    </div>
  );
}
