import React from 'react';
import { Layers, Database, Activity } from 'lucide-react';

interface HeaderProps {
  reportsCount: number;
  cachesCount: number;
  datasetsCount: number;
  lmConnected: boolean;
  activeModel?: string;
  onRefreshLM?: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  reportsCount,
  cachesCount,
  datasetsCount,
  lmConnected,
  activeModel,
  onRefreshLM,
}) => {
  return (
    <header
      id="main-header"
      className="flex flex-wrap items-center justify-between gap-4 px-6 py-5 bg-[#12141c] border border-white/10 rounded-xl shadow-xl mb-6"
    >
      <div className="flex items-center gap-3">
        <div className="flex items-baseline gap-3">
          <h1 className="text-xl font-bold text-zinc-100 tracking-tight">VLM Studio</h1>
          <span className="text-xs font-medium text-zinc-400 bg-white/5 px-2.5 py-1 rounded-md border border-white/10">
            Detection Benchmark
          </span>
        </div>
      </div>

      <div className="flex items-center flex-wrap gap-2.5 text-xs sm:text-sm">
        <button
          id="header-lm-status-btn"
          onClick={onRefreshLM}
          title="Click to check endpoint status"
          className="flex items-center gap-2 px-3.5 py-2 bg-[#161822] border border-white/10 rounded-lg text-zinc-300 hover:border-white/20 transition cursor-pointer"
        >
          <span
            className={`w-2.5 h-2.5 rounded-full ${
              lmConnected
                ? 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.7)]'
                : 'bg-zinc-500'
            }`}
          />
          <span>
            LM Studio:{' '}
            {lmConnected ? (
              <strong className="text-emerald-400 font-medium">
                {activeModel ? activeModel : 'Connected (Idle)'}
              </strong>
            ) : (
              <span className="text-zinc-500">Offline</span>
            )}
          </span>
        </button>

        <div id="header-stat-datasets" className="flex items-center gap-2 px-3 py-2 bg-[#161822] border border-white/10 rounded-lg text-zinc-400">
          <Database className="w-4 h-4 text-zinc-500" />
          <span>Datasets:</span>
          <strong className="text-zinc-100 font-semibold">{datasetsCount}</strong>
        </div>

        <div id="header-stat-runs" className="flex items-center gap-2 px-3 py-2 bg-[#161822] border border-white/10 rounded-lg text-zinc-400">
          <Activity className="w-4 h-4 text-zinc-500" />
          <span>Runs:</span>
          <strong className="text-zinc-100 font-semibold">{reportsCount}</strong>
        </div>

        <div id="header-stat-caches" className="flex items-center gap-2 px-3 py-2 bg-[#161822] border border-white/10 rounded-lg text-zinc-400">
          <Layers className="w-4 h-4 text-zinc-500" />
          <span>Caches:</span>
          <strong className="text-zinc-100 font-semibold">{cachesCount}</strong>
        </div>
      </div>
    </header>
  );
};
