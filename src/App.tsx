import React, { useState } from 'react';
import { Header } from './components/Header';
import { LeaderboardTab } from './components/Tabs/LeaderboardTab';
import { VisualInspectorTab } from './components/Tabs/VisualInspectorTab';
import { RunEvaluationTab } from './components/Tabs/RunEvaluationTab';
import { ArchitectureTab } from './components/Tabs/ArchitectureTab';
import { INITIAL_REPORTS, PRESET_PREDICTION_CACHES } from './data/benchmarkData';
import { EvaluationReport, PredictionCache } from './types';
import { Trophy, Eye, Play, BookOpen } from 'lucide-react';

export function App() {
  const [activeTab, setActiveTab] = useState<'leaderboard' | 'inspector' | 'run' | 'architecture'>('leaderboard');
  const [reports, setReports] = useState<Record<string, EvaluationReport>>(INITIAL_REPORTS);
  const [selectedReportKey, setSelectedReportKey] = useState<string>(Object.keys(INITIAL_REPORTS)[0]);
  const [caches, setCaches] = useState<PredictionCache[]>(PRESET_PREDICTION_CACHES);
  const [selectedCacheId, setSelectedCacheId] = useState<string>(PRESET_PREDICTION_CACHES[0].id);
  const [apiBase, setApiBase] = useState<string>('http://localhost:1234/v1');
  const [lmConnected, setLmConnected] = useState<boolean>(true);

  const availableDatasets = [
    'datasets/Barista_workflow_small',
    'datasets/Robot_tabletop_manipulation',
    'datasets/COCO_val2017',
  ];

  const handleAddReportAndCache = (
    reportKey: string,
    newReport: EvaluationReport,
    newCache: PredictionCache
  ) => {
    setReports((prev) => ({ ...prev, [reportKey]: newReport }));
    setSelectedReportKey(reportKey);
    setCaches((prev) => {
      const filtered = prev.filter((c) => c.id !== newCache.id);
      return [newCache, ...filtered];
    });
    setSelectedCacheId(newCache.id);
  };

  return (
    <div className="min-h-screen bg-[#0b0c10] text-[#e6e8ec] p-4 md:p-6 max-w-7xl mx-auto flex flex-col">
      {/* Top Header */}
      <Header
        reportsCount={Object.keys(reports).length}
        cachesCount={caches.length}
        datasetsCount={availableDatasets.length}
        lmConnected={lmConnected}
        activeModel="zai-org/glm-4.6v-flash"
        onRefreshLM={() => setLmConnected((prev) => !prev)}
      />

      {/* Primary Navigation Tabs */}
      <div className="flex border-b border-white/10 mb-6 gap-1 overflow-x-auto">
        <button
          onClick={() => setActiveTab('leaderboard')}
          className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold border-b-2 transition whitespace-nowrap cursor-pointer ${
            activeTab === 'leaderboard'
              ? 'border-indigo-500 text-indigo-400 bg-white/[0.02]'
              : 'border-transparent text-zinc-400 hover:text-zinc-200 hover:bg-white/[0.01]'
          }`}
        >
          <Trophy className="w-3.5 h-3.5" /> Leaderboard & Metrics
        </button>

        <button
          onClick={() => setActiveTab('inspector')}
          className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold border-b-2 transition whitespace-nowrap cursor-pointer ${
            activeTab === 'inspector'
              ? 'border-indigo-500 text-indigo-400 bg-white/[0.02]'
              : 'border-transparent text-zinc-400 hover:text-zinc-200 hover:bg-white/[0.01]'
          }`}
        >
          <Eye className="w-3.5 h-3.5" /> Visual Inspector
        </button>

        <button
          onClick={() => setActiveTab('run')}
          className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold border-b-2 transition whitespace-nowrap cursor-pointer ${
            activeTab === 'run'
              ? 'border-indigo-500 text-indigo-400 bg-white/[0.02]'
              : 'border-transparent text-zinc-400 hover:text-zinc-200 hover:bg-white/[0.01]'
          }`}
        >
          <Play className="w-3.5 h-3.5" /> Run Evaluation
        </button>

        <button
          onClick={() => setActiveTab('architecture')}
          className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold border-b-2 transition whitespace-nowrap cursor-pointer ${
            activeTab === 'architecture'
              ? 'border-indigo-500 text-indigo-400 bg-white/[0.02]'
              : 'border-transparent text-zinc-400 hover:text-zinc-200 hover:bg-white/[0.01]'
          }`}
        >
          <BookOpen className="w-3.5 h-3.5" /> Architecture & Docs
        </button>
      </div>

      {/* Main Tab Content Viewport */}
      <main className="flex-1 pb-10">
        {activeTab === 'leaderboard' && (
          <LeaderboardTab
            reports={reports}
            selectedReportKey={selectedReportKey}
            onSelectReportKey={setSelectedReportKey}
          />
        )}

        {activeTab === 'inspector' && (
          <VisualInspectorTab
            caches={caches}
            selectedCacheId={selectedCacheId}
            onSelectCacheId={setSelectedCacheId}
          />
        )}

        {activeTab === 'run' && (
          <RunEvaluationTab
            apiBase={apiBase}
            onUpdateApiBase={setApiBase}
            availableDatasets={availableDatasets}
            caches={caches}
            onAddReportAndCache={handleAddReportAndCache}
          />
        )}

        {activeTab === 'architecture' && <ArchitectureTab />}
      </main>

      {/* Footer */}
      <footer className="pt-4 mt-auto border-t border-white/5 flex flex-wrap items-center justify-between text-[11px] text-zinc-500 gap-2">
        <div>VLM Studio — Local Vision-Language Model Object Detection Evaluator</div>
        <div className="font-mono">COCO 101-point & Continuous AP Engine</div>
      </footer>
    </div>
  );
}

export default App;
