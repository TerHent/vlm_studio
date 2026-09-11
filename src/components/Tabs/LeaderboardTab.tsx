import React, { useState } from 'react';
import { EvaluationReport } from '../../types';
import { Search, Download, ChevronDown, ChevronRight, BarChart2, TrendingUp, Info } from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
  BarChart,
  Bar,
} from 'recharts';

interface LeaderboardTabProps {
  reports: Record<string, EvaluationReport>;
  selectedReportKey: string;
  onSelectReportKey: (key: string) => void;
}

export const LeaderboardTab: React.FC<LeaderboardTabProps> = ({
  reports,
  selectedReportKey,
  onSelectReportKey,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [showMetadata, setShowMetadata] = useState(false);

  const reportKeys = Object.keys(reports);
  const activeReport = reports[selectedReportKey] || reports[reportKeys[0]];

  // Leaderboard rows
  const leaderboardRows = reportKeys
    .map((key) => {
      const rep = reports[key];
      const model = rep.metadata?.model_name || key;
      const dataset = rep.metadata?.dataset_path ? rep.metadata.dataset_path.split('/').pop() || 'N/A' : 'N/A';
      return {
        key,
        model,
        dataset,
        split: rep.metadata?.dataset_split || 'all',
        mapCoco: rep.mAP_50_95,
        ap50: rep.mAP_50,
        ap75: rep.mAP_75,
        samples: rep.image_count,
        gtBoxes: rep.total_ground_truths,
        preds: rep.total_predictions,
      };
    })
    .filter((row) => {
      if (!searchQuery.trim()) return true;
      const q = searchQuery.toLowerCase();
      return row.model.toLowerCase().includes(q) || row.dataset.toLowerCase().includes(q);
    })
    .sort((a, b) => b.mapCoco - a.mapCoco);

  // IoU degradation chart data
  const iouChartData = activeReport?.per_iou
    ? Object.entries(activeReport.per_iou)
        .map(([k, v]) => ({
          iou: parseFloat(k.replace('iou_', '')),
          mAP: parseFloat(v.mAP.toFixed(4)),
          precision: parseFloat(v.mean_precision.toFixed(4)),
          recall: parseFloat(v.mean_recall.toFixed(4)),
        }))
        .sort((a, b) => a.iou - b.iou)
    : [];

  // Category performance bar chart data
  const categoryChartData = activeReport?.per_class_summary
    ? Object.entries(activeReport.per_class_summary)
        .map(([cls, vals]) => ({
          category: cls,
          mAP: parseFloat(vals.AP_50_95.toFixed(4)),
        }))
        .sort((a, b) => a.mAP - b.mAP)
    : [];

  const handleDownloadReport = () => {
    if (!activeReport) return;
    const blob = new Blob([JSON.stringify(activeReport, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = selectedReportKey || 'evaluation_report.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Benchmark Runs Section */}
      <div>
        <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-zinc-400">
            <span className="text-indigo-400">◈</span> Model Benchmark Run(s)
          </div>
          <div className="relative min-w-[260px]">
            <Search className="w-3.5 h-3.5 absolute left-3 top-2.5 text-zinc-500" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Filter runs by model or dataset..."
              className="w-full pl-8 pr-3 py-1.5 bg-[#12141c] border border-white/10 rounded-md text-xs text-zinc-200 placeholder-zinc-500 focus:outline-none focus:border-indigo-500/50"
            />
          </div>
        </div>

        {/* Table of benchmark runs */}
        <div className="overflow-x-auto border border-white/10 rounded-lg bg-[#12141c]">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-white/10 text-zinc-400 bg-white/[0.02]">
                <th className="py-2.5 px-3 font-medium">Model</th>
                <th className="py-2.5 px-3 font-medium">Dataset</th>
                <th className="py-2.5 px-3 font-medium">Split</th>
                <th className="py-2.5 px-3 font-medium">mAP [0.50-0.95]</th>
                <th className="py-2.5 px-3 font-medium">AP@.50</th>
                <th className="py-2.5 px-3 font-medium">AP@.75</th>
                <th className="py-2.5 px-3 font-medium text-right">Samples</th>
                <th className="py-2.5 px-3 font-medium text-right">GT Boxes</th>
                <th className="py-2.5 px-3 font-medium text-right">Preds</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {leaderboardRows.length === 0 ? (
                <tr>
                  <td colSpan={9} className="py-6 text-center text-zinc-500">
                    No benchmark runs found matching filter.
                  </td>
                </tr>
              ) : (
                leaderboardRows.map((row) => {
                  const isSelected = row.key === selectedReportKey;
                  return (
                    <tr
                      key={row.key}
                      onClick={() => onSelectReportKey(row.key)}
                      className={`cursor-pointer transition-colors ${
                        isSelected ? 'bg-indigo-950/40 text-white' : 'hover:bg-white/[0.03] text-zinc-300'
                      }`}
                    >
                      <td className="py-2 px-3 font-mono font-medium text-indigo-300 flex items-center gap-1.5">
                        {isSelected && <span className="w-1.5 h-1.5 rounded-full bg-indigo-400" />}
                        {row.model}
                      </td>
                      <td className="py-2 px-3">{row.dataset}</td>
                      <td className="py-2 px-3 text-zinc-400">{row.split}</td>
                      <td className="py-2 px-3 font-mono">
                        <div className="flex items-center gap-2">
                          <span className="w-12">{row.mapCoco.toFixed(4)}</span>
                          <div className="flex-1 min-w-[60px] h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-indigo-500 rounded-full"
                              style={{ width: `${Math.min(100, Math.max(0, row.mapCoco * 100))}%` }}
                            />
                          </div>
                        </div>
                      </td>
                      <td className="py-2 px-3 font-mono text-emerald-400">{row.ap50.toFixed(4)}</td>
                      <td className="py-2 px-3 font-mono text-amber-400">{row.ap75.toFixed(4)}</td>
                      <td className="py-2 px-3 font-mono text-right text-zinc-400">{row.samples}</td>
                      <td className="py-2 px-3 font-mono text-right text-zinc-400">{row.gtBoxes}</td>
                      <td className="py-2 px-3 font-mono text-right text-zinc-400">{row.preds}</td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Detailed Evaluation Report Section */}
      {activeReport && (
        <div className="space-y-4 pt-2">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-zinc-400">
              <span className="text-indigo-400">◈</span> Detailed Evaluation Report:
              <select
                value={selectedReportKey}
                onChange={(e) => onSelectReportKey(e.target.value)}
                className="ml-2 bg-[#12141c] border border-white/10 rounded px-2 py-1 text-xs text-zinc-200 font-mono focus:outline-none focus:border-indigo-500"
              >
                {reportKeys.map((k) => (
                  <option key={k} value={k}>
                    {k}
                  </option>
                ))}
              </select>
            </div>
            <button
              onClick={handleDownloadReport}
              className="flex items-center gap-1.5 px-3 py-1 bg-[#161822] hover:bg-white/10 border border-white/10 rounded text-xs text-zinc-300 transition"
            >
              <Download className="w-3.5 h-3.5" />
              Download Report JSON
            </button>
          </div>

          {/* 4 KPI Metric Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
            <div className="bg-[#12141c] border border-white/10 rounded-lg p-4 shadow-sm hover:border-white/20 transition">
              <div className="flex items-center justify-between text-[11px] font-semibold uppercase tracking-wider text-zinc-400 mb-1.5">
                <span>COCO Primary AP</span>
                <span className="text-indigo-400 font-mono">mAP@[.50:.95]</span>
              </div>
              <div className="text-2xl font-bold font-mono text-zinc-100 tracking-tight">
                {activeReport.mAP_50_95.toFixed(4)}
              </div>
              <div className="text-[11px] text-zinc-500 mt-1">10-step ladder mean [0.50:0.05:0.95]</div>
            </div>

            <div className="bg-[#12141c] border border-white/10 rounded-lg p-4 shadow-sm hover:border-white/20 transition">
              <div className="flex items-center justify-between text-[11px] font-semibold uppercase tracking-wider text-zinc-400 mb-1.5">
                <span>PASCAL VOC</span>
                <span className="text-emerald-400 font-mono">AP@.50</span>
              </div>
              <div className="text-2xl font-bold font-mono text-zinc-100 tracking-tight">
                {activeReport.mAP_50.toFixed(4)}
              </div>
              <div className="text-[11px] text-zinc-500 mt-1">
                Precision@50:{' '}
                {activeReport.per_iou?.['iou_0.50']?.mean_precision.toFixed(3) || '0.000'}
              </div>
            </div>

            <div className="bg-[#12141c] border border-white/10 rounded-lg p-4 shadow-sm hover:border-white/20 transition">
              <div className="flex items-center justify-between text-[11px] font-semibold uppercase tracking-wider text-zinc-400 mb-1.5">
                <span>Strict Loc</span>
                <span className="text-amber-400 font-mono">AP@.75</span>
              </div>
              <div className="text-2xl font-bold font-mono text-zinc-100 tracking-tight">
                {activeReport.mAP_75.toFixed(4)}
              </div>
              <div className="text-[11px] text-zinc-500 mt-1">
                Precision@75:{' '}
                {activeReport.per_iou?.['iou_0.75']?.mean_precision.toFixed(3) || '0.000'}
              </div>
            </div>

            <div className="bg-[#12141c] border border-white/10 rounded-lg p-4 shadow-sm hover:border-white/20 transition">
              <div className="flex items-center justify-between text-[11px] font-semibold uppercase tracking-wider text-zinc-400 mb-1.5">
                <span>Evaluation Set</span>
                <span className="text-zinc-400 font-mono">{activeReport.image_count} Images</span>
              </div>
              <div className="text-2xl font-bold font-mono text-zinc-100 tracking-tight">
                {activeReport.total_ground_truths}
              </div>
              <div className="text-[11px] text-zinc-500 mt-1">
                {activeReport.total_predictions} Predictions Total
              </div>
            </div>
          </div>

          {/* Charts Row */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* IoU Degradation Curve */}
            <div className="bg-[#12141c] border border-white/10 rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-semibold uppercase tracking-wider text-zinc-400 flex items-center gap-1.5">
                  <TrendingUp className="w-3.5 h-3.5 text-indigo-400" /> IoU Threshold Degradation Curve
                </span>
                <span className="text-[11px] text-zinc-500">COCO 10-Step Ladder</span>
              </div>
              <div className="h-56 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={iouChartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                    <CartesianGrid stroke="rgba(255,255,255,0.05)" />
                    <XAxis
                      dataKey="iou"
                      tick={{ fill: '#71717a', fontSize: 10 }}
                      tickFormatter={(v) => v.toFixed(2)}
                      stroke="rgba(255,255,255,0.1)"
                    />
                    <YAxis
                      domain={[0, 1]}
                      tick={{ fill: '#71717a', fontSize: 10 }}
                      stroke="rgba(255,255,255,0.1)"
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#161822',
                        borderColor: 'rgba(255,255,255,0.15)',
                        fontSize: '11px',
                        color: '#f4f4f5',
                        borderRadius: '6px',
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '6px' }} />
                    <Line
                      type="monotone"
                      dataKey="mAP"
                      stroke="#818cf8"
                      strokeWidth={2}
                      name="Mean AP"
                      dot={{ r: 3, fill: '#6366f1' }}
                    />
                    <Line
                      type="monotone"
                      dataKey="precision"
                      stroke="#10b981"
                      strokeWidth={1.5}
                      strokeDasharray="4 4"
                      name="Precision"
                      dot={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="recall"
                      stroke="#f59e0b"
                      strokeWidth={1.5}
                      strokeDasharray="2 2"
                      name="Recall"
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Category Performance Bar Chart */}
            <div className="bg-[#12141c] border border-white/10 rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-semibold uppercase tracking-wider text-zinc-400 flex items-center gap-1.5">
                  <BarChart2 className="w-3.5 h-3.5 text-indigo-400" /> Category Performance (COCO mAP)
                </span>
                <span className="text-[11px] text-zinc-500">AP@[.50:.95]</span>
              </div>
              <div className="h-56 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={categoryChartData}
                    layout="vertical"
                    margin={{ top: 10, right: 15, left: 20, bottom: 0 }}
                  >
                    <CartesianGrid stroke="rgba(255,255,255,0.05)" horizontal={false} />
                    <XAxis
                      type="number"
                      domain={[0, 1]}
                      tick={{ fill: '#71717a', fontSize: 10 }}
                      stroke="rgba(255,255,255,0.1)"
                    />
                    <YAxis
                      dataKey="category"
                      type="category"
                      tick={{ fill: '#cbd5e1', fontSize: 10 }}
                      width={100}
                      stroke="rgba(255,255,255,0.1)"
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#161822',
                        borderColor: 'rgba(255,255,255,0.15)',
                        fontSize: '11px',
                        color: '#f4f4f5',
                        borderRadius: '6px',
                      }}
                      formatter={(val: any) => [val.toFixed(4), 'COCO mAP']}
                    />
                    <Bar dataKey="mAP" fill="#6366f1" radius={[0, 4, 4, 0]} opacity={0.88} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Detailed Per-Category Breakdown Table */}
          <div className="bg-[#12141c] border border-white/10 rounded-lg p-4">
            <div className="text-xs font-semibold uppercase tracking-wider text-zinc-400 mb-3 flex items-center gap-2">
              <span className="text-indigo-400">◈</span> Per-Category Metric Breakdown
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-white/10 text-zinc-400 bg-white/[0.02]">
                    <th className="py-2 px-3 font-medium">Category</th>
                    <th className="py-2 px-3 font-medium">mAP [0.50-0.95]</th>
                    <th className="py-2 px-3 font-medium">AP@.50</th>
                    <th className="py-2 px-3 font-medium">AP@.75</th>
                    <th className="py-2 px-3 font-medium">Precision@.50</th>
                    <th className="py-2 px-3 font-medium">Recall@.50</th>
                    <th className="py-2 px-3 font-medium text-right">Ground Truth</th>
                    <th className="py-2 px-3 font-medium text-right">Predictions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5 font-mono">
                  {Object.entries(activeReport.per_class_summary || {})
                    .sort((a, b) => b[1].AP_50_95 - a[1].AP_50_95)
                    .map(([cls, v]) => (
                      <tr key={cls} className="hover:bg-white/[0.02] transition-colors">
                        <td className="py-2 px-3 font-sans font-medium text-zinc-200">{cls}</td>
                        <td className="py-2 px-3">
                          <div className="flex items-center gap-2">
                            <span className="w-12 text-indigo-300">{v.AP_50_95.toFixed(4)}</span>
                            <div className="w-16 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                              <div
                                className="h-full bg-indigo-500 rounded-full"
                                style={{ width: `${Math.min(100, Math.max(0, v.AP_50_95 * 100))}%` }}
                              />
                            </div>
                          </div>
                        </td>
                        <td className="py-2 px-3 text-emerald-400">{v.AP_50.toFixed(4)}</td>
                        <td className="py-2 px-3 text-amber-400">{v.AP_75.toFixed(4)}</td>
                        <td className="py-2 px-3 text-zinc-300">{v.precision_50.toFixed(4)}</td>
                        <td className="py-2 px-3 text-zinc-300">{v.recall_50.toFixed(4)}</td>
                        <td className="py-2 px-3 text-right text-zinc-400">{v.num_ground_truth}</td>
                        <td className="py-2 px-3 text-right text-zinc-400">{v.num_predictions}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Collapsible Metadata Accordion */}
          <div className="border border-white/10 rounded-lg overflow-hidden bg-[#12141c]">
            <button
              onClick={() => setShowMetadata(!showMetadata)}
              className="w-full flex items-center justify-between px-4 py-2.5 text-xs font-medium text-zinc-300 hover:bg-white/[0.03] transition"
            >
              <span className="flex items-center gap-2">
                <Info className="w-3.5 h-3.5 text-zinc-400" />
                Runtime Configuration & Metadata
              </span>
              {showMetadata ? <ChevronDown className="w-4 h-4 text-zinc-500" /> : <ChevronRight className="w-4 h-4 text-zinc-500" />}
            </button>
            {showMetadata && (
              <div className="p-4 border-t border-white/10 bg-black/20 text-xs">
                <table className="w-full text-left font-mono">
                  <tbody>
                    {Object.entries(activeReport.metadata || {}).map(([k, val]) => (
                      <tr key={k} className="border-b border-white/5">
                        <td className="py-1.5 px-2 text-zinc-400 w-1/3">{k}</td>
                        <td className="py-1.5 px-2 text-zinc-200">{String(val)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
