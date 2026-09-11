import React, { useState, useMemo } from 'react';
import { PredictionCache, GroundTruthBox, PredictionBox, BoundingBox } from '../../types';
import { getCategoryColor } from '../../data/benchmarkData';
import { ChevronLeft, ChevronRight, Shuffle, Eye, Sliders, CheckSquare, Square } from 'lucide-react';

interface VisualInspectorTabProps {
  caches: PredictionCache[];
  selectedCacheId: string;
  onSelectCacheId: (id: string) => void;
}

export const VisualInspectorTab: React.FC<VisualInspectorTabProps> = ({
  caches,
  selectedCacheId,
  onSelectCacheId,
}) => {
  const [sampleIndex, setSampleIndex] = useState(0);
  const [layoutMode, setLayoutMode] = useState<'side_by_side' | 'gt_only' | 'pred_only'>('side_by_side');
  const [confidenceFilter, setConfidenceFilter] = useState(0.0);
  const [selectedClasses, setSelectedClasses] = useState<Set<string>>(new Set());
  const [hoveredBox, setHoveredBox] = useState<{ type: 'gt' | 'pred'; idx: number } | null>(null);

  const activeCache = caches.find((c) => c.id === selectedCacheId) || caches[0];
  const samples = activeCache?.samples || [];
  const currentSample = samples[sampleIndex] || samples[0];

  // Extract all distinct classes across active cache
  const allClasses = useMemo(() => {
    const set = new Set<string>();
    samples.forEach((s) => {
      s.ground_truths.forEach((g) => set.add(g.label));
      s.predictions.forEach((p) => set.add(p.label));
    });
    return Array.from(set).sort();
  }, [samples]);

  // Sync selected classes when cache changes
  React.useEffect(() => {
    setSelectedClasses(new Set(allClasses));
    setSampleIndex(0);
  }, [selectedCacheId, allClasses.join(',')]);

  const toggleClass = (cls: string) => {
    const next = new Set(selectedClasses);
    if (next.has(cls)) {
      next.delete(cls);
    } else {
      next.add(cls);
    }
    setSelectedClasses(next);
  };

  const selectAllClasses = () => setSelectedClasses(new Set(allClasses));
  const deselectAllClasses = () => setSelectedClasses(new Set());

  // Filtered detections for active sample
  const filteredGTs = useMemo(() => {
    if (!currentSample) return [];
    return currentSample.ground_truths.filter((gt) => selectedClasses.has(gt.label));
  }, [currentSample, selectedClasses]);

  const filteredPreds = useMemo(() => {
    if (!currentSample) return [];
    return currentSample.predictions.filter((p) => {
      const score = p.score ?? 1.0;
      return score >= confidenceFilter && selectedClasses.has(p.label);
    });
  }, [currentSample, confidenceFilter, selectedClasses]);

  const handlePrev = () => setSampleIndex((prev) => Math.max(0, prev - 1));
  const handleNext = () => setSampleIndex((prev) => Math.min(samples.length - 1, prev + 1));
  const handleRandom = () => {
    if (samples.length <= 1) return;
    const rand = Math.floor(Math.random() * samples.length);
    setSampleIndex(rand);
  };

  // Helper to render bounding boxes over an SVG or container
  const renderBoxesOverlay = (
    boxes: { bbox: BoundingBox; label: string; score?: number }[],
    type: 'gt' | 'pred'
  ) => {
    return boxes.map((item, idx) => {
      const [ymin, xmin, ymax, xmax] = item.bbox;
      const left = `${xmin * 100}%`;
      const top = `${ymin * 100}%`;
      const width = `${(xmax - xmin) * 100}%`;
      const height = `${(ymax - ymin) * 100}%`;
      const color = getCategoryColor(item.label);
      const isHovered = hoveredBox?.type === type && hoveredBox?.idx === idx;

      return (
        <div
          key={`${type}-${idx}`}
          onMouseEnter={() => setHoveredBox({ type, idx })}
          onMouseLeave={() => setHoveredBox(null)}
          style={{
            position: 'absolute',
            left,
            top,
            width,
            height,
            border: `2px ${type === 'gt' ? 'solid' : 'dashed'} ${color}`,
            backgroundColor: isHovered ? `${color}33` : `${color}15`,
            boxShadow: isHovered ? `0 0 12px ${color}88` : 'none',
            transition: 'all 0.15s ease',
            pointerEvents: 'auto',
          }}
          className="group cursor-pointer"
        >
          {/* Label tag */}
          <div
            style={{ backgroundColor: color }}
            className="absolute -top-5 left-0 px-1.5 py-0.5 text-[10px] font-semibold text-white rounded-t whitespace-nowrap shadow-sm flex items-center gap-1 z-10"
          >
            <span>{item.label}</span>
            {item.score !== undefined && (
              <span className="opacity-90 font-mono text-[9px]">
                {(item.score * 100).toFixed(0)}%
              </span>
            )}
          </div>
        </div>
      );
    });
  };

  const renderImageViewport = (
    title: string,
    boxes: { bbox: BoundingBox; label: string; score?: number }[],
    type: 'gt' | 'pred'
  ) => {
    return (
      <div className="flex-1 min-w-[320px] bg-[#090a0d] border border-white/10 rounded-lg overflow-hidden flex flex-col shadow-md">
        <div className="bg-[#12141c] border-b border-white/10 px-3 py-2 flex items-center justify-between">
          <span className="text-xs font-semibold uppercase tracking-wider text-zinc-300">
            {title}
          </span>
          <span className="text-[11px] font-mono text-zinc-400 bg-white/5 px-2 py-0.5 rounded">
            {boxes.length} {boxes.length === 1 ? 'Box' : 'Boxes'}
          </span>
        </div>

        <div className="relative w-full aspect-[4/3] bg-[#0f1117] flex items-center justify-center overflow-hidden">
          {currentSample?.image_svg ? (
            <div
              className="absolute inset-0 w-full h-full"
              dangerouslySetInnerHTML={{ __html: currentSample.image_svg }}
            />
          ) : (
            <div className="text-zinc-600 text-xs">No image asset available</div>
          )}

          {/* Bounding Boxes Layer */}
          <div className="absolute inset-0 w-full h-full pointer-events-none">
            {renderBoxesOverlay(boxes, type)}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-5">
      {/* Top Configuration Bar */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 bg-[#12141c] border border-white/10 p-3.5 rounded-lg">
        <div>
          <label className="block text-xs font-semibold uppercase tracking-wider text-zinc-400 mb-1.5">
            Prediction Cache
          </label>
          <select
            value={selectedCacheId}
            onChange={(e) => onSelectCacheId(e.target.value)}
            className="w-full bg-[#161822] border border-white/10 rounded px-2.5 py-1.5 text-xs text-zinc-200 font-mono focus:outline-none focus:border-indigo-500"
          >
            {caches.map((c) => (
              <option key={c.id} value={c.id}>
                {c.name}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-xs font-semibold uppercase tracking-wider text-zinc-400 mb-1.5">
            Viewport Layout
          </label>
          <div className="grid grid-cols-3 gap-1.5">
            <button
              onClick={() => setLayoutMode('side_by_side')}
              className={`py-1.5 px-2 rounded text-xs font-medium transition cursor-pointer ${
                layoutMode === 'side_by_side'
                  ? 'bg-indigo-600 text-white shadow'
                  : 'bg-[#161822] text-zinc-400 hover:text-zinc-200 border border-white/5'
              }`}
            >
              Side-by-Side
            </button>
            <button
              onClick={() => setLayoutMode('gt_only')}
              className={`py-1.5 px-2 rounded text-xs font-medium transition cursor-pointer ${
                layoutMode === 'gt_only'
                  ? 'bg-emerald-600 text-white shadow'
                  : 'bg-[#161822] text-zinc-400 hover:text-zinc-200 border border-white/5'
              }`}
            >
              Ground Truth
            </button>
            <button
              onClick={() => setLayoutMode('pred_only')}
              className={`py-1.5 px-2 rounded text-xs font-medium transition cursor-pointer ${
                layoutMode === 'pred_only'
                  ? 'bg-indigo-600 text-white shadow'
                  : 'bg-[#161822] text-zinc-400 hover:text-zinc-200 border border-white/5'
              }`}
            >
              Predictions
            </button>
          </div>
        </div>
      </div>

      {/* Sample Navigation Strip */}
      <div className="flex flex-wrap items-center justify-between gap-3 bg-[#12141c] border border-white/10 px-4 py-2.5 rounded-lg">
        <div className="flex items-center gap-2">
          <button
            onClick={handlePrev}
            disabled={sampleIndex <= 0}
            className="flex items-center gap-1 px-3 py-1.5 bg-[#161822] disabled:opacity-40 hover:bg-white/10 border border-white/10 rounded text-xs text-zinc-200 font-medium transition cursor-pointer disabled:cursor-not-allowed"
          >
            <ChevronLeft className="w-3.5 h-3.5" /> Prev
          </button>

          <button
            onClick={handleNext}
            disabled={sampleIndex >= samples.length - 1}
            className="flex items-center gap-1 px-3 py-1.5 bg-[#161822] disabled:opacity-40 hover:bg-white/10 border border-white/10 rounded text-xs text-zinc-200 font-medium transition cursor-pointer disabled:cursor-not-allowed"
          >
            Next <ChevronRight className="w-3.5 h-3.5" />
          </button>

          <button
            onClick={handleRandom}
            className="flex items-center gap-1 px-2.5 py-1.5 bg-[#161822] hover:bg-white/10 border border-white/10 rounded text-xs text-zinc-300 transition cursor-pointer"
          >
            <Shuffle className="w-3.5 h-3.5" /> Random
          </button>
        </div>

        <div className="flex items-center gap-2 font-mono text-xs">
          <span className="text-zinc-400">Sample</span>
          <span className="bg-indigo-950/60 text-indigo-300 px-2 py-0.5 rounded border border-indigo-500/20 font-semibold">
            #{sampleIndex + 1} of {samples.length}
          </span>
          <span className="text-zinc-600">|</span>
          <span className="text-zinc-300 font-sans">{currentSample?.file_name}</span>
        </div>

        {samples.length > 1 && (
          <div className="flex items-center gap-2 w-full sm:w-auto">
            <input
              type="range"
              min={0}
              max={samples.length - 1}
              value={sampleIndex}
              onChange={(e) => setSampleIndex(parseInt(e.target.value))}
              className="w-36 h-1.5 bg-zinc-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
            />
          </div>
        )}
      </div>

      {/* Dynamic Filters Bar */}
      <div className="bg-[#12141c] border border-white/10 p-3.5 rounded-lg space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <Sliders className="w-4 h-4 text-indigo-400" />
            <span className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
              Confidence Filter:
            </span>
            <input
              type="range"
              min={0.0}
              max={1.0}
              step={0.05}
              value={confidenceFilter}
              onChange={(e) => setConfidenceFilter(parseFloat(e.target.value))}
              className="w-36 h-1.5 bg-zinc-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
            />
            <span className="font-mono text-xs font-semibold text-zinc-200 bg-[#161822] px-2 py-0.5 rounded border border-white/10">
              {confidenceFilter.toFixed(2)}
            </span>
          </div>

          <div className="flex items-center gap-2 text-xs">
            <button
              onClick={selectAllClasses}
              className="text-zinc-400 hover:text-zinc-200 underline cursor-pointer"
            >
              Select All
            </button>
            <span className="text-zinc-600">•</span>
            <button
              onClick={deselectAllClasses}
              className="text-zinc-400 hover:text-zinc-200 underline cursor-pointer"
            >
              Clear
            </button>
          </div>
        </div>

        {/* Category Pills */}
        <div className="flex flex-wrap gap-1.5 pt-1">
          {allClasses.map((cls) => {
            const isSelected = selectedClasses.has(cls);
            const color = getCategoryColor(cls);
            return (
              <button
                key={cls}
                onClick={() => toggleClass(cls)}
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium transition cursor-pointer border ${
                  isSelected
                    ? 'border-transparent text-white shadow-sm'
                    : 'border-white/10 text-zinc-500 bg-transparent opacity-60'
                }`}
                style={{
                  backgroundColor: isSelected ? `${color}33` : undefined,
                  borderColor: isSelected ? color : undefined,
                }}
              >
                <span
                  className="w-2 h-2 rounded-full"
                  style={{ backgroundColor: color }}
                />
                <span>{cls}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Visual Renderers */}
      <div className="flex flex-wrap gap-4">
        {(layoutMode === 'side_by_side' || layoutMode === 'gt_only') &&
          renderImageViewport('Ground Truth Objects', filteredGTs, 'gt')}

        {(layoutMode === 'side_by_side' || layoutMode === 'pred_only') &&
          renderImageViewport('Model Predictions', filteredPreds, 'pred')}
      </div>

      {/* Detection Breakdown Tables */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Ground Truth Breakdown */}
        <div className="bg-[#12141c] border border-white/10 rounded-lg p-3.5">
          <div className="text-xs font-semibold uppercase tracking-wider text-zinc-400 mb-2.5 flex items-center justify-between">
            <span>● Ground Truth Objects ({filteredGTs.length})</span>
            <span className="text-[10px] text-zinc-500 font-mono">[ymin, xmin, ymax, xmax]</span>
          </div>

          <div className="overflow-x-auto max-h-56 overflow-y-auto">
            {filteredGTs.length === 0 ? (
              <div className="py-4 text-center text-xs text-zinc-500">No ground truth boxes matching filters.</div>
            ) : (
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-white/10 text-zinc-500">
                    <th className="pb-1.5">Class</th>
                    <th className="pb-1.5 font-mono text-right">Bounding Box</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5 font-mono text-[11px]">
                  {filteredGTs.map((g, idx) => {
                    const isHovered = hoveredBox?.type === 'gt' && hoveredBox?.idx === idx;
                    return (
                      <tr
                        key={idx}
                        onMouseEnter={() => setHoveredBox({ type: 'gt', idx })}
                        onMouseLeave={() => setHoveredBox(null)}
                        className={`transition ${isHovered ? 'bg-indigo-950/40 text-white' : 'text-zinc-300'}`}
                      >
                        <td className="py-1.5 font-sans font-medium flex items-center gap-1.5">
                          <span
                            className="w-2 h-2 rounded-full"
                            style={{ backgroundColor: getCategoryColor(g.label) }}
                          />
                          {g.label}
                        </td>
                        <td className="py-1.5 text-right text-zinc-400">
                          [{g.bbox.map((v) => v.toFixed(3)).join(', ')}]
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>
        </div>

        {/* Model Predictions Breakdown */}
        <div className="bg-[#12141c] border border-white/10 rounded-lg p-3.5">
          <div className="text-xs font-semibold uppercase tracking-wider text-zinc-400 mb-2.5 flex items-center justify-between">
            <span>● Model Predictions ({filteredPreds.length})</span>
            <span className="text-[10px] text-zinc-500 font-mono">[ymin, xmin, ymax, xmax]</span>
          </div>

          <div className="overflow-x-auto max-h-56 overflow-y-auto">
            {filteredPreds.length === 0 ? (
              <div className="py-4 text-center text-xs text-zinc-500">No predictions matching filters.</div>
            ) : (
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-white/10 text-zinc-500">
                    <th className="pb-1.5">Class</th>
                    <th className="pb-1.5">Confidence</th>
                    <th className="pb-1.5 font-mono text-right">Bounding Box</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5 font-mono text-[11px]">
                  {filteredPreds.map((p, idx) => {
                    const isHovered = hoveredBox?.type === 'pred' && hoveredBox?.idx === idx;
                    const score = p.score ?? 1.0;
                    return (
                      <tr
                        key={idx}
                        onMouseEnter={() => setHoveredBox({ type: 'pred', idx })}
                        onMouseLeave={() => setHoveredBox(null)}
                        className={`transition ${isHovered ? 'bg-indigo-950/40 text-white' : 'text-zinc-300'}`}
                      >
                        <td className="py-1.5 font-sans font-medium flex items-center gap-1.5">
                          <span
                            className="w-2 h-2 rounded-full"
                            style={{ backgroundColor: getCategoryColor(p.label) }}
                          />
                          {p.label}
                        </td>
                        <td className="py-1.5">
                          <div className="flex items-center gap-1.5">
                            <span className="w-10">{(score * 100).toFixed(0)}%</span>
                            <div className="w-12 h-1 bg-zinc-800 rounded-full overflow-hidden">
                              <div
                                className="h-full bg-indigo-500"
                                style={{ width: `${score * 100}%` }}
                              />
                            </div>
                          </div>
                        </td>
                        <td className="py-1.5 text-right text-zinc-400">
                          [{p.bbox.map((v) => v.toFixed(3)).join(', ')}]
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
