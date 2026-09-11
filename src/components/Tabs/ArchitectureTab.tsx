import React from 'react';
import { Layers, Workflow, ShieldCheck, Calculator, GitBranch, Cpu } from 'lucide-react';

export const ArchitectureTab: React.FC = () => {
  return (
    <div className="space-y-6">
      {/* Intro Header */}
      <div className="bg-[#12141c] border border-white/10 rounded-lg p-5">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-zinc-200 mb-1 flex items-center gap-2">
          <Workflow className="w-4 h-4 text-indigo-400" /> VLM Studio System Architecture & Topology
        </h2>
        <p className="text-xs text-zinc-400 leading-relaxed">
          A decoupled, robust framework engineered for benchmarking Vision-Language Models (VLMs) against object detection ground truths. Designed for offline reproducibility, resilient JSON output parsing, and standard COCO/PASCAL metric parity.
        </p>
      </div>

      {/* Macro-Architecture Diagram & Lineage */}
      <div className="bg-[#12141c] border border-white/10 rounded-lg p-5 space-y-4">
        <div className="text-xs font-semibold uppercase tracking-wider text-zinc-400 flex items-center gap-2">
          <Layers className="w-3.5 h-3.5 text-indigo-400" /> Macro Architecture & Data Lineage
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-2 text-center text-xs">
          <div className="p-3 bg-[#161822] border border-white/10 rounded-lg flex flex-col justify-center">
            <span className="text-zinc-400 font-mono text-[10px]">STAGE 1</span>
            <span className="font-semibold text-zinc-200 mt-1">Dataset Layer</span>
            <span className="text-[11px] text-zinc-500 mt-0.5">COCO / HuggingFace / Local</span>
          </div>

          <div className="hidden md:flex items-center justify-center text-indigo-400 font-bold">→</div>

          <div className="p-3 bg-[#161822] border border-white/10 rounded-lg flex flex-col justify-center">
            <span className="text-zinc-400 font-mono text-[10px]">STAGE 2</span>
            <span className="font-semibold text-indigo-300 mt-1">Model Adapter</span>
            <span className="text-[11px] text-zinc-500 mt-0.5">LM Studio / Transformers</span>
          </div>

          <div className="hidden md:flex items-center justify-center text-indigo-400 font-bold">→</div>

          <div className="p-3 bg-[#161822] border border-white/10 rounded-lg flex flex-col justify-center">
            <span className="text-zinc-400 font-mono text-[10px]">STAGE 3</span>
            <span className="font-semibold text-emerald-400 mt-1">Prediction Cache</span>
            <span className="text-[11px] text-zinc-500 mt-0.5">Durable JSON Storage</span>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-2">
          <div className="p-3.5 bg-black/20 border border-white/5 rounded-lg space-y-1">
            <div className="font-semibold text-xs text-zinc-200 flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-indigo-400" />
              Decoupled Inference & Evaluation
            </div>
            <p className="text-[11px] text-zinc-400">
              Inference is decoupled from evaluation. Predictions are saved as standalone JSON caches, allowing repeated offline evaluation at varying IoU thresholds without re-querying models or expending GPU cycles.
            </p>
          </div>

          <div className="p-3.5 bg-black/20 border border-white/5 rounded-lg space-y-1">
            <div className="font-semibold text-xs text-zinc-200 flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              Unified Normalized Coordinates
            </div>
            <p className="text-[11px] text-zinc-400">
              All adapters convert heterogeneous model output representations (0-1000 integers, unnormalized absolute pixels, or string brackets) into standard normalized floats <code>[ymin, xmin, ymax, xmax]</code> in <code>[0.0, 1.0]</code>.
            </p>
          </div>

          <div className="p-3.5 bg-black/20 border border-white/5 rounded-lg space-y-1">
            <div className="font-semibold text-xs text-zinc-200 flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-amber-400" />
              Stateless Metric Evaluation
            </div>
            <p className="text-[11px] text-zinc-400">
              <code>DetectionEvaluator</code> processes batches in memory, accumulates predictions and ground truths, matches detections per class using greedy IoU sorting, and integrates precision-recall curves.
            </p>
          </div>
        </div>
      </div>

      {/* 3 Execution Modes */}
      <div className="bg-[#12141c] border border-white/10 rounded-lg p-5 space-y-4">
        <div className="text-xs font-semibold uppercase tracking-wider text-zinc-400 flex items-center gap-2">
          <GitBranch className="w-3.5 h-3.5 text-indigo-400" /> Execution Modes & Lifecycle
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div className="p-3.5 bg-[#161822] border border-white/10 rounded-lg space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-mono text-xs font-bold text-indigo-400">--mode all</span>
              <span className="text-[10px] bg-indigo-950/60 text-indigo-300 px-2 py-0.5 rounded border border-indigo-500/20">End-to-End</span>
            </div>
            <p className="text-[11px] text-zinc-400">
              Runs model inference over dataset, caches normalized predictions to disk, and immediately computes evaluation metrics and reports.
            </p>
          </div>

          <div className="p-3.5 bg-[#161822] border border-white/10 rounded-lg space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-mono text-xs font-bold text-amber-400">--mode predict</span>
              <span className="text-[10px] bg-amber-950/60 text-amber-300 px-2 py-0.5 rounded border border-amber-500/20">Inference Only</span>
            </div>
            <p className="text-[11px] text-zinc-400">
              Queries the model and saves raw predictions to cache. Skips metric evaluation; useful for running remote inference jobs or distributed workers.
            </p>
          </div>

          <div className="p-3.5 bg-[#161822] border border-white/10 rounded-lg space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-mono text-xs font-bold text-emerald-400">--mode evaluate</span>
              <span className="text-[10px] bg-emerald-950/60 text-emerald-300 px-2 py-0.5 rounded border border-emerald-500/20">Metrics Only</span>
            </div>
            <p className="text-[11px] text-zinc-400">
              Loads an existing prediction cache and computes metrics without loading any models or querying inference endpoints. Runs in sub-second time.
            </p>
          </div>
        </div>
      </div>

      {/* Defensive Output Parsing Engine */}
      <div className="bg-[#12141c] border border-white/10 rounded-lg p-5 space-y-3">
        <div className="text-xs font-semibold uppercase tracking-wider text-zinc-400 flex items-center gap-2">
          <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" /> Defensive Output Parsing Strategy
        </div>
        <p className="text-xs text-zinc-400">
          Small open-weight VLMs frequently produce malformed structured outputs. The parsing engine employs multi-stage fallback pipelines:
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
          <div className="p-3 bg-black/20 border border-white/5 rounded space-y-1">
            <strong className="text-zinc-200">1. Strict JSON Extraction</strong>
            <p className="text-zinc-400 text-[11px]">
              Extracts outermost markdown blocks (<code>```json ... ```</code>) or locates bracket boundaries <code>[ ... ]</code>.
            </p>
          </div>
          <div className="p-3 bg-black/20 border border-white/5 rounded space-y-1">
            <strong className="text-zinc-200">2. Regex Key Permutation</strong>
            <p className="text-zinc-400 text-[11px]">
              Tolerates alternate keys: <code>box_2d</code>, <code>bbox</code>, <code>bndbox</code>, <code>coordinates</code>, and single-quoted JSON.
            </p>
          </div>
          <div className="p-3 bg-black/20 border border-white/5 rounded space-y-1">
            <strong className="text-zinc-200">3. Coordinate Normalization</strong>
            <p className="text-zinc-400 text-[11px]">
              Auto-detects integer scale [0-1000] vs normalized float [0.0-1.0] and handles coordinate inversion gracefully.
            </p>
          </div>
          <div className="p-3 bg-black/20 border border-white/5 rounded space-y-1">
            <strong className="text-zinc-200">4. Truncation Repair</strong>
            <p className="text-zinc-400 text-[11px]">
              Heuristically balances dangling brackets or extracts complete objects from truncated token streams.
            </p>
          </div>
        </div>
      </div>

      {/* Mathematical Formulations */}
      <div className="bg-[#12141c] border border-white/10 rounded-lg p-5 space-y-3">
        <div className="text-xs font-semibold uppercase tracking-wider text-zinc-400 flex items-center gap-2">
          <Calculator className="w-3.5 h-3.5 text-amber-400" /> Core Mathematical Parity
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
          <div className="p-3 bg-[#161822] border border-white/10 rounded space-y-1 font-mono">
            <div className="text-indigo-400 font-semibold font-sans">Intersection over Union (IoU)</div>
            <div className="text-zinc-300 text-[11px] bg-black/30 p-2 rounded">
              IoU = Area(B_p ∩ B_gt) / Area(B_p ∪ B_gt)
            </div>
            <p className="text-[11px] text-zinc-400 font-sans">
              Measures spatial overlap between prediction box B_p and ground truth B_gt.
            </p>
          </div>

          <div className="p-3 bg-[#161822] border border-white/10 rounded space-y-1 font-mono">
            <div className="text-emerald-400 font-semibold font-sans">COCO mAP@[.50:.05:.95]</div>
            <div className="text-zinc-300 text-[11px] bg-black/30 p-2 rounded">
              mAP = (1 / 10) * Σ AP(threshold_t)
            </div>
            <p className="text-[11px] text-zinc-400 font-sans">
              Mean over 10 IoU thresholds from 0.50 to 0.95 with 0.05 step size.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
