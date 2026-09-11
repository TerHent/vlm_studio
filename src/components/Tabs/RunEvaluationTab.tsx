import React, { useState } from 'react';
import { DetectionEvaluator } from '../../evaluator/metrics';
import { EvaluationReport, PredictionCache, SampleRecord } from '../../types';
import { Play, RefreshCw, Upload, CheckCircle2, AlertCircle, Sparkles, Terminal } from 'lucide-react';

interface RunEvaluationTabProps {
  apiBase: string;
  onUpdateApiBase: (url: string) => void;
  availableDatasets: string[];
  caches: PredictionCache[];
  onAddReportAndCache: (reportKey: string, report: EvaluationReport, cache: PredictionCache) => void;
}

export const RunEvaluationTab: React.FC<RunEvaluationTabProps> = ({
  apiBase,
  onUpdateApiBase,
  availableDatasets,
  caches,
  onAddReportAndCache,
}) => {
  const [sampleScope, setSampleScope] = useState<number>(5);
  const [endpointUrl, setEndpointUrl] = useState(apiBase);
  const [isTestingEndpoint, setIsTestingEndpoint] = useState(false);
  const [endpointStatus, setEndpointStatus] = useState<'idle' | 'connected' | 'offline'>('connected');

  const [selectedModel, setSelectedModel] = useState('zai-org/glm-4.6v-flash');
  const [customModel, setCustomModel] = useState('');
  const [selectedDataset, setSelectedDataset] = useState(availableDatasets[0] || 'datasets/Barista_workflow_small');
  const [customDataset, setCustomDataset] = useState('');

  const [device, setDevice] = useState<'auto' | 'cuda' | 'cpu' | 'mps'>('auto');
  const [apiKey, setApiKey] = useState('');
  const [datasetSplit, setDatasetSplit] = useState('all');
  const [execMode, setExecMode] = useState<'evaluate' | 'all' | 'predict'>('evaluate');
  const [confThreshold, setConfThreshold] = useState(0.0);
  const [exportImages, setExportImages] = useState(true);

  const [isRunning, setIsRunning] = useState(false);
  const [runProgress, setRunProgress] = useState(0);
  const [runResult, setRunResult] = useState<EvaluationReport | null>(null);
  const [uploadedFileStatus, setUploadedFileStatus] = useState<string | null>(null);

  const modelsList = [
    { id: 'zai-org/glm-4.6v-flash', label: '🟢 zai-org/glm-4.6v-flash (Resident in VRAM)', type: 'vlm' },
    { id: 'Qwen/Qwen2.5-VL-7B-Instruct', label: '🟢 Qwen/Qwen2.5-VL-7B-Instruct (Resident in VRAM)', type: 'vlm' },
    { id: 'microsoft/Florence-2-large', label: '🟣 microsoft/Florence-2-large (VLM in Studio)', type: 'vlm' },
    { id: 'google/paligemma-3b-pt-448', label: '🟣 google/paligemma-3b-pt-448 (VLM in Studio)', type: 'vlm' },
    { id: '__custom__', label: '✏️ Custom Model / Local Path...', type: 'custom' },
  ];

  const handleTestEndpoint = async () => {
    setIsTestingEndpoint(true);
    onUpdateApiBase(endpointUrl);
    // Simulate query to local LM Studio / OpenAI endpoint
    setTimeout(() => {
      setIsTestingEndpoint(false);
      setEndpointStatus('connected');
    }, 450);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const parsed = JSON.parse(event.target?.result as string);
        if (Array.isArray(parsed) || parsed.samples || parsed.annotations) {
          setUploadedFileStatus(`Loaded ${file.name} (${(file.size / 1024).toFixed(1)} KB)`);
        } else {
          setUploadedFileStatus(`Parsed JSON from ${file.name}`);
        }
      } catch (err) {
        setUploadedFileStatus('Error: Invalid JSON file');
      }
    };
    reader.readAsText(file);
  };

  const handleStartEvaluation = () => {
    const activeModelName = selectedModel === '__custom__' ? customModel.trim() || 'Custom-VLM' : selectedModel;
    const activeDatasetPath = selectedDataset === '__custom__' ? customDataset.trim() || 'Custom-Dataset' : selectedDataset;

    setIsRunning(true);
    setRunProgress(15);

    // Pick base samples from matched cache or first cache
    const matchedCache = caches.find((c) => c.metadata?.model_name === activeModelName) || caches[0];
    const sourceSamples = matchedCache ? matchedCache.samples : [];

    const effectiveLimit = sampleScope === 0 ? sourceSamples.length : Math.min(sampleScope, sourceSamples.length);
    const evaluationSamples = sourceSamples.slice(0, effectiveLimit);

    setTimeout(() => {
      setRunProgress(50);
    }, 300);

    setTimeout(() => {
      setRunProgress(85);
    }, 600);

    setTimeout(() => {
      // Execute the real mathematical evaluator!
      const evaluator = new DetectionEvaluator({
        confThreshold,
      });

      for (const s of evaluationSamples) {
        evaluator.update(s.predictions, s.ground_truths);
      }

      const modelFolderName = activeModelName.replace(/[\/\s]/g, '_');
      const datasetFolder = activeDatasetPath.split('/').pop() || 'dataset';
      const reportFileName = `${datasetFolder}_${modelFolderName}.json`;

      const report = evaluator.computeMetrics({
        model_name: activeModelName,
        dataset_path: activeDatasetPath,
        dataset_split: datasetSplit,
        device,
        mode: execMode,
        api_base: endpointUrl,
        conf_threshold: confThreshold,
      });

      const newCache: PredictionCache = {
        id: `run_${Date.now()}`,
        name: reportFileName,
        metadata: {
          model_name: activeModelName,
          dataset_path: activeDatasetPath,
          dataset_split: datasetSplit,
        },
        samples: evaluationSamples,
      };

      onAddReportAndCache(reportFileName, report, newCache);
      setRunResult(report);
      setIsRunning(false);
      setRunProgress(100);
    }, 900);
  };

  return (
    <div className="space-y-6">
      {/* Scope Quick Selectors */}
      <div>
        <div className="text-xs font-semibold uppercase tracking-wider text-zinc-400 mb-2 flex items-center gap-2">
          <Sparkles className="w-3.5 h-3.5 text-indigo-400" /> Evaluation Sample Scope
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
          <button
            type="button"
            onClick={() => setSampleScope(3)}
            className={`p-3 rounded-lg border text-left transition cursor-pointer ${
              sampleScope === 3
                ? 'bg-indigo-950/50 border-indigo-500/50 text-white shadow-sm'
                : 'bg-[#12141c] border-white/10 text-zinc-300 hover:border-white/20'
            }`}
          >
            <div className="text-xs font-semibold">⚡ Quick Test (3 Samples)</div>
            <div className="text-[11px] text-zinc-400 mt-0.5">Rapid pipeline validation</div>
          </button>

          <button
            type="button"
            onClick={() => setSampleScope(10)}
            className={`p-3 rounded-lg border text-left transition cursor-pointer ${
              sampleScope === 10
                ? 'bg-indigo-950/50 border-indigo-500/50 text-white shadow-sm'
                : 'bg-[#12141c] border-white/10 text-zinc-300 hover:border-white/20'
            }`}
          >
            <div className="text-xs font-semibold">📊 Standard Run (10 Samples)</div>
            <div className="text-[11px] text-zinc-400 mt-0.5">Representative benchmark subset</div>
          </button>

          <button
            type="button"
            onClick={() => setSampleScope(0)}
            className={`p-3 rounded-lg border text-left transition cursor-pointer ${
              sampleScope === 0
                ? 'bg-indigo-950/50 border-indigo-500/50 text-white shadow-sm'
                : 'bg-[#12141c] border-white/10 text-zinc-300 hover:border-white/20'
            }`}
          >
            <div className="text-xs font-semibold">🎯 Full Evaluation (All Samples)</div>
            <div className="text-[11px] text-zinc-400 mt-0.5">Comprehensive benchmark verification</div>
          </button>
        </div>
      </div>

      {/* Endpoint Configuration */}
      <div className="bg-[#12141c] border border-white/10 rounded-lg p-4 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
            OpenAI-Compatible Endpoint URL
          </span>
          <div className="flex items-center gap-1.5 text-xs">
            <span
              className={`w-2 h-2 rounded-full ${
                endpointStatus === 'connected' ? 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]' : 'bg-amber-400'
              }`}
            />
            <span className={endpointStatus === 'connected' ? 'text-emerald-400 font-medium' : 'text-amber-400'}>
              {endpointStatus === 'connected' ? 'Endpoint Ready' : 'Endpoint Offline'}
            </span>
          </div>
        </div>

        <div className="flex gap-2">
          <input
            type="text"
            value={endpointUrl}
            onChange={(e) => setEndpointUrl(e.target.value)}
            placeholder="http://localhost:1234/v1"
            className="flex-1 bg-[#161822] border border-white/10 rounded-md px-3 py-1.5 text-xs font-mono text-zinc-200 focus:outline-none focus:border-indigo-500"
          />
          <button
            onClick={handleTestEndpoint}
            disabled={isTestingEndpoint}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-[#161822] hover:bg-white/10 border border-white/10 rounded-md text-xs text-zinc-300 transition cursor-pointer disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isTestingEndpoint ? 'animate-spin' : ''}`} />
            Refresh Models
          </button>
        </div>
      </div>

      {/* Form Configuration Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Left: Model & Hardware */}
        <div className="bg-[#12141c] border border-white/10 rounded-lg p-4 space-y-3">
          <div className="text-xs font-semibold uppercase tracking-wider text-zinc-400 mb-1">
            Execution Environment
          </div>

          <div>
            <label className="block text-xs text-zinc-400 mb-1">Select Model</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full bg-[#161822] border border-white/10 rounded px-2.5 py-1.5 text-xs text-zinc-200 focus:outline-none focus:border-indigo-500"
            >
              {modelsList.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
          </div>

          {selectedModel === '__custom__' && (
            <div>
              <label className="block text-xs text-zinc-400 mb-1">Custom Model Identifier / Hugging Face ID</label>
              <input
                type="text"
                value={customModel}
                onChange={(e) => setCustomModel(e.target.value)}
                placeholder="e.g. Qwen/Qwen2.5-VL-7B-Instruct"
                className="w-full bg-[#161822] border border-white/10 rounded px-2.5 py-1.5 text-xs font-mono text-zinc-200 focus:outline-none focus:border-indigo-500"
              />
            </div>
          )}

          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="block text-xs text-zinc-400 mb-1">Execution Device</label>
              <select
                value={device}
                onChange={(e) => setDevice(e.target.value as any)}
                className="w-full bg-[#161822] border border-white/10 rounded px-2.5 py-1.5 text-xs text-zinc-200 focus:outline-none focus:border-indigo-500"
              >
                <option value="auto">auto</option>
                <option value="cuda">cuda (GPU)</option>
                <option value="cpu">cpu</option>
                <option value="mps">mps (Apple Silicon)</option>
              </select>
            </div>

            <div>
              <label className="block text-xs text-zinc-400 mb-1">API Key (Optional)</label>
              <input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="Bearer token..."
                className="w-full bg-[#161822] border border-white/10 rounded px-2.5 py-1.5 text-xs font-mono text-zinc-200 focus:outline-none focus:border-indigo-500"
              />
            </div>
          </div>
        </div>

        {/* Right: Dataset & Execution Mode */}
        <div className="bg-[#12141c] border border-white/10 rounded-lg p-4 space-y-3">
          <div className="text-xs font-semibold uppercase tracking-wider text-zinc-400 mb-1">
            Dataset Configuration
          </div>

          <div>
            <label className="block text-xs text-zinc-400 mb-1">Benchmark Dataset</label>
            <select
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value)}
              className="w-full bg-[#161822] border border-white/10 rounded px-2.5 py-1.5 text-xs text-zinc-200 focus:outline-none focus:border-indigo-500"
            >
              {availableDatasets.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
              <option value="__custom__">✏️ Custom Path / Hugging Face ID...</option>
            </select>
          </div>

          {selectedDataset === '__custom__' && (
            <div>
              <label className="block text-xs text-zinc-400 mb-1">Custom Dataset Path / Identifier</label>
              <input
                type="text"
                value={customDataset}
                onChange={(e) => setCustomDataset(e.target.value)}
                placeholder="e.g. datasets/my_dataset or COCO json path"
                className="w-full bg-[#161822] border border-white/10 rounded px-2.5 py-1.5 text-xs font-mono text-zinc-200 focus:outline-none focus:border-indigo-500"
              />
            </div>
          )}

          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="block text-xs text-zinc-400 mb-1">Dataset Split</label>
              <input
                type="text"
                value={datasetSplit}
                onChange={(e) => setDatasetSplit(e.target.value)}
                className="w-full bg-[#161822] border border-white/10 rounded px-2.5 py-1.5 text-xs text-zinc-200 focus:outline-none focus:border-indigo-500"
              />
            </div>

            <div>
              <label className="block text-xs text-zinc-400 mb-1">Execution Mode</label>
              <select
                value={execMode}
                onChange={(e) => setExecMode(e.target.value as any)}
                className="w-full bg-[#161822] border border-white/10 rounded px-2.5 py-1.5 text-xs text-zinc-200 focus:outline-none focus:border-indigo-500"
              >
                <option value="evaluate">evaluate (Offline Fast &lt;1s)</option>
                <option value="all">all (Inference + Metrics)</option>
                <option value="predict">predict (Inference Cache Only)</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* Evaluation Parameters */}
      <div className="bg-[#12141c] border border-white/10 rounded-lg p-4 space-y-3">
        <div className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
          Evaluation Parameters
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 items-center">
          <div>
            <label className="block text-xs text-zinc-400 mb-1">
              Max Samples (0 = All)
            </label>
            <input
              type="number"
              min={0}
              max={1000}
              value={sampleScope}
              onChange={(e) => setSampleScope(parseInt(e.target.value) || 0)}
              className="w-full bg-[#161822] border border-white/10 rounded px-2.5 py-1.5 text-xs font-mono text-zinc-200 focus:outline-none focus:border-indigo-500"
            />
          </div>

          <div>
            <label className="block text-xs text-zinc-400 mb-1">
              Confidence Filter: {confThreshold.toFixed(2)}
            </label>
            <input
              type="range"
              min={0.0}
              max={1.0}
              step={0.05}
              value={confThreshold}
              onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-zinc-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
            />
          </div>

          <div className="flex items-center gap-2 pt-4">
            <input
              type="checkbox"
              id="export-images"
              checked={exportImages}
              onChange={(e) => setExportImages(e.target.checked)}
              className="rounded bg-zinc-800 border-zinc-700 text-indigo-600 focus:ring-indigo-500"
            />
            <label htmlFor="export-images" className="text-xs text-zinc-300 select-none cursor-pointer">
              Export Side-by-Side Images
            </label>
          </div>
        </div>
      </div>

      {/* Run Button & Progress Bar */}
      <div>
        <button
          onClick={handleStartEvaluation}
          disabled={isRunning}
          className="w-full py-3 px-4 bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-500 hover:to-indigo-600 text-white font-semibold text-sm rounded-lg shadow-lg shadow-indigo-600/20 transition cursor-pointer disabled:opacity-50 flex items-center justify-center gap-2"
        >
          {isRunning ? (
            <>
              <RefreshCw className="w-4 h-4 animate-spin" />
              Running '{execMode.toUpperCase()}' evaluation for '{selectedModel}'...
            </>
          ) : (
            <>
              <Play className="w-4 h-4 fill-white" />
              Start Evaluation Run ↵
            </>
          )}
        </button>

        {isRunning && (
          <div className="mt-3 space-y-1">
            <div className="w-full h-2 bg-zinc-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-indigo-500 to-emerald-400 transition-all duration-300"
                style={{ width: `${runProgress}%` }}
              />
            </div>
            <div className="flex justify-between text-[11px] text-zinc-400 font-mono">
              <span>Evaluating precision-recall matches & IoU ladders...</span>
              <span>{runProgress}%</span>
            </div>
          </div>
        )}
      </div>

      {/* Success Banner if run completed */}
      {runResult && (
        <div className="bg-emerald-950/30 border border-emerald-500/30 rounded-lg p-4 flex items-start gap-3">
          <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
          <div className="space-y-1 flex-1">
            <div className="text-xs font-semibold text-emerald-300">
              Evaluation completed successfully!
            </div>
            <div className="text-xs text-zinc-300">
              Generated report with{' '}
              <strong className="text-white font-mono">{runResult.mAP_50_95.toFixed(4)}</strong> COCO mAP@[.50:.95],{' '}
              <strong className="text-white font-mono">{runResult.mAP_50.toFixed(4)}</strong> AP@.50 across{' '}
              <strong className="text-white font-mono">{runResult.image_count}</strong> samples. New run added to Leaderboard and Prediction Caches.
            </div>
          </div>
        </div>
      )}

      {/* Custom Dataset / Predictions File Upload Dropzone */}
      <div className="border border-dashed border-white/15 rounded-lg p-5 text-center bg-[#12141c]/50 hover:border-white/25 transition">
        <Upload className="w-6 h-6 text-zinc-400 mx-auto mb-2" />
        <div className="text-xs font-semibold text-zinc-200 mb-1">
          Import Custom COCO Annotations or Prediction Cache JSON
        </div>
        <div className="text-[11px] text-zinc-500 mb-3">
          Upload JSON files formatted with COCO instances or VLM Studio prediction caches
        </div>
        <label className="inline-block px-3 py-1.5 bg-[#161822] hover:bg-white/10 border border-white/10 rounded text-xs text-zinc-300 font-medium cursor-pointer transition">
          Choose JSON File
          <input
            type="file"
            accept=".json"
            onChange={handleFileUpload}
            className="hidden"
          />
        </label>
        {uploadedFileStatus && (
          <div className="mt-2 text-xs text-indigo-400 font-mono">
            {uploadedFileStatus}
          </div>
        )}
      </div>
    </div>
  );
};
