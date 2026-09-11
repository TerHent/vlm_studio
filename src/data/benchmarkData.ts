import { EvaluationReport, PredictionCache, SampleRecord } from '../types';
import { DetectionEvaluator } from '../evaluator/metrics';

// SVG Visual Scenes for authentic side-by-side rendering
const SCENE_BARISTA_STATION = `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="100%" height="100%">
  <defs>
    <linearGradient id="bgGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#181a24"/>
      <stop offset="60%" stop-color="#222533"/>
      <stop offset="100%" stop-color="#14151e"/>
    </linearGradient>
    <linearGradient id="counterGrad" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#2a2725"/>
      <stop offset="50%" stop-color="#3c3734"/>
      <stop offset="100%" stop-color="#282523"/>
    </linearGradient>
    <linearGradient id="steelGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#94a3b8"/>
      <stop offset="50%" stop-color="#64748b"/>
      <stop offset="100%" stop-color="#475569"/>
    </linearGradient>
    <linearGradient id="brassGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#d97706"/>
      <stop offset="100%" stop-color="#92400e"/>
    </linearGradient>
  </defs>
  <!-- Background Wall -->
  <rect width="800" height="600" fill="url(#bgGrad)"/>
  <!-- Tiles Pattern Line -->
  <line x1="0" y1="120" x2="800" y2="120" stroke="rgba(255,255,255,0.05)" stroke-width="1"/>
  <line x1="0" y1="240" x2="800" y2="240" stroke="rgba(255,255,255,0.05)" stroke-width="1"/>
  
  <!-- Wooden Counter -->
  <rect x="0" y="380" width="800" height="220" fill="url(#counterGrad)"/>
  <line x1="0" y1="380" x2="800" y2="380" stroke="#524a46" stroke-width="3"/>
  <line x1="0" y1="395" x2="800" y2="395" stroke="rgba(255,255,255,0.08)" stroke-width="1"/>

  <!-- Espresso Machine (ymin=0.15, xmin=0.08, ymax=0.68, xmax=0.55) -->
  <!-- 800 * 0.08 = 64, 600 * 0.15 = 90, width=376, height=318 -->
  <rect x="64" y="90" width="376" height="300" rx="10" fill="url(#steelGrad)" stroke="#334155" stroke-width="3"/>
  <!-- Machine Details -->
  <rect x="80" y="110" width="344" height="60" rx="4" fill="#0f172a"/>
  <!-- Pressure Gauges -->
  <circle cx="120" cy="140" r="18" fill="#1e293b" stroke="#cbd5e1" stroke-width="2"/>
  <line x1="120" y1="140" x2="130" y2="132" stroke="#ef4444" stroke-width="2"/>
  <circle cx="170" cy="140" r="18" fill="#1e293b" stroke="#cbd5e1" stroke-width="2"/>
  <line x1="170" y1="140" x2="182" y2="136" stroke="#10b981" stroke-width="2"/>
  <!-- Digital Display -->
  <rect x="230" y="125" width="80" height="30" rx="4" fill="#020617" stroke="#334155"/>
  <text x="270" y="146" fill="#38bdf8" font-family="monospace" font-size="14" text-anchor="middle" font-weight="bold">93.4°C</text>
  <!-- Group Heads -->
  <rect x="130" y="210" width="50" height="40" rx="4" fill="#1e293b"/>
  <rect x="270" y="210" width="50" height="40" rx="4" fill="#1e293b"/>
  <!-- Portafilters -->
  <path d="M 140 240 L 170 240 L 165 290 L 145 290 Z" fill="url(#brassGrad)"/>
  <rect x="150" y="290" width="10" height="50" rx="3" fill="#0f172a"/>
  <!-- Drip Tray -->
  <rect x="74" y="360" width="356" height="25" fill="#334155" rx="2"/>

  <!-- Coffee Grinder (ymin=0.12, xmin=0.58, ymax=0.65, xmax=0.76) -->
  <rect x="464" y="72" width="144" height="318" rx="8" fill="#1e293b" stroke="#0f172a" stroke-width="2"/>
  <!-- Hopper -->
  <polygon points="484,80 588,80 568,180 504,180" fill="rgba(245, 158, 11, 0.4)" stroke="#78350f" stroke-width="2"/>
  <rect x="500" y="180" width="72" height="40" fill="#334155"/>
  <circle cx="536" cy="270" r="16" fill="#0f172a" stroke="#64748b"/>

  <!-- Milk Pitcher (ymin=0.50, xmin=0.38, ymax=0.76, xmax=0.52) -->
  <!-- 800 * 0.38 = 304, 600 * 0.50 = 300, width = 112, height = 156 -->
  <path d="M 320 310 L 400 310 L 412 440 L 312 440 Z" fill="url(#steelGrad)" stroke="#1e293b" stroke-width="2"/>
  <path d="M 400 330 C 430 330, 430 420, 395 420" fill="none" stroke="#64748b" stroke-width="8" stroke-linecap="round"/>

  <!-- Digital Scale (ymin=0.68, xmin=0.18, ymax=0.82, xmax=0.36) -->
  <rect x="144" y="408" width="144" height="35" rx="6" fill="#020617" stroke="#334155" stroke-width="2"/>
  <rect x="180" y="415" width="70" height="20" rx="3" fill="#1e293b"/>
  <text x="215" y="430" fill="#34d399" font-family="monospace" font-size="12" text-anchor="middle" font-weight="bold">18.2g</text>

  <!-- Ceramic Coffee Cup (ymin=0.58, xmin=0.20, ymax=0.73, xmax=0.32) -->
  <ellipse cx="208" cy="400" rx="48" ry="18" fill="#f8fafc" stroke="#94a3b8" stroke-width="2"/>
  <path d="M 160 400 C 160 435, 256 435, 256 400 Z" fill="#f8fafc" stroke="#94a3b8" stroke-width="2"/>
  <ellipse cx="208" cy="400" rx="42" ry="12" fill="#451a03"/>
  <!-- Latte Art Microfoam -->
  <ellipse cx="208" cy="400" rx="20" ry="6" fill="#fef3c7"/>
</svg>
`;

const SCENE_ROBOT_TABLETOP = `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="100%" height="100%">
  <defs>
    <linearGradient id="labBg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#0f172a"/>
      <stop offset="100%" stop-color="#1e293b"/>
    </linearGradient>
    <radialGradient id="matGrad" cx="50%" cy="50%" r="60%">
      <stop offset="0%" stop-color="#1e3a5f"/>
      <stop offset="100%" stop-color="#0f172a"/>
    </radialGradient>
  </defs>
  <rect width="800" height="600" fill="url(#labBg)"/>
  <!-- Anti-static ESD Green/Blue Work Mat -->
  <rect x="60" y="80" width="680" height="460" rx="12" fill="url(#matGrad)" stroke="#38bdf8" stroke-width="2" stroke-opacity="0.4"/>
  <!-- Grid Lines on Mat -->
  <g stroke="rgba(56, 189, 248, 0.15)" stroke-width="1">
    <line x1="140" y1="80" x2="140" y2="540"/>
    <line x1="220" y1="80" x2="220" y2="540"/>
    <line x1="300" y1="80" x2="300" y2="540"/>
    <line x1="380" y1="80" x2="380" y2="540"/>
    <line x1="460" y1="80" x2="460" y2="540"/>
    <line x1="540" y1="80" x2="540" y2="540"/>
    <line x1="620" y1="80" x2="620" y2="540"/>
    <line x1="60" y1="160" x2="740" y2="160"/>
    <line x1="60" y1="240" x2="740" y2="240"/>
    <line x1="60" y1="320" x2="740" y2="320"/>
    <line x1="60" y1="400" x2="740" y2="400"/>
    <line x1="60" y1="480" x2="740" y2="480"/>
  </g>

  <!-- Robotic Gripper Arm End-Effector (ymin=0.08, xmin=0.35, ymax=0.38, xmax=0.62) -->
  <rect x="360" y="40" width="80" height="100" rx="6" fill="#334155" stroke="#64748b" stroke-width="2"/>
  <rect x="330" y="140" width="30" height="80" rx="4" fill="#64748b"/>
  <rect x="440" y="140" width="30" height="80" rx="4" fill="#64748b"/>
  <circle cx="400" cy="90" r="10" fill="#10b981"/>

  <!-- Red Target Cube (ymin=0.45, xmin=0.20, ymax=0.65, xmax=0.35) -->
  <polygon points="170,300 240,260 280,285 210,325" fill="#ef4444"/>
  <polygon points="170,300 210,325 210,390 170,365" fill="#b91c1c"/>
  <polygon points="210,325 280,285 280,350 210,390" fill="#dc2626"/>

  <!-- Blue Cylindrical Peg (ymin=0.38, xmin=0.55, ymax=0.60, xmax=0.68) -->
  <ellipse cx="492" cy="250" rx="48" ry="18" fill="#38bdf8"/>
  <path d="M 444 250 L 444 330 A 48 18 0 0 0 540 330 L 540 250 Z" fill="#0284c7" stroke="#38bdf8"/>

  <!-- Yellow Plastic Gear / Cog (ymin=0.60, xmin=0.38, ymax=0.82, xmax=0.54) -->
  <circle cx="368" cy="426" r="48" fill="#eab308" stroke="#ca8a04" stroke-width="6"/>
  <circle cx="368" cy="426" r="20" fill="#1e3a5f"/>

  <!-- Electronic Screwdriver Tool (ymin=0.25, xmin=0.72, ymax=0.70, xmax=0.85) -->
  <rect x="600" y="160" width="35" height="150" rx="10" fill="#f97316"/>
  <rect x="612" y="310" width="11" height="90" fill="#94a3b8"/>
</svg>
`;

const SCENE_COCO_STREET = `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="100%" height="100%">
  <defs>
    <linearGradient id="skyGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#0f172a"/>
      <stop offset="60%" stop-color="#1e293b"/>
      <stop offset="100%" stop-color="#334155"/>
    </linearGradient>
    <linearGradient id="asphaltGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#1e293b"/>
      <stop offset="100%" stop-color="#0f172a"/>
    </linearGradient>
  </defs>
  <!-- Sky -->
  <rect width="800" height="320" fill="url(#skyGrad)"/>
  <!-- Buildings Silhouette -->
  <rect x="40" y="120" width="110" height="200" fill="#1e1e2d"/>
  <rect x="170" y="80" width="140" height="240" fill="#171824"/>
  <rect x="330" y="140" width="160" height="180" fill="#1a1c29"/>
  <rect x="520" y="100" width="190" height="220" fill="#131420"/>

  <!-- Road -->
  <polygon points="0,320 800,320 800,600 0,600" fill="url(#asphaltGrad)"/>
  <line x1="400" y1="320" x2="400" y2="600" stroke="#facc15" stroke-dasharray="25,25" stroke-width="4"/>

  <!-- Car Left (ymin=0.48, xmin=0.08, ymax=0.78, xmax=0.42) -->
  <path d="M 80 430 L 140 370 L 260 370 L 320 430 L 320 460 L 80 460 Z" fill="#3b82f6"/>
  <rect x="145" y="380" width="45" height="40" fill="#93c5fd" rx="3"/>
  <rect x="200" y="380" width="55" height="40" fill="#93c5fd" rx="3"/>
  <circle cx="130" cy="460" r="22" fill="#0f172a" stroke="#64748b" stroke-width="4"/>
  <circle cx="270" cy="460" r="22" fill="#0f172a" stroke="#64748b" stroke-width="4"/>

  <!-- Pedestrian (ymin=0.40, xmin=0.48, ymax=0.82, xmax=0.58) -->
  <!-- 800 * 0.48 = 384, 600 * 0.40 = 240, width=80, height=252 -->
  <circle cx="424" cy="260" r="16" fill="#fde047"/>
  <rect x="410" y="280" width="28" height="70" rx="6" fill="#10b981"/>
  <line x1="416" y1="350" x2="414" y2="480" stroke="#1e293b" stroke-width="8"/>
  <line x1="432" y1="350" x2="434" y2="480" stroke="#1e293b" stroke-width="8"/>

  <!-- Bicycle / Rider (ymin=0.46, xmin=0.62, ymax=0.84, xmax=0.88) -->
  <circle cx="536" cy="460" r="24" fill="none" stroke="#e2e8f0" stroke-width="4"/>
  <circle cx="660" cy="460" r="24" fill="none" stroke="#e2e8f0" stroke-width="4"/>
  <line x1="536" y1="460" x2="598" y2="460" stroke="#ef4444" stroke-width="5"/>
  <line x1="598" y1="460" x2="570" y2="380" stroke="#ef4444" stroke-width="5"/>
  <line x1="570" y1="380" x2="536" y2="460" stroke="#ef4444" stroke-width="5"/>
  <line x1="598" y1="460" x2="636" y2="400" stroke="#ef4444" stroke-width="5"/>
  <line x1="636" y1="400" x2="660" y2="460" stroke="#ef4444" stroke-width="5"/>
  <!-- Cyclist Torso -->
  <circle cx="570" cy="330" r="14" fill="#fde047"/>
  <rect x="560" y="348" width="22" height="40" rx="4" fill="#8b5cf6"/>
</svg>
`;

export const PRESET_PREDICTION_CACHES: PredictionCache[] = [
  {
    id: 'barista_glm4v',
    name: 'Barista_workflow_small_glm-4.6v-flash.json',
    metadata: {
      model_name: 'zai-org/glm-4.6v-flash',
      dataset_path: 'datasets/Barista_workflow_small',
      dataset_split: 'all',
      api_base: 'http://localhost:1234/v1',
      device: 'cuda',
      timestamp: '2026-09-10T14:32:00Z',
    },
    samples: [
      {
        file_name: 'barista_station_frame_001.jpg',
        image_svg: SCENE_BARISTA_STATION,
        width: 800,
        height: 600,
        ground_truths: [
          { bbox: [0.15, 0.08, 0.68, 0.55], label: 'Espresso Machine' },
          { bbox: [0.12, 0.58, 0.65, 0.76], label: 'Coffee Grinder' },
          { bbox: [0.50, 0.38, 0.76, 0.52], label: 'Milk Pitcher' },
          { bbox: [0.68, 0.18, 0.82, 0.36], label: 'Scale' },
          { bbox: [0.58, 0.20, 0.73, 0.32], label: 'Coffee Mug' },
        ],
        predictions: [
          { bbox: [0.152, 0.083, 0.675, 0.548], label: 'Espresso Machine', score: 0.96 },
          { bbox: [0.118, 0.575, 0.648, 0.762], label: 'Coffee Grinder', score: 0.94 },
          { bbox: [0.505, 0.382, 0.755, 0.518], label: 'Milk Pitcher', score: 0.91 },
          { bbox: [0.675, 0.185, 0.815, 0.355], label: 'Scale', score: 0.88 },
          { bbox: [0.578, 0.202, 0.725, 0.318], label: 'Coffee Mug', score: 0.85 },
          { bbox: [0.720, 0.450, 0.850, 0.600], label: 'Coffee Mug', score: 0.24 }, // FP low score
        ],
      },
      {
        file_name: 'barista_station_frame_002.jpg',
        image_svg: SCENE_BARISTA_STATION,
        width: 800,
        height: 600,
        ground_truths: [
          { bbox: [0.15, 0.08, 0.68, 0.55], label: 'Espresso Machine' },
          { bbox: [0.50, 0.38, 0.76, 0.52], label: 'Milk Pitcher' },
          { bbox: [0.58, 0.20, 0.73, 0.32], label: 'Coffee Mug' },
        ],
        predictions: [
          { bbox: [0.148, 0.085, 0.682, 0.552], label: 'Espresso Machine', score: 0.95 },
          { bbox: [0.498, 0.385, 0.762, 0.525], label: 'Milk Pitcher', score: 0.92 },
          { bbox: [0.582, 0.198, 0.734, 0.325], label: 'Coffee Mug', score: 0.89 },
        ],
      },
      {
        file_name: 'barista_station_frame_003.jpg',
        image_svg: SCENE_BARISTA_STATION,
        width: 800,
        height: 600,
        ground_truths: [
          { bbox: [0.12, 0.58, 0.65, 0.76], label: 'Coffee Grinder' },
          { bbox: [0.68, 0.18, 0.82, 0.36], label: 'Scale' },
        ],
        predictions: [
          { bbox: [0.122, 0.583, 0.654, 0.758], label: 'Coffee Grinder', score: 0.93 },
          { bbox: [0.685, 0.178, 0.818, 0.362], label: 'Scale', score: 0.86 },
        ],
      },
    ],
  },
  {
    id: 'robot_tabletop_florence',
    name: 'Robot_tabletop_florence-2-large.json',
    metadata: {
      model_name: 'microsoft/Florence-2-large',
      dataset_path: 'datasets/Robot_tabletop_manipulation',
      dataset_split: 'test',
      device: 'cuda',
      timestamp: '2026-09-09T18:15:00Z',
    },
    samples: [
      {
        file_name: 'robot_manipulation_sample_01.jpg',
        image_svg: SCENE_ROBOT_TABLETOP,
        width: 800,
        height: 600,
        ground_truths: [
          { bbox: [0.08, 0.35, 0.38, 0.62], label: 'Robotic Gripper' },
          { bbox: [0.45, 0.20, 0.65, 0.35], label: 'Target Cube' },
          { bbox: [0.38, 0.55, 0.60, 0.68], label: 'Cylindrical Peg' },
          { bbox: [0.60, 0.38, 0.82, 0.54], label: 'Gear Cog' },
        ],
        predictions: [
          { bbox: [0.078, 0.345, 0.385, 0.625], label: 'Robotic Gripper', score: 0.94 },
          { bbox: [0.448, 0.198, 0.652, 0.352], label: 'Target Cube', score: 0.91 },
          { bbox: [0.375, 0.545, 0.605, 0.685], label: 'Cylindrical Peg', score: 0.87 },
          { bbox: [0.595, 0.375, 0.825, 0.545], label: 'Gear Cog', score: 0.82 },
          { bbox: [0.250, 0.720, 0.700, 0.850], label: 'Target Cube', score: 0.35 }, // False Positive
        ],
      },
      {
        file_name: 'robot_manipulation_sample_02.jpg',
        image_svg: SCENE_ROBOT_TABLETOP,
        width: 800,
        height: 600,
        ground_truths: [
          { bbox: [0.45, 0.20, 0.65, 0.35], label: 'Target Cube' },
          { bbox: [0.60, 0.38, 0.82, 0.54], label: 'Gear Cog' },
        ],
        predictions: [
          { bbox: [0.452, 0.205, 0.648, 0.348], label: 'Target Cube', score: 0.93 },
          { bbox: [0.605, 0.382, 0.818, 0.538], label: 'Gear Cog', score: 0.85 },
        ],
      },
    ],
  },
  {
    id: 'coco_val_qwen',
    name: 'COCO_val2017_qwen2.5-vl-7b-instruct.json',
    metadata: {
      model_name: 'Qwen/Qwen2.5-VL-7B-Instruct',
      dataset_path: 'datasets/COCO_val2017',
      dataset_split: 'val',
      device: 'cuda',
      timestamp: '2026-09-08T11:20:00Z',
    },
    samples: [
      {
        file_name: '000000039769.jpg',
        image_svg: SCENE_COCO_STREET,
        width: 800,
        height: 600,
        ground_truths: [
          { bbox: [0.48, 0.08, 0.78, 0.42], label: 'car' },
          { bbox: [0.40, 0.48, 0.82, 0.58], label: 'person' },
          { bbox: [0.46, 0.62, 0.84, 0.88], label: 'bicycle' },
        ],
        predictions: [
          { bbox: [0.482, 0.085, 0.778, 0.418], label: 'car', score: 0.95 },
          { bbox: [0.395, 0.478, 0.825, 0.582], label: 'person', score: 0.89 },
          { bbox: [0.458, 0.615, 0.845, 0.875], label: 'bicycle', score: 0.84 },
        ],
      },
    ],
  },
];

// Helper to compute realistic reports directly from the samples
export function generateReportFromCache(cache: PredictionCache): EvaluationReport {
  const evaluator = new DetectionEvaluator();
  for (const s of cache.samples) {
    evaluator.update(s.predictions, s.ground_truths);
  }
  return evaluator.computeMetrics(cache.metadata);
}

export const INITIAL_REPORTS: Record<string, EvaluationReport> = {
  'Barista_workflow_small_glm-4.6v-flash.json': generateReportFromCache(PRESET_PREDICTION_CACHES[0]),
  'Robot_tabletop_florence-2-large.json': generateReportFromCache(PRESET_PREDICTION_CACHES[1]),
  'COCO_val2017_qwen2.5-vl-7b-instruct.json': generateReportFromCache(PRESET_PREDICTION_CACHES[2]),
};

// Stable Distinct Colors for object detection categories
const CATEGORY_COLORS = [
  '#6366f1', // Indigo
  '#10b981', // Emerald
  '#f59e0b', // Amber
  '#ec4899', // Pink
  '#06b6d4', // Cyan
  '#8b5cf6', // Violet
  '#f97316', // Orange
  '#14b8a6', // Teal
  '#3b82f6', // Blue
  '#ef4444', // Red
];

const categoryColorMap = new Map<string, string>();

export function getCategoryColor(category: string): string {
  if (!categoryColorMap.has(category)) {
    const color = CATEGORY_COLORS[categoryColorMap.size % CATEGORY_COLORS.length];
    categoryColorMap.set(category, color);
  }
  return categoryColorMap.get(category)!;
}
