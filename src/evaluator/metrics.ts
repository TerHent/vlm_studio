import {
  BoundingBox,
  GroundTruthBox,
  PredictionBox,
  EvaluationReport,
  IoUReport,
  ClassMetric,
  ClassSummary,
} from '../types';

export const COCO_IOU_THRESHOLDS: number[] = [
  0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
];

export function computeIoU(box1: BoundingBox, box2: BoundingBox): number {
  const [ymin1, xmin1, ymax1, xmax1] = box1;
  const [ymin2, xmin2, ymax2, xmax2] = box2;

  const yMin = Math.max(ymin1, ymin2);
  const xMin = Math.max(xmin1, xmin2);
  const yMax = Math.min(ymax1, ymax2);
  const xMax = Math.min(xmax1, xmax2);

  if (yMax <= yMin || xMax <= xMin) {
    return 0.0;
  }

  const intersection = (yMax - yMin) * (xMax - xMin);
  const area1 = (ymax1 - ymin1) * (xmax1 - xmin1);
  const area2 = (ymax2 - ymin2) * (xmax2 - xmin2);

  const union = area1 + area2 - intersection;
  if (union <= 0.0) {
    return 0.0;
  }

  return intersection / union;
}

export function parseIoUThresholds(
  arg?: string | number[] | null
): number[] {
  if (!arg || (typeof arg === 'string' && ['coco', 'default', 'standard'].includes(arg.trim().toLowerCase()))) {
    return [...COCO_IOU_THRESHOLDS];
  }

  if (Array.isArray(arg)) {
    return Array.from(new Set(arg.map((x) => Math.round(x * 10000) / 10000))).sort((a, b) => a - b);
  }

  const clean = arg.trim().toLowerCase();
  if (clean.includes(':')) {
    const parts = clean.split(':').map((p) => parseFloat(p.trim()));
    let start = parts[0];
    let end = parts[1];
    let step = parts.length === 3 ? parts[2] : 0.05;

    if (step <= 0) step = 0.05;
    const res: number[] = [];
    for (let v = start; v <= end + step / 2.0; v += step) {
      res.push(Math.round(v * 10000) / 10000);
    }
    return Array.from(new Set(res)).sort((a, b) => a - b);
  }

  const values = clean
    .split(',')
    .map((s) => parseFloat(s.trim()))
    .filter((n) => !isNaN(n) && n >= 0 && n <= 1);

  return values.length > 0 ? Array.from(new Set(values)).sort((a, b) => a - b) : [...COCO_IOU_THRESHOLDS];
}

interface StoredPred {
  imageId: number;
  bbox: BoundingBox;
  label: string;
  score: number;
  origLabel: string;
}

interface StoredGT {
  imageId: number;
  bbox: BoundingBox;
  label: string;
}

export class DetectionEvaluator {
  private labelMap: Record<string, string>;
  private iouThresholds: number[];
  private apMethod: 'all_points' | 'coco_101';
  private confThreshold: number;
  private allPredictions: StoredPred[] = [];
  private allGroundTruths: StoredGT[] = [];
  private imageCounter: number = 0;

  constructor(options?: {
    labelMap?: Record<string, string>;
    iouThresholds?: number[] | string;
    apMethod?: 'all_points' | 'coco_101';
    confThreshold?: number;
  }) {
    this.labelMap = options?.labelMap || {};
    this.iouThresholds = parseIoUThresholds(options?.iouThresholds);
    this.apMethod = options?.apMethod || 'all_points';
    this.confThreshold = options?.confThreshold ?? 0.0;
  }

  public update(predictions: PredictionBox[], groundTruths: GroundTruthBox[]): void {
    const imgIdx = this.imageCounter++;

    for (const p of predictions) {
      const score = p.score ?? 1.0;
      if (score < this.confThreshold) continue;

      const rawLabel = p.label;
      const mappedLabel = this.labelMap[rawLabel] || rawLabel;

      this.allPredictions.push({
        imageId: imgIdx,
        bbox: p.bbox,
        label: mappedLabel,
        score,
        origLabel: rawLabel,
      });
    }

    for (const gt of groundTruths) {
      this.allGroundTruths.push({
        imageId: imgIdx,
        bbox: gt.bbox,
        label: gt.label,
      });
    }
  }

  private calculateAPForClass(
    clsPreds: StoredPred[],
    clsGts: StoredGT[],
    iouThreshold: number
  ): [number, number, number] {
    const totalGts = clsGts.length;
    if (totalGts === 0 || clsPreds.length === 0) {
      return [0.0, 0.0, 0.0];
    }

    // Sort predictions by confidence descending
    const sortedPreds = [...clsPreds].sort((a, b) => b.score - a.score);

    // Group ground truths by image ID
    const gtByImg = new Map<number, { idx: number; bbox: BoundingBox }[]>();
    for (let idx = 0; idx < clsGts.length; idx++) {
      const gt = clsGts[idx];
      if (!gtByImg.has(gt.imageId)) {
        gtByImg.set(gt.imageId, []);
      }
      gtByImg.get(gt.imageId)!.push({ idx, bbox: gt.bbox });
    }

    const matchedGtIndices = new Set<number>();
    const tp: number[] = new Array(sortedPreds.length).fill(0);
    const fp: number[] = new Array(sortedPreds.length).fill(0);

    for (let i = 0; i < sortedPreds.length; i++) {
      const pred = sortedPreds[i];
      let bestIoU = -1.0;
      let bestGtIdx = -1;

      const imgGts = gtByImg.get(pred.imageId);
      if (imgGts) {
        for (const { idx, bbox } of imgGts) {
          const iou = computeIoU(pred.bbox, bbox);
          if (iou > bestIoU) {
            bestIoU = iou;
            bestGtIdx = idx;
          }
        }
      }

      if (bestIoU >= iouThreshold) {
        if (!matchedGtIndices.has(bestGtIdx)) {
          tp[i] = 1.0;
          matchedGtIndices.add(bestGtIdx);
        } else {
          fp[i] = 1.0; // Duplicate detection
        }
      } else {
        fp[i] = 1.0;
      }
    }

    // Cumulative sums
    let cumTp = 0;
    let cumFp = 0;
    const recalls: number[] = [];
    const precisions: number[] = [];

    for (let i = 0; i < sortedPreds.length; i++) {
      cumTp += tp[i];
      cumFp += fp[i];
      recalls.push(cumTp / totalGts);
      precisions.push(cumTp / (cumTp + cumFp + 1e-16));
    }

    let ap = 0.0;
    if (this.apMethod === 'coco_101') {
      // 101-point interpolation at points 0.00:0.01:1.00
      const precsAtRec: number[] = [];
      for (let rIdx = 0; rIdx <= 100; rIdx++) {
        const r = rIdx / 100.0;
        let maxPrec = 0.0;
        for (let i = 0; i < recalls.length; i++) {
          if (recalls[i] >= r && precisions[i] > maxPrec) {
            maxPrec = precisions[i];
          }
        }
        precsAtRec.push(maxPrec);
      }
      ap = precsAtRec.reduce((acc, v) => acc + v, 0) / 101.0;
    } else {
      // Continuous all-points interpolation (PASCAL VOC 2012+ / AUC)
      const mrec = [0.0, ...recalls, 1.0];
      const mpre = [0.0, ...precisions, 0.0];

      // Precision envelope
      for (let j = mpre.length - 2; j >= 0; j--) {
        mpre[j] = Math.max(mpre[j], mpre[j + 1]);
      }

      // Integrate
      for (let i = 0; i < mrec.length - 1; i++) {
        if (mrec[i + 1] !== mrec[i]) {
          ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1];
        }
      }
    }

    const finalPrecision = precisions.length > 0 ? precisions[precisions.length - 1] : 0.0;
    const finalRecall = recalls.length > 0 ? recalls[recalls.length - 1] : 0.0;

    return [ap, finalPrecision, finalRecall];
  }

  public computeMetrics(metadata?: Record<string, any>): EvaluationReport {
    const classSet = new Set<string>();
    for (const g of this.allGroundTruths) classSet.add(g.label);
    for (const p of this.allPredictions) classSet.add(p.label);
    const allClasses = Array.from(classSet).sort();

    const report: EvaluationReport = {
      metadata: {
        model_name: metadata?.model_name || 'VLM-Model',
        dataset_path: metadata?.dataset_path || 'datasets/sample',
        dataset_split: metadata?.dataset_split || 'all',
        conf_threshold: this.confThreshold,
        mode: metadata?.mode || 'evaluate',
        timestamp: new Date().toISOString(),
        ...metadata,
      },
      per_iou: {},
      mAP_50_95: 0.0,
      mAP_50: 0.0,
      mAP_75: 0.0,
      mAP_mean: 0.0,
      is_coco_standard: false,
      image_count: this.imageCounter,
      total_predictions: this.allPredictions.length,
      total_ground_truths: this.allGroundTruths.length,
      per_class_summary: {},
    };

    const mapValues: number[] = [];

    for (const iouTh of this.iouThresholds) {
      const iouKey = `iou_${iouTh.toFixed(2)}`;
      const classMetrics: Record<string, ClassMetric> = {};
      const classAps: number[] = [];
      const classPrecs: number[] = [];
      const classRecs: number[] = [];

      for (const cls of allClasses) {
        const clsPreds = this.allPredictions.filter((p) => p.label === cls);
        const clsGts = this.allGroundTruths.filter((g) => g.label === cls);

        const [ap, precision, recall] = this.calculateAPForClass(clsPreds, clsGts, iouTh);

        classMetrics[cls] = {
          AP: ap,
          precision,
          recall,
          num_ground_truth: clsGts.length,
          num_predictions: clsPreds.length,
        };

        if (clsGts.length > 0) {
          classAps.push(ap);
          classPrecs.push(precision);
          classRecs.push(recall);
        }
      }

      const mAP = classAps.length > 0 ? classAps.reduce((a, b) => a + b, 0) / classAps.length : 0.0;
      const meanPrecision = classPrecs.length > 0 ? classPrecs.reduce((a, b) => a + b, 0) / classPrecs.length : 0.0;
      const meanRecall = classRecs.length > 0 ? classRecs.reduce((a, b) => a + b, 0) / classRecs.length : 0.0;

      report.per_iou[iouKey] = {
        mAP,
        mean_precision: meanPrecision,
        mean_recall: meanRecall,
        class_metrics: classMetrics,
      };

      mapValues.push(mAP);
      if (Math.abs(iouTh - 0.50) < 1e-4) report.mAP_50 = mAP;
      if (Math.abs(iouTh - 0.75) < 1e-4) report.mAP_75 = mAP;
    }

    report.mAP_mean = mapValues.length > 0 ? mapValues.reduce((a, b) => a + b, 0) / mapValues.length : 0.0;

    const cocoKeys = COCO_IOU_THRESHOLDS.map((th) => `iou_${th.toFixed(2)}`);
    const isCocoStd = cocoKeys.every((k) => k in report.per_iou);
    report.is_coco_standard = isCocoStd;

    if (isCocoStd) {
      const cocoMaps = cocoKeys.map((k) => report.per_iou[k].mAP);
      report.mAP_50_95 = cocoMaps.reduce((a, b) => a + b, 0) / cocoMaps.length;
    } else {
      report.mAP_50_95 = report.mAP_mean;
    }

    // Per-class summary across thresholds
    for (const cls of allClasses) {
      const clsApsAcross = this.iouThresholds.map((th) => report.per_iou[`iou_${th.toFixed(2)}`].class_metrics[cls]?.AP ?? 0.0);
      const ap50 = report.per_iou['iou_0.50']?.class_metrics[cls]?.AP ?? 0.0;
      const ap75 = report.per_iou['iou_0.75']?.class_metrics[cls]?.AP ?? 0.0;
      const prec50 = report.per_iou['iou_0.50']?.class_metrics[cls]?.precision ?? 0.0;
      const rec50 = report.per_iou['iou_0.50']?.class_metrics[cls]?.recall ?? 0.0;

      let apCoco = 0.0;
      if (isCocoStd) {
        const cocoClsAps = cocoKeys.map((k) => report.per_iou[k].class_metrics[cls]?.AP ?? 0.0);
        apCoco = cocoClsAps.reduce((a, b) => a + b, 0) / cocoClsAps.length;
      } else {
        apCoco = clsApsAcross.length > 0 ? clsApsAcross.reduce((a, b) => a + b, 0) / clsApsAcross.length : 0.0;
      }

      const numGts = this.allGroundTruths.filter((g) => g.label === cls).length;
      const numPreds = this.allPredictions.filter((p) => p.label === cls).length;

      report.per_class_summary[cls] = {
        AP_50_95: apCoco,
        AP_50: ap50,
        AP_75: ap75,
        AP_mean: clsApsAcross.length > 0 ? clsApsAcross.reduce((a, b) => a + b, 0) / clsApsAcross.length : 0.0,
        precision_50: prec50,
        recall_50: rec50,
        num_ground_truth: numGts,
        num_predictions: numPreds,
      };
    }

    return report;
  }
}
