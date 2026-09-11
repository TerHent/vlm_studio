export type BoundingBox = [number, number, number, number]; // [ymin, xmin, ymax, xmax] in normalized [0, 1]

export interface GroundTruthBox {
  bbox: BoundingBox;
  label: string;
}

export interface PredictionBox {
  bbox: BoundingBox;
  label: string;
  score: number;
  orig_label?: string;
}

export interface SampleRecord {
  file_name: string;
  image_url?: string;
  image_svg?: string;
  width: number;
  height: number;
  ground_truths: GroundTruthBox[];
  predictions: PredictionBox[];
}

export interface ClassMetric {
  AP: number;
  precision: number;
  recall: number;
  num_ground_truth: number;
  num_predictions: number;
}

export interface IoUReport {
  mAP: number;
  mean_precision: number;
  mean_recall: number;
  class_metrics: Record<string, ClassMetric>;
}

export interface ClassSummary {
  AP_50_95: number;
  AP_50: number;
  AP_75: number;
  AP_mean: number;
  precision_50: number;
  recall_50: number;
  num_ground_truth: number;
  num_predictions: number;
}

export interface EvaluationReport {
  metadata: {
    model_name: string;
    dataset_path: string;
    dataset_split: string;
    conf_threshold: number;
    mode: string;
    device?: string;
    api_base?: string;
    timestamp?: string;
    [key: string]: any;
  };
  per_iou: Record<string, IoUReport>;
  mAP_50_95: number;
  mAP_50: number;
  mAP_75: number;
  mAP_mean: number;
  is_coco_standard: boolean;
  image_count: number;
  total_predictions: number;
  total_ground_truths: number;
  per_class_summary: Record<string, ClassSummary>;
}

export interface PredictionCache {
  id: string;
  name: string;
  metadata: {
    model_name: string;
    dataset_path: string;
    dataset_split: string;
    [key: string]: any;
  };
  samples: SampleRecord[];
}
