import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

# Standard COCO 10-step IoU thresholds: [0.50:0.95:0.05]
COCO_IOU_THRESHOLDS: List[float] = [
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
]

def parse_iou_thresholds(arg: Union[str, List[float], Tuple[float, ...], None]) -> List[float]:
    """Parses IoU thresholds from a CLI string, list of floats, or alias.
    
    Supports:
        - None / 'coco' / 'default' / 'standard': Standard COCO 10-step thresholds [0.50:0.95:0.05]
        - Range string: '0.5:0.95:0.05' or '0.5:0.95'
        - Comma-separated floats: '0.5' or '0.3,0.5,0.75'
        - List/Tuple of floats: [0.5, 0.75]
    """
    if arg is None:
        return list(COCO_IOU_THRESHOLDS)
        
    if isinstance(arg, (list, tuple)):
        thresholds = [float(x) for x in arg]
    elif isinstance(arg, str):
        clean_arg = arg.strip().lower()
        if clean_arg in ("coco", "default", "standard"):
            return list(COCO_IOU_THRESHOLDS)
            
        if ":" in clean_arg:
            parts = [float(p.strip()) for p in clean_arg.split(":")]
            if len(parts) == 2:
                start, end = parts
                step = 0.05
            elif len(parts) == 3:
                start, end, step = parts
            else:
                raise ValueError(f"Invalid range format '{arg}'. Use 'start:stop' or 'start:stop:step'.")
            if step <= 0:
                raise ValueError(f"Step in range '{arg}' must be positive.")
            # Use np.arange with buffer to ensure upper boundary inclusion
            vals = np.arange(start, end + (step / 2.0), step)
            thresholds = [round(float(v), 4) for v in vals]
        else:
            thresholds = [float(x.strip()) for x in clean_arg.split(",") if x.strip()]
    else:
        raise TypeError(f"Expected str, list of floats, or None for iou_thresholds, got {type(arg).__name__}")
        
    if not thresholds:
        raise ValueError("IoU thresholds cannot be empty.")
        
    for th in thresholds:
        if not (0.0 <= th <= 1.0):
            raise ValueError(f"IoU threshold {th} must be between 0.0 and 1.0.")
            
    # Return sorted unique list
    return sorted(list(set(round(th, 4) for th in thresholds)))

def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Computes Intersection over Union (IoU) of two bounding boxes.
    
    Boxes are in format [ymin, xmin, ymax, xmax].
    """
    ymin1, xmin1, ymax1, xmax1 = box1
    ymin2, xmin2, ymax2, xmax2 = box2
    
    y_min = max(ymin1, ymin2)
    x_min = max(xmin1, xmin2)
    y_max = min(ymax1, ymax2)
    x_max = min(xmax1, xmax2)
    
    if y_max <= y_min or x_max <= x_min:
        return 0.0
        
    intersection = (y_max - y_min) * (x_max - x_min)
    area1 = (ymax1 - ymin1) * (xmax1 - xmin1)
    area2 = (ymax2 - ymin2) * (xmax2 - xmin2)
    
    union = area1 + area2 - intersection
    if union <= 0.0:
        return 0.0
        
    return intersection / union

class DetectionEvaluator:
    """Computes Object Detection metrics (mAP, Precision, Recall) at multiple IoU thresholds,
    supporting standard COCO benchmark evaluation (mAP@[.50:.95], mAP@.50, mAP@.75).
    """
    
    def __init__(
        self, 
        label_map: Optional[Dict[str, str]] = None, 
        iou_thresholds: Optional[Union[List[float], Tuple[float, ...], str]] = None,
        ap_method: str = "all_points"
    ) -> None:
        self.label_map = label_map or {}
        self.iou_thresholds = parse_iou_thresholds(iou_thresholds)
        self.ap_method = ap_method
        self.all_predictions: List[Dict[str, Any]] = []
        self.all_ground_truths: List[Dict[str, Any]] = []
        self.image_counter = 0

    def update(self, predictions: List[Dict[str, Any]], ground_truths: List[Dict[str, Any]]) -> None:
        """Adds predictions and ground truths for a single image to the evaluator state.
        
        Args:
            predictions: List of dicts like: [{'bbox': [ymin, xmin, ymax, xmax], 'label': str, 'score': float}]
            ground_truths: List of dicts like: [{'bbox': [ymin, xmin, ymax, xmax], 'label': str}]
        """
        img_idx = self.image_counter
        self.image_counter += 1
        
        for p in predictions:
            # Map predictions to standard vocabulary
            raw_label = p["label"]
            mapped_label = self.label_map.get(raw_label, raw_label)
            self.all_predictions.append({
                "image_id": img_idx,
                "bbox": p["bbox"],
                "label": mapped_label,
                "score": p.get("score", 1.0),
                "orig_label": raw_label
            })
            
        for gt in ground_truths:
            self.all_ground_truths.append({
                "image_id": img_idx,
                "bbox": gt["bbox"],
                "label": gt["label"]
            })

    def _calculate_ap_for_class(
        self, 
        cls_preds: List[Dict[str, Any]], 
        cls_gts: List[Dict[str, Any]], 
        iou_threshold: float
    ) -> Tuple[float, float, float]:
        """Calculates AP, Precision, and Recall for a single class at a given IoU threshold."""
        total_gts = len(cls_gts)
        if total_gts == 0:
            return 0.0, 0.0, 0.0
            
        if len(cls_preds) == 0:
            return 0.0, 0.0, 0.0
            
        # Sort predictions by score descending
        cls_preds = sorted(cls_preds, key=lambda x: x["score"], reverse=True)
        
        # Group ground truths by image ID
        gt_by_img: Dict[int, List[Tuple[int, List[float]]]] = {}
        for idx, gt in enumerate(cls_gts):
            img_id = gt["image_id"]
            if img_id not in gt_by_img:
                gt_by_img[img_id] = []
            gt_by_img[img_id].append((idx, gt["bbox"]))
            
        # Track matched ground truths
        matched_gt_indices = set()
        
        tp = np.zeros(len(cls_preds))
        fp = np.zeros(len(cls_preds))
        
        for i, pred in enumerate(cls_preds):
            img_id = pred["image_id"]
            pred_box = pred["bbox"]
            
            best_iou = -1.0
            best_gt_idx = -1
            
            if img_id in gt_by_img:
                for gt_idx, gt_box in gt_by_img[img_id]:
                    iou = compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                        
            if best_iou >= iou_threshold:
                if best_gt_idx not in matched_gt_indices:
                    tp[i] = 1.0
                    matched_gt_indices.add(best_gt_idx)
                else:
                    fp[i] = 1.0  # Duplicate prediction
            else:
                fp[i] = 1.0
                
        cumulative_tp = np.cumsum(tp)
        cumulative_fp = np.cumsum(fp)
        
        recalls = cumulative_tp / total_gts
        precisions = cumulative_tp / (cumulative_tp + cumulative_fp + 1e-16)
        
        # Compute AP using selected method
        if self.ap_method == "coco_101":
            # 101-point COCO interpolation at recall points 0.00:0.01:1.00
            rec_thresholds = np.linspace(0.0, 1.0, 101)
            precisions_at_rec = []
            for r in rec_thresholds:
                mask = recalls >= r
                if np.any(mask):
                    precisions_at_rec.append(np.max(precisions[mask]))
                else:
                    precisions_at_rec.append(0.0)
            ap = float(np.mean(precisions_at_rec))
        else:
            # Continuous all-points interpolation (PASCAL VOC 2012+ / continuous AUC)
            mrec = np.concatenate(([0.0], recalls, [1.0]))
            mpre = np.concatenate(([0.0], precisions, [0.0]))
            
            # Compute the precision envelope
            for j in range(len(mpre) - 2, -1, -1):
                mpre[j] = max(mpre[j], mpre[j + 1])
                
            # Integrate area under curve
            split_indices = np.where(mrec[1:] != mrec[:-1])[0]
            ap = float(np.sum((mrec[split_indices + 1] - mrec[split_indices]) * mpre[split_indices + 1]))
        
        # Final precision and recall values at the end of prediction list
        final_precision = float(precisions[-1]) if len(precisions) > 0 else 0.0
        final_recall = float(recalls[-1]) if len(recalls) > 0 else 0.0
        
        return ap, final_precision, final_recall

    def compute_metrics(self) -> Dict[str, Any]:
        """Calculates evaluation metrics over all collected state."""
        # Get unique classes present in ground truths or predictions
        all_classes = set(gt["label"] for gt in self.all_ground_truths) | set(pred["label"] for pred in self.all_predictions)
        all_classes = sorted(list(all_classes))
        
        report: Dict[str, Any] = {
            "per_iou": {},
            "mAP_50_95": 0.0,
            "mAP_50": 0.0,
            "mAP_75": 0.0,
            "mAP_mean": 0.0,
            "is_coco_standard": False,
            "image_count": self.image_counter,
            "total_predictions": len(self.all_predictions),
            "total_ground_truths": len(self.all_ground_truths),
            "per_class_summary": {}
        }
        
        map_values = []
        
        for iou_th in self.iou_thresholds:
            iou_key = f"iou_{iou_th:.2f}"
            iou_report = {}
            class_aps = []
            class_precisions = []
            class_recalls = []
            
            for cls in all_classes:
                cls_preds = [p for p in self.all_predictions if p["label"] == cls]
                cls_gts = [g for g in self.all_ground_truths if g["label"] == cls]
                
                ap, precision, recall = self._calculate_ap_for_class(cls_preds, cls_gts, iou_th)
                
                iou_report[cls] = {
                    "AP": ap,
                    "precision": precision,
                    "recall": recall,
                    "num_ground_truth": len(cls_gts),
                    "num_predictions": len(cls_preds)
                }
                
                if len(cls_gts) > 0:
                    class_aps.append(ap)
                    class_precisions.append(precision)
                    class_recalls.append(recall)
                    
            mAP = float(np.mean(class_aps)) if len(class_aps) > 0 else 0.0
            mean_precision = float(np.mean(class_precisions)) if len(class_precisions) > 0 else 0.0
            mean_recall = float(np.mean(class_recalls)) if len(class_recalls) > 0 else 0.0
            
            report["per_iou"][iou_key] = {
                "mAP": mAP,
                "mean_precision": mean_precision,
                "mean_recall": mean_recall,
                "class_metrics": iou_report
            }
            
            map_values.append(mAP)
            if abs(iou_th - 0.50) < 1e-4:
                report["mAP_50"] = mAP
            if abs(iou_th - 0.75) < 1e-4:
                report["mAP_75"] = mAP
                
        report["mAP_mean"] = float(np.mean(map_values)) if len(map_values) > 0 else 0.0

        # Check if all 10 standard COCO thresholds (0.50:0.05:0.95) are present
        coco_keys = [f"iou_{th:.2f}" for th in COCO_IOU_THRESHOLDS]
        is_coco_std = all(k in report["per_iou"] for k in coco_keys)
        report["is_coco_standard"] = is_coco_std

        if is_coco_std:
            coco_maps = [report["per_iou"][k]["mAP"] for k in coco_keys]
            report["mAP_50_95"] = float(np.mean(coco_maps))
        else:
            report["mAP_50_95"] = report["mAP_mean"]

        # Build per-class summary across thresholds
        for cls in all_classes:
            cls_aps_across_ious = [
                report["per_iou"][f"iou_{th:.2f}"]["class_metrics"][cls]["AP"] 
                for th in self.iou_thresholds
            ]
            
            ap_50 = report["per_iou"]["iou_0.50"]["class_metrics"][cls]["AP"] if "iou_0.50" in report["per_iou"] else 0.0
            ap_75 = report["per_iou"]["iou_0.75"]["class_metrics"][cls]["AP"] if "iou_0.75" in report["per_iou"] else 0.0
            prec_50 = report["per_iou"]["iou_0.50"]["class_metrics"][cls]["precision"] if "iou_0.50" in report["per_iou"] else 0.0
            rec_50 = report["per_iou"]["iou_0.50"]["class_metrics"][cls]["recall"] if "iou_0.50" in report["per_iou"] else 0.0

            if is_coco_std:
                coco_cls_aps = [report["per_iou"][k]["class_metrics"][cls]["AP"] for k in coco_keys]
                ap_coco = float(np.mean(coco_cls_aps))
            else:
                ap_coco = float(np.mean(cls_aps_across_ious)) if cls_aps_across_ious else 0.0

            num_gts = len([g for g in self.all_ground_truths if g["label"] == cls])
            num_preds = len([p for p in self.all_predictions if p["label"] == cls])

            report["per_class_summary"][cls] = {
                "AP_50_95": ap_coco,
                "AP_50": ap_50,
                "AP_75": ap_75,
                "AP_mean": float(np.mean(cls_aps_across_ious)) if cls_aps_across_ious else 0.0,
                "precision_50": prec_50,
                "recall_50": rec_50,
                "num_ground_truth": num_gts,
                "num_predictions": num_preds
            }
        
        return report
