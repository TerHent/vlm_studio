import numpy as np
from typing import Dict, List, Tuple, Any

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
    """Computes Object Detection metrics (mAP, Precision, Recall) at multiple IoU thresholds."""
    
    def __init__(self, label_map: Dict[str, str], iou_thresholds: List[float]) -> None:
        self.label_map = label_map
        self.iou_thresholds = sorted(iou_thresholds)
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
        
        # Compute AP using all-points interpolation
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))
        
        # Compute the precision envelope
        for j in range(len(mpre) - 2, -1, -1):
            mpre[j] = max(mpre[j], mpre[j + 1])
            
        # Integrate area under curve
        split_indices = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[split_indices + 1] - mrec[split_indices]) * mpre[split_indices + 1])
        
        # Final precision and recall values at the end of prediction list
        final_precision = float(precisions[-1]) if len(precisions) > 0 else 0.0
        final_recall = float(recalls[-1]) if len(recalls) > 0 else 0.0
        
        return float(ap), final_precision, final_recall

    def compute_metrics(self) -> Dict[str, Any]:
        """Calculates evaluation metrics over all collected state."""
        # Get unique classes present in ground truths or predictions
        all_classes = set(gt["label"] for gt in self.all_ground_truths) | set(pred["label"] for pred in self.all_predictions)
        all_classes = sorted(list(all_classes))
        
        report: Dict[str, Any] = {
            "per_iou": {},
            "mAP_50": 0.0,
            "mAP_50_95": 0.0,
            "image_count": self.image_counter,
            "total_predictions": len(self.all_predictions),
            "total_ground_truths": len(self.all_ground_truths)
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
            if abs(iou_th - 0.5) < 1e-4:
                report["mAP_50"] = mAP
                
        # Calculate standard mAP@[.50:.95] if thresholds match or are a range
        report["mAP_50_95"] = float(np.mean(map_values)) if len(map_values) > 0 else 0.0
        
        return report
