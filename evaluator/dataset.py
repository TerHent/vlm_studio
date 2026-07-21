import os
import re
from typing import Iterator, Tuple, List, Dict, Any, Optional
from PIL import Image
from datasets import load_dataset, load_from_disk

class DatasetLoader:
    """Loads and formats local or hub Hugging Face datasets for Object Detection.
    
    Supports standard HF object detection structures as well as custom text-based
    response coordinates (e.g. "Detections: Class [ymin, xmin, ymax, xmax]").
    """
    
    def __init__(self, dataset_path: str, split: str = "validation") -> None:
        self.dataset_path = dataset_path
        self.split = split
        
        # Load Hugging Face dataset
        dataset_raw = None
        if os.path.exists(dataset_path):
            try:
                dataset_raw = load_from_disk(dataset_path)
            except Exception:
                if split == "all":
                    dataset_raw = load_dataset(dataset_path)
                else:
                    dataset_raw = load_dataset(dataset_path, split=split)
        else:
            if split == "all":
                dataset_raw = load_dataset(dataset_path)
            else:
                dataset_raw = load_dataset(dataset_path, split=split)
                
        # Resolve split or concatenate if "all" is requested
        if isinstance(dataset_raw, dict):
            if split == "all":
                from datasets import concatenate_datasets
                self.dataset = concatenate_datasets(list(dataset_raw.values()))
            elif split in dataset_raw:
                self.dataset = dataset_raw[split]
            else:
                # Default fallback
                first_key = list(dataset_raw.keys())[0]
                self.dataset = dataset_raw[first_key]
        else:
            self.dataset = dataset_raw
            
        self._extract_category_names()

    def _extract_category_names(self) -> None:
        """Attempts to extract category names mapping from dataset features metadata or parses text response."""
        self.category_names = []
        features = self.dataset.features
        
        # Scenario 1: Standard HF object detection structures with "objects" column
        if "objects" in features:
            objects_feat = features["objects"]
            # objects is sequence of features
            if hasattr(objects_feat, "feature"):
                inner_feat = objects_feat.feature
                if isinstance(inner_feat, dict) and "category" in inner_feat:
                    cat_feat = inner_feat["category"]
                    if hasattr(cat_feat, "names"):
                        self.category_names = cat_feat.names
                elif hasattr(inner_feat, "category"):
                    cat_feat = inner_feat.category
                    if hasattr(cat_feat, "names"):
                        self.category_names = cat_feat.names
                        
        # Scenario 2: Flat category names feature (ClassLabel)
        elif "category" in features:
            cat_feat = features["category"]
            if hasattr(cat_feat, "names"):
                self.category_names = cat_feat.names
        elif "label" in features:
            label_feat = features["label"]
            if hasattr(label_feat, "names"):
                self.category_names = label_feat.names

        # Scenario 3: Custom "response" text scanning (for datasets like Barista_workflow_small)
        if not self.category_names and "response" in features:
            label_set = set()
            pattern = re.compile(r'([a-zA-Z0-9\s_-]+)\s*\[\d+,\d+,\d+,\d+\]')
            for item in self.dataset:
                resp = item.get("response", "")
                if resp:
                    matches = pattern.findall(resp)
                    for lbl in matches:
                        label_set.add(lbl.strip())
            self.category_names = sorted(list(label_set))

    def __len__(self) -> int:
        return len(self.dataset)

    def _resolve_label(self, raw_label: Any) -> str:
        """Resolves label index to string if names mapping exists."""
        if isinstance(raw_label, int):
            if 0 <= raw_label < len(self.category_names):
                return self.category_names[raw_label]
            return str(raw_label)
        return str(raw_label)

    def _normalize_box(self, box: List[float], img_width: int, img_height: int) -> List[float]:
        """Converts box [xmin, ymin, w, h] or similar to normalized [ymin, xmin, ymax, xmax] range [0.0, 1.0]."""
        # We assume standard COCO format: [xmin, ymin, width, height]
        # Check if coordinates are already normalized
        is_normalized = all(0.0 <= c <= 1.0001 for c in box)
        
        if is_normalized:
            # Let's check format: VOC [xmin, ymin, xmax, ymax] or COCO [xmin, ymin, width, height]
            x0, y0, c2, c3 = box
            if c2 >= x0 and c3 >= y0:
                # voc: [xmin, ymin, xmax, ymax]
                ymin, xmin, ymax, xmax = y0, x0, c3, c2
            else:
                # coco: [xmin, ymin, w, h]
                ymin, xmin, ymax, xmax = y0, x0, min(1.0, y0 + c3), min(1.0, x0 + c2)
        else:
            # Absolute pixel coordinates
            x0, y0, c2, c3 = box
            # If voc absolute: xmax >= xmin and ymax >= ymin
            if c2 >= x0 and c3 >= y0 and c2 > img_width * 0.1 and c3 > img_height * 0.1:
                ymin = y0 / img_height
                xmin = x0 / img_width
                ymax = c3 / img_height
                xmax = c2 / img_width
            else:
                # coco absolute: [xmin, ymin, w, h]
                ymin = y0 / img_height
                xmin = x0 / img_width
                ymax = (y0 + c3) / img_height
                xmax = (x0 + c2) / img_width
                
        # Clamp to [0.0, 1.0]
        return [
            max(0.0, min(1.0, ymin)),
            max(0.0, min(1.0, xmin)),
            max(0.0, min(1.0, ymax)),
            max(0.0, min(1.0, xmax))
        ]

    def __iter__(self) -> Iterator[Tuple[Image.Image, List[Dict[str, Any]], str]]:
        """Iterates over the dataset, yielding (Image, list of ground truth dicts, file_name)."""
        for idx, item in enumerate(self.dataset):
            # Handle image loading
            img = item.get("image") or item.get("img")
            
            # If no image object is returned, check for file_name and load from folder
            if img is None:
                file_name = item.get("file_name")
                if file_name:
                    # Look inside dataset directory
                    img_path = os.path.join(self.dataset_path, file_name)
                    if os.path.exists(img_path):
                        img = img_path
                    else:
                        # Try validation or train subfolders
                        for sub in [self.split, "train", "validation"]:
                            path_test = os.path.join(self.dataset_path, sub, file_name)
                            if os.path.exists(path_test):
                                img = path_test
                                break

            if img is None:
                continue

            if not isinstance(img, Image.Image):
                try:
                    img = Image.open(img).convert("RGB")
                except Exception:
                    continue
            else:
                img = img.convert("RGB")
                
            w, h = img.size
            gts = []
            
            # Format ground truth annotations
            # Scenario A: Standard sequence format e.g. item['objects'] = {'bbox': [[...]], 'category': [...]}
            if "objects" in item and isinstance(item["objects"], dict):
                objects = item["objects"]
                bboxes = objects.get("bbox", [])
                categories = objects.get("category", []) or objects.get("label", [])
                for bbox, cat in zip(bboxes, categories):
                    norm_bbox = self._normalize_box(bbox, w, h)
                    gts.append({
                        "bbox": norm_bbox,
                        "label": self._resolve_label(cat)
                    })
            # Scenario B: Flat list columns: item['bbox'] and item['category']
            elif "bbox" in item and ("category" in item or "label" in item):
                bboxes = item["bbox"]
                categories = item.get("category") or item.get("label")
                if len(bboxes) > 0 and not isinstance(bboxes[0], list):
                    bboxes = [bboxes]
                    categories = [categories]
                for bbox, cat in zip(bboxes, categories):
                    norm_bbox = self._normalize_box(bbox, w, h)
                    gts.append({
                        "bbox": norm_bbox,
                        "label": self._resolve_label(cat)
                    })
            # Scenario C: Custom text "response" format (e.g. Barista_workflow_small: "Label [ymin, xmin, ymax, xmax]")
            elif "response" in item and isinstance(item["response"], str):
                response_str = item["response"]
                # Matches label and 4 coordinate integers (e.g., Milk Pitcher [309,500,423,658])
                pattern = re.compile(r'([a-zA-Z0-9\s_-]+)\s*\[(\d+),(\d+),(\d+),(\d+)\]')
                matches = pattern.findall(response_str)
                for match in matches:
                    # Coordinates in Barista workflow are scaled between 0 and 1000 in [xmin, ymin, xmax, ymax] order
                    label, xmin_raw, ymin_raw, xmax_raw, ymax_raw = match
                    xmin = int(xmin_raw) / 1000.0
                    ymin = int(ymin_raw) / 1000.0
                    xmax = int(xmax_raw) / 1000.0
                    ymax = int(ymax_raw) / 1000.0
                    
                    gts.append({
                        "bbox": [
                            max(0.0, min(1.0, ymin)),
                            max(0.0, min(1.0, xmin)),
                            max(0.0, min(1.0, ymax)),
                            max(0.0, min(1.0, xmax))
                        ],
                        "label": label.strip()
                    })
                    
            file_name = item.get("file_name") or f"image_{idx}.jpg"
            yield img, gts, file_name
