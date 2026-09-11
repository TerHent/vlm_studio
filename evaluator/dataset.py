import os
import re
import json
from typing import Iterator, Tuple, List, Dict, Any, Optional
from PIL import Image
from datasets import load_dataset, load_from_disk

def is_coco_json_file(filepath: str) -> bool:
    """Checks if a file is a valid COCO annotation JSON."""
    if not (os.path.isfile(filepath) and filepath.endswith(".json")):
        return False
    try:
        with open(filepath, "r") as f:
            chunk = f.read(8192)
            if '"images"' in chunk and ('"annotations"' in chunk or '"categories"' in chunk):
                return True
        if os.path.getsize(filepath) < 50 * 1024 * 1024:
            with open(filepath, "r") as f:
                data = json.load(f)
            return isinstance(data, dict) and "images" in data and ("annotations" in data or "categories" in data)
        return False
    except Exception:
        return False

def find_coco_annotation_file(dirpath: str, split: str = "validation") -> Optional[str]:
    """Finds standard COCO annotation JSON in a directory."""
    split_variants = [split]
    if split in ["val", "validation"]:
        split_variants = ["val", "validation", "val2017", "validation2017"]
    elif split in ["train", "training"]:
        split_variants = ["train", "training", "train2017", "training2017"]
    elif split in ["test", "testing"]:
        split_variants = ["test", "testing", "test2017", "testing2017"]

    candidates = []
    for s in split_variants:
        candidates.extend([
            os.path.join(dirpath, f"instances_{s}.json"),
            os.path.join(dirpath, "annotations", f"instances_{s}.json"),
            os.path.join(dirpath, f"{s}.json"),
            os.path.join(dirpath, "annotations", f"{s}.json"),
        ])
    candidates.extend([
        os.path.join(dirpath, "_annotations.coco.json"),
        os.path.join(dirpath, "annotations.json"),
        os.path.join(dirpath, "annotations", "_annotations.coco.json"),
        os.path.join(dirpath, "annotations", "annotations.json"),
    ])

    for c in candidates:
        if os.path.isfile(c) and is_coco_json_file(c):
            return c
            
    for sub in ["annotations", ""]:
        search_dir = os.path.join(dirpath, sub) if sub else dirpath
        if os.path.isdir(search_dir):
            try:
                for fname in sorted(os.listdir(search_dir)):
                    if fname.endswith(".json"):
                        full_p = os.path.join(search_dir, fname)
                        if is_coco_json_file(full_p):
                            return full_p
            except Exception:
                pass
    return None

class DatasetLoader:
    """Loads and formats datasets for Object Detection.
    
    Supports:
      1. Standard COCO JSON annotation files and directory structures.
      2. Local or hub Hugging Face datasets (with standard 'objects' sequence or flat columns).
      3. Custom text-based response coordinates (e.g. Barista workflow: 'Label [ymin, xmin, ymax, xmax]').
    """
    
    def __init__(
        self, 
        dataset_path: str, 
        split: str = "validation",
        images_dir: Optional[str] = None
    ) -> None:
        self.dataset_path = dataset_path
        self.split = split
        self.images_dir = images_dir
        self.is_coco = False
        self.coco_json_path = None
        self.category_names = []
        
        coco_candidate = None
        if os.path.exists(dataset_path):
            if os.path.isfile(dataset_path) and is_coco_json_file(dataset_path):
                coco_candidate = dataset_path
            elif os.path.isdir(dataset_path):
                coco_candidate = find_coco_annotation_file(dataset_path, split=split)
                
        if coco_candidate:
            self._init_coco(coco_candidate)
        else:
            self._init_huggingface()

    def _init_coco(self, coco_json_path: str) -> None:
        self.is_coco = True
        self.coco_json_path = os.path.abspath(coco_json_path)
        with open(self.coco_json_path, "r") as f:
            data = json.load(f)
            
        self.category_id_to_name: Dict[Any, str] = {}
        for cat in data.get("categories", []):
            cid = cat["id"]
            cname = str(cat.get("name", cid)).strip()
            self.category_id_to_name[cid] = cname
            self.category_id_to_name[str(cid)] = cname
            
        names = [str(cat.get("name", cat["id"])).strip() for cat in data.get("categories", [])]
        if not names and "annotations" in data:
            unique_cats = sorted(list(set(ann.get("category_id") for ann in data["annotations"] if "category_id" in ann)))
            names = [str(c) for c in unique_cats]
            for c in unique_cats:
                self.category_id_to_name[c] = str(c)
                self.category_id_to_name[str(c)] = str(c)

        self.category_names = list(dict.fromkeys(names))
        self.coco_images: List[Dict[str, Any]] = data.get("images", [])
        
        self.coco_annotations_by_img_id: Dict[Any, List[Dict[str, Any]]] = {}
        for ann in data.get("annotations", []):
            img_id = ann.get("image_id")
            if img_id not in self.coco_annotations_by_img_id:
                self.coco_annotations_by_img_id[img_id] = []
            self.coco_annotations_by_img_id[img_id].append(ann)
            if not isinstance(img_id, str):
                str_id = str(img_id)
                if str_id not in self.coco_annotations_by_img_id:
                    self.coco_annotations_by_img_id[str_id] = self.coco_annotations_by_img_id[img_id]

    def _resolve_image_path(self, file_name: str) -> Optional[str]:
        """Resolves file_name to an existing file path using search directories."""
        if not file_name:
            return None

        if os.path.isabs(file_name) and os.path.exists(file_name):
            return file_name
            
        search_dirs: List[str] = []
        if self.images_dir and os.path.isdir(self.images_dir):
            search_dirs.append(os.path.abspath(self.images_dir))
            
        base_dir = os.path.dirname(self.coco_json_path) if self.coco_json_path else ""
        parent_dir = os.path.dirname(base_dir) if base_dir else ""
        
        split_variants = [self.split]
        if self.split in ["val", "validation"]:
            split_variants = ["val", "validation", "val2017", "validation2017"]
        elif self.split in ["train", "training"]:
            split_variants = ["train", "training", "train2017", "training2017"]
        elif self.split in ["test", "testing"]:
            split_variants = ["test", "testing", "test2017", "testing2017"]

        candidate_dirs: List[str] = []
        if self.dataset_path and os.path.isdir(self.dataset_path):
            candidate_dirs.append(os.path.abspath(self.dataset_path))
            candidate_dirs.append(os.path.join(self.dataset_path, "images"))
            for s in split_variants:
                candidate_dirs.append(os.path.join(self.dataset_path, s))
                candidate_dirs.append(os.path.join(self.dataset_path, "images", s))

        if base_dir:
            candidate_dirs.append(base_dir)
            candidate_dirs.append(os.path.join(base_dir, "images"))
            for s in split_variants:
                candidate_dirs.append(os.path.join(base_dir, s))
                candidate_dirs.append(os.path.join(base_dir, "images", s))
                
        if parent_dir:
            candidate_dirs.append(parent_dir)
            candidate_dirs.append(os.path.join(parent_dir, "images"))
            for s in split_variants:
                candidate_dirs.append(os.path.join(parent_dir, s))
                candidate_dirs.append(os.path.join(parent_dir, "images", s))

        for d in candidate_dirs:
            if d and d not in search_dirs:
                search_dirs.append(d)
                
        base_file_name = os.path.basename(file_name)
        for d in search_dirs:
            p = os.path.join(d, file_name)
            if os.path.exists(p):
                return p
            if base_file_name != file_name:
                p_base = os.path.join(d, base_file_name)
                if os.path.exists(p_base):
                    return p_base

        if os.path.exists(file_name):
            return os.path.abspath(file_name)

        return None

    def _init_huggingface(self) -> None:
        self.is_coco = False
        dataset_raw = None
        if os.path.exists(self.dataset_path):
            try:
                dataset_raw = load_from_disk(self.dataset_path)
            except Exception:
                if self.split == "all":
                    dataset_raw = load_dataset(self.dataset_path)
                else:
                    dataset_raw = load_dataset(self.dataset_path, split=self.split)
        else:
            if self.split == "all":
                dataset_raw = load_dataset(self.dataset_path)
            else:
                dataset_raw = load_dataset(self.dataset_path, split=self.split)
                
        if isinstance(dataset_raw, dict):
            if self.split == "all":
                from datasets import concatenate_datasets
                self.dataset = concatenate_datasets(list(dataset_raw.values()))
            elif self.split in dataset_raw:
                self.dataset = dataset_raw[self.split]
            else:
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
        if self.is_coco:
            return len(self.coco_images)
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
        if self.is_coco:
            for idx, item in enumerate(self.coco_images):
                file_name = item.get("file_name") or f"image_{idx}.jpg"
                img_id = item.get("id")

                img_path = self._resolve_image_path(file_name)
                if not img_path:
                    continue

                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception:
                    continue

                w, h = img.size
                if w <= 0 or h <= 0:
                    w = item.get("width") or 1
                    h = item.get("height") or 1

                gts = []
                annotations = self.coco_annotations_by_img_id.get(img_id) or self.coco_annotations_by_img_id.get(str(img_id), [])
                for ann in annotations:
                    bbox = ann.get("bbox", [])
                    if len(bbox) != 4:
                        continue
                    xmin, ymin, bw, bh = bbox
                    if bw <= 0 or bh <= 0:
                        continue
                    # Standard COCO format: [xmin, ymin, width, height] in absolute pixel coordinates
                    # Normalized to [ymin, xmin, ymax, xmax] in [0.0, 1.0] range
                    ymin_norm = max(0.0, min(1.0, float(ymin) / h))
                    xmin_norm = max(0.0, min(1.0, float(xmin) / w))
                    ymax_norm = max(0.0, min(1.0, float(ymin + bh) / h))
                    xmax_norm = max(0.0, min(1.0, float(xmin + bw) / w))

                    cat_id = ann.get("category_id", ann.get("category"))
                    label = self.category_id_to_name.get(cat_id, str(cat_id))

                    gts.append({
                        "bbox": [ymin_norm, xmin_norm, ymax_norm, xmax_norm],
                        "label": label
                    })

                yield img, gts, file_name
            return

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
