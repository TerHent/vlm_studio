import io
import json
import base64
import requests
from typing import Dict, List, Any, Tuple, Optional
from PIL import Image
from evaluator.models.base import BaseVLMAdapter

class LMStudioAdapter(BaseVLMAdapter):
    """Adapter for OpenAI-compatible local endpoints (e.g., LM Studio, vLLM, Ollama) hosting VLMs."""
    
    def __init__(self, model_name: str, device: str, classes: Optional[List[str]] = None) -> None:
        super().__init__(model_name, device)
        self.classes = classes or []
        
        # Load API Base URL from env or fallback to default LM Studio port 1234
        import os
        api_base = os.environ.get("LM_STUDIO_API_BASE") or os.environ.get("OPENAI_API_BASE") or "http://127.0.0.1:1234/v1"
        self.endpoint = f"{api_base.rstrip('/')}/chat/completions"

    def predict(self, image: Image.Image) -> List[Dict[str, Any]]:
        # 1. Convert Image to base64 string
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # 2. Build prompt with class constraints
        class_str = ", ".join(self.classes) if self.classes else "objects of interest"
        prompt = (
            f"Identify and locate the coordinates of all {class_str} in the image. "
            "You MUST format the output as a JSON list of objects: "
            '[{"bbox": [xmin, ymin, xmax, ymax], "label": "class_name"}] '
            "where xmin, ymin, xmax, ymax are coordinates scaled from 0 to 1000 "
            "representing the bounding box relative to the image size. "
            "Do not write explanations. Return ONLY the raw valid JSON list."
        )

        # 3. Build OpenAI-compatible request payload
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_str}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.0,
            "max_tokens": 1024
        }
        
        headers = {"Content-Type": "application/json"}
        
        # 4. Query the API
        try:
            response = requests.post(self.endpoint, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            res_data = response.json()
            content = res_data["choices"][0]["message"]["content"]
            return self.parse_output(content, image.size)
        except Exception as e:
            raise RuntimeError(f"Failed to query LM Studio API at {self.endpoint}: {e}")

    def parse_output(self, generated_text: str, image_size: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Parses JSON-structured predictions from the API text response."""
        text_clean = generated_text.strip() if generated_text else ""
        
        # Short-circuit empty response or explicit conversational "no detections" responses
        no_detection_keywords = ["no objects", "no detections", "no items", "none detected", "no hands", "no cups", "no class"]
        if not text_clean or text_clean.lower() == "none" or any(keyword in text_clean.lower() for keyword in no_detection_keywords):
            return []
        
        # Attempt to strip markdown code blocks
        if "```json" in text_clean:
            text_clean = text_clean.split("```json")[1].split("```")[0].strip()
        elif "```" in text_clean:
            text_clean = text_clean.split("```")[1].split("```")[0].strip()

        # Pre-sanitize common LLM JSON syntax hallucinations
        import re
        text_clean = text_clean.replace("]]", "]")
        text_clean = re.sub(r',\s*([\]}])', r'\1', text_clean)  # Strip trailing commas
            
        detections_raw = []
        is_parsed = False
        
        json_error_msg = ""
        # 1. Try standard JSON parsing
        try:
            parsed = json.loads(text_clean)
            if isinstance(parsed, list):
                detections_raw = parsed
            else:
                detections_raw = [parsed]
            is_parsed = True
        except json.JSONDecodeError as err:
            json_error_msg = str(err)
            # Fallback search for outer brackets
            start = text_clean.find("[")
            end = text_clean.rfind("]") + 1
            if start != -1 and end != 0:
                try:
                    parsed = json.loads(text_clean[start:end])
                    if isinstance(parsed, list):
                        detections_raw = parsed
                    else:
                        detections_raw = [parsed]
                    is_parsed = True
                except json.JSONDecodeError:
                    pass

        # 2. If standard JSON parsing failed, apply regex-based recovery
        if not is_parsed:
            print("\n" + "-" * 60)
            print("[Debug Parser] JSON Decode Failure Notice:")
            print("  Expected Syntax : [{'bbox': [xmin, ymin, xmax, ymax], 'label': 'class_name'}]")
            if json_error_msg:
                print(f"  JSON Error      : {json_error_msg}")
            print(f"  LLM Raw Output  :\n{generated_text.strip()}")
            print("-" * 60)
            print("Attempting regex recovery...")
            import re
            
            # Pattern A: bbox then label, e.g. {"bbox": [100, 200, 300, 400]], "label": "Cup"}
            # Supports duplicate closing brackets "]]" by matching \]+
            pattern_bbox_label = re.compile(
                r'"?(?:bbox_2d|bbox|box|coordinates)"?\s*:\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]+[^}]+?"?(?:label|category|name)"?\s*:\s*"([^"]+)"',
                re.IGNORECASE
            )
            
            # Pattern B: label then bbox, e.g. {"label": "Cup", "bbox": [100, 200, 300, 400]]}
            pattern_label_bbox = re.compile(
                r'"?(?:label|category|name)"?\s*:\s*"([^"]+)"[^}]+?"?(?:bbox_2d|bbox|box|coordinates)"?\s*:\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]+',
                re.IGNORECASE
            )
            
            matches_a = pattern_bbox_label.findall(text_clean)
            matches_b = pattern_label_bbox.findall(text_clean)
            
            for m in matches_a:
                detections_raw.append({
                    "bbox": [float(m[0]), float(m[1]), float(m[2]), float(m[3])],
                    "label": m[4],
                    "score": 1.0
                })
                
            for m in matches_b:
                detections_raw.append({
                    "bbox": [float(m[1]), float(m[2]), float(m[3]), float(m[4])],
                    "label": m[0],
                    "score": 1.0
                })
                
            # If still no detections, try loose custom text fallback (e.g. Milk Pitcher [309, 500, 423, 658])
            if not detections_raw:
                pattern_loose = re.compile(
                    r'(?:\b(?!(?:bbox|box|coordinates|label|category|name)\b))([a-zA-Z0-9\s_-]+?)\s*[:\-=]?\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]+',
                    re.IGNORECASE
                )
                matches_loose = pattern_loose.findall(text_clean)
                for m in matches_loose:
                    lbl = m[0].strip()
                    # Skip common keywords
                    if lbl.lower() in ("bbox", "box", "coordinates", "label", "category", "name", "score", "confidence", "detections"):
                        continue
                    detections_raw.append({
                        "bbox": [float(m[1]), float(m[2]), float(m[3]), float(m[4])],
                        "label": lbl,
                        "score": 1.0
                    })
                    
            if detections_raw:
                print(f"Successfully recovered {len(detections_raw)} detections using regex extraction.")
            else:
                print(f"[Warning] Regex recovery failed. Raw text was:\n{generated_text}")
                return []
                
        detections = []
        for det in detections_raw:
            if not isinstance(det, dict):
                continue
                
            bbox = det.get("bbox") or det.get("bbox_2d") or det.get("box") or det.get("coordinates")
            label = det.get("label") or det.get("category") or det.get("name")
            score = det.get("score") or det.get("confidence") or 1.0
            
            if bbox is None or label is None or len(bbox) != 4:
                continue
                
            xmin, ymin, xmax, ymax = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            
            # Scale coordinates down to [0.0, 1.0] if they exceed 1.0001
            is_scaled_1000 = any(c > 1.0001 for c in bbox)
            if is_scaled_1000:
                ymin /= 1000.0
                xmin /= 1000.0
                ymax /= 1000.0
                xmax /= 1000.0
                
            detections.append({
                "bbox": [
                    max(0.0, min(1.0, ymin)),
                    max(0.0, min(1.0, xmin)),
                    max(0.0, min(1.0, ymax)),
                    max(0.0, min(1.0, xmax))
                ],
                "label": str(label).strip(),
                "score": float(score)
            })
            
        return detections
