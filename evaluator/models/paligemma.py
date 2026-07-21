import re
import torch
from typing import Dict, List, Any, Tuple, Optional
from PIL import Image
from evaluator.models.base import BaseVLMAdapter

class PaliGemmaAdapter(BaseVLMAdapter):
    """Adapter for Google PaliGemma models."""
    
    def __init__(self, model_name: str, device: str, classes: Optional[List[str]] = None) -> None:
        super().__init__(model_name, device)
        self.classes = classes or []
        self.model = None
        self.processor = None
        self.resolved_device = None

    def _ensure_loaded(self) -> None:
        """Lazily loads the processor and model on the configured device."""
        if self.model is not None:
            return
            
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
        
        if self.device == "auto":
            self.resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.resolved_device = self.device
            
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_name
        ).to(self.resolved_device)
        
        if self.resolved_device == "cuda":
            self.model = self.model.to(torch.float16)

    def predict(self, image: Image.Image) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        
        # Build prompt: "detect {class1} ; {class2} ..."
        if self.classes:
            class_str = " ; ".join(self.classes)
            prompt = f"detect {class_str}"
        else:
            prompt = "detect"
            
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.resolved_device) for k, v in inputs.items()}
        
        if self.resolved_device == "cuda":
            # PaliGemma pixel values are typically float32, keep float32 or cast to float16 depending on precision
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
            
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024
            )
            
        # Extract only the newly generated tokens
        input_len = inputs["input_ids"].shape[-1]
        generated_ids = generated_ids[0][input_len:]
        generated_text = self.processor.decode(generated_ids, skip_special_tokens=False)
        
        return self.parse_output(generated_text, image.size)

    def parse_output(self, generated_text: str, image_size: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Parses location tokens from PaliGemma output.
        
        Format example: "<loc0120><loc0050><loc0800><loc0900> person ; <loc0010>..."
        PaliGemma uses [ymin, xmin, ymax, xmax] scaled [0, 1023].
        """
        # Regex to find: <locXXXX><locXXXX><locXXXX><locXXXX> label
        pattern = re.compile(r'<loc(\d{4})>\s*<loc(\d{4})>\s*<loc(\d{4})>\s*<loc(\d{4})>\s*([^;<>]+)')
        matches = pattern.findall(generated_text)
        
        detections = []
        for match in matches:
            ymin_raw, xmin_raw, ymax_raw, xmax_raw, label_raw = match
            
            # Convert raw coordinates to floats [0, 1]
            ymin = int(ymin_raw) / 1024.0
            xmin = int(xmin_raw) / 1024.0
            ymax = int(ymax_raw) / 1024.0
            xmax = int(xmax_raw) / 1024.0
            
            label = label_raw.strip()
            
            detections.append({
                "bbox": [
                    max(0.0, min(1.0, ymin)),
                    max(0.0, min(1.0, xmin)),
                    max(0.0, min(1.0, ymax)),
                    max(0.0, min(1.0, xmax))
                ],
                "label": label,
                "score": 1.0  # PaliGemma generation decoder does not yield confidences by default
            })
            
        return detections
