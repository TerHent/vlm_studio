import torch
from typing import Dict, List, Any, Tuple, Optional
from PIL import Image
from evaluator.models.base import BaseVLMAdapter

class Florence2Adapter(BaseVLMAdapter):
    """Adapter for Microsoft Florence-2 models."""
    
    def __init__(self, model_name: str, device: str) -> None:
        super().__init__(model_name, device)
        self.model = None
        self.processor = None
        self.resolved_device = None

    def _ensure_loaded(self) -> None:
        """Lazily loads the processor and model on the configured device."""
        if self.model is not None:
            return
            
        from transformers import AutoModelForCausalLM, AutoProcessor
        
        if self.device == "auto":
            self.resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.resolved_device = self.device
            
        # Florence-2 models require trust_remote_code=True
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        ).to(self.resolved_device)
        
        if self.resolved_device == "cuda":
            self.model = self.model.to(torch.float16)

    def predict(self, image: Image.Image) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        
        # Object detection task prompt for Florence-2
        prompt = "<OD>"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        
        inputs = {k: v.to(self.resolved_device) for k, v in inputs.items()}
        if self.resolved_device == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
            
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )
            
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        return self.parse_output(generated_text, image.size)

    def parse_output(self, generated_text: str, image_size: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Parses generated text from model output using processor post-processing."""
        self._ensure_loaded()
        
        w, h = image_size
        parsed_answer = self.processor.post_process_generation(
            generated_text, 
            task="<OD>", 
            image_size=(w, h)
        )
        
        detections = []
        if "<OD>" in parsed_answer:
            od_data = parsed_answer["<OD>"]
            bboxes = od_data.get("bboxes", [])
            labels = od_data.get("labels", [])
            for bbox, label in zip(bboxes, labels):
                # Florence-2 outputs absolute [xmin, ymin, xmax, ymax]
                x1, y1, x2, y2 = bbox
                
                # Normalize and convert to [ymin, xmin, ymax, xmax]
                ymin = y1 / h
                xmin = x1 / w
                ymax = y2 / h
                xmax = x2 / w
                
                detections.append({
                    "bbox": [
                        max(0.0, min(1.0, ymin)),
                        max(0.0, min(1.0, xmin)),
                        max(0.0, min(1.0, ymax)),
                        max(0.0, min(1.0, xmax))
                    ],
                    "label": label,
                    "score": 1.0  # Florence-2 standard output does not yield confidence scores
                })
        return detections
