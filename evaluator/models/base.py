from abc import ABC, abstractmethod
from typing import Dict, List, Any
from PIL import Image

class BaseVLMAdapter(ABC):
    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device

    @abstractmethod
    def predict(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Runs VLM inference on a single image.
        
        Args:
            image: A PIL Image object.
            
        Returns:
            A list of dictionary detections, where each dict has:
            {
                "bbox": [ymin, xmin, ymax, xmax],  # Normalized [0.0, 1.0]
                "label": str,
                "score": float
            }
        """
        pass
