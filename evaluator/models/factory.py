from typing import List, Optional
from evaluator.models.base import BaseVLMAdapter
from evaluator.models.florence2 import Florence2Adapter
from evaluator.models.paligemma import PaliGemmaAdapter
from evaluator.models.lmstudio import LMStudioAdapter

def get_model_adapter(
    model_name: str, 
    device: str, 
    classes: Optional[List[str]] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None
) -> BaseVLMAdapter:
    """Factory function to resolve and instantiate VLM adapters by name."""
    name_lower = model_name.lower()
    
    if "florence" in name_lower:
        return Florence2Adapter(model_name=model_name, device=device)
    elif "paligemma" in name_lower:
        return PaliGemmaAdapter(model_name=model_name, device=device, classes=classes)
    else:
        # Fall back to LM Studio / local OpenAI-compatible endpoint hosting VLMs
        return LMStudioAdapter(
            model_name=model_name, 
            device=device, 
            classes=classes,
            api_base=api_base,
            api_key=api_key
        )



