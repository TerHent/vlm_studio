import os
import json
from typing import Dict, List, Any, Tuple

def save_prediction_cache(
    filepath: str, 
    samples: List[Dict[str, Any]], 
    metadata: Dict[str, Any]
) -> None:
    """Saves raw model predictions and ground truth records to a JSON cache file. Overwrites existing file."""
    output_dir = os.path.dirname(filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    data = {
        "metadata": metadata,
        "samples": samples
    }
    
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def load_prediction_cache(filepath: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Loads prediction cache from a JSON file.
    
    Returns:
        Tuple of (samples list, metadata dict)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Prediction cache file not found at '{filepath}'. "
            f"Please run inference first using '--mode predict' or '--mode all'."
        )
        
    with open(filepath, "r") as f:
        data = json.load(f)
        
    samples = data.get("samples", [])
    metadata = data.get("metadata", {})
    return samples, metadata
