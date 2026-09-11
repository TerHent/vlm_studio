import os
import tempfile
import pytest
from evaluator.predictions import save_prediction_cache, load_prediction_cache

def test_save_and_load_prediction_cache() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, "test_cache.json")
        
        sample_data = [
            {
                "file_name": "frame_001.jpg",
                "ground_truths": [{"bbox": [0.1, 0.1, 0.5, 0.5], "label": "Cup"}],
                "predictions": [{"bbox": [0.12, 0.12, 0.48, 0.48], "label": "Cup", "score": 0.92}]
            }
        ]
        metadata = {"model_name": "test-model", "dataset_path": "test-dataset"}
        
        # Save
        save_prediction_cache(cache_path, sample_data, metadata)
        assert os.path.exists(cache_path)
        
        # Load
        loaded_samples, loaded_meta = load_prediction_cache(cache_path)
        assert len(loaded_samples) == 1
        assert loaded_samples[0]["file_name"] == "frame_001.jpg"
        assert loaded_samples[0]["predictions"][0]["label"] == "Cup"
        assert loaded_meta["model_name"] == "test-model"

def test_load_nonexistent_cache_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_prediction_cache("/nonexistent/path/cache.json")
