from unittest.mock import MagicMock
from evaluator.models.florence2 import Florence2Adapter
from evaluator.models.paligemma import PaliGemmaAdapter

def test_florence_parser() -> None:
    adapter = Florence2Adapter("mock-model", "cpu")
    # Mock model loading to run completely offline
    adapter._ensure_loaded = MagicMock()
    adapter.processor = MagicMock()
    
    # Mock post_process_generation output
    # Florence-2 outputs absolute [xmin, ymin, xmax, ymax]
    adapter.processor.post_process_generation.return_value = {
        "<OD>": {
            "bboxes": [[100.0, 150.0, 200.0, 250.0]],
            "labels": ["person"]
        }
    }
    
    # Image size: width=500, height=1000
    # ymin = 150 / 1000 = 0.15
    # xmin = 100 / 500 = 0.20
    # ymax = 250 / 1000 = 0.25
    # xmax = 200 / 500 = 0.40
    detections = adapter.parse_output("mock-tokens-text", (500, 1000))
    
    assert len(detections) == 1
    det = detections[0]
    assert det["label"] == "person"
    assert det["score"] == 1.0
    assert abs(det["bbox"][0] - 0.15) < 1e-6
    assert abs(det["bbox"][1] - 0.20) < 1e-6
    assert abs(det["bbox"][2] - 0.25) < 1e-6
    assert abs(det["bbox"][3] - 0.40) < 1e-6

def test_paligemma_parser() -> None:
    adapter = PaliGemmaAdapter("mock-model", "cpu")
    adapter._ensure_loaded = MagicMock()
    
    # Coordinate tokens correspond to bin / 1024
    generated_text = "<loc0128><loc0256><loc0512><loc0768> dog ; <loc0064><loc0032><loc0192><loc0224> cat"
    
    # PaliGemma coordinate normalization is independent of image size in the parser output
    detections = adapter.parse_output(generated_text, (800, 600))
    
    assert len(detections) == 2
    
    # First detection: dog
    # ymin = 128 / 1024 = 0.125
    # xmin = 256 / 1024 = 0.25
    # ymax = 512 / 1024 = 0.50
    # xmax = 768 / 1024 = 0.75
    det0 = detections[0]
    assert det0["label"] == "dog"
    assert abs(det0["bbox"][0] - 0.125) < 1e-6
    assert abs(det0["bbox"][1] - 0.25) < 1e-6
    assert abs(det0["bbox"][2] - 0.50) < 1e-6
    assert abs(det0["bbox"][3] - 0.75) < 1e-6
    
    # Second detection: cat
    # ymin = 64 / 1024 = 0.0625
    # xmin = 32 / 1024 = 0.03125
    # ymax = 192 / 1024 = 0.1875
    # xmax = 224 / 1024 = 0.21875
    det1 = detections[1]
    assert det1["label"] == "cat"
    assert abs(det1["bbox"][0] - 0.0625) < 1e-6
    assert abs(det1["bbox"][1] - 0.03125) < 1e-6
    assert abs(det1["bbox"][2] - 0.1875) < 1e-6
    assert abs(det1["bbox"][3] - 0.21875) < 1e-6

def test_lmstudio_parser() -> None:
    from evaluator.models.lmstudio import LMStudioAdapter
    adapter = LMStudioAdapter("mock-model", "cpu")
    
    # Valid markdown JSON block returned by the LLM
    llm_output = """
    ```json
    [
        {"bbox_2d": [100, 200, 300, 400], "label": "cup", "confidence": 0.95},
        {"bbox": [0.15, 0.25, 0.35, 0.45], "label": "milk"}
    ]
    ```
    """
    
    detections = adapter.parse_output(llm_output, (800, 600))
    
    assert len(detections) == 2
    
    # First detection (scaled 0-1000): bbox [xmin=100, ymin=200, xmax=300, ymax=400]
    # Expected bbox [ymin, xmin, ymax, xmax]: [0.2, 0.1, 0.4, 0.3]
    det0 = detections[0]
    assert det0["label"] == "cup"
    assert det0["score"] == 0.95
    assert abs(det0["bbox"][0] - 0.2) < 1e-6
    assert abs(det0["bbox"][1] - 0.1) < 1e-6
    assert abs(det0["bbox"][2] - 0.4) < 1e-6
    assert abs(det0["bbox"][3] - 0.3) < 1e-6
    
    # Second detection (already normalized 0-1): bbox [xmin=0.15, ymin=0.25, xmax=0.35, ymax=0.45]
    # Expected bbox [ymin, xmin, ymax, xmax]: [0.25, 0.15, 0.45, 0.35]
    det1 = detections[1]
    assert det1["label"] == "milk"
    assert det1["score"] == 1.0
    assert abs(det1["bbox"][0] - 0.25) < 1e-6
    assert abs(det1["bbox"][1] - 0.15) < 1e-6
    assert abs(det1["bbox"][2] - 0.45) < 1e-6
    assert abs(det1["bbox"][3] - 0.35) < 1e-6

    # Test malformed double bracket output parsing recovery: [xmin=318, ymin=526, xmax=371, ymax=664]
    # Expected bbox [ymin, xmin, ymax, xmax]: [0.526, 0.318, 0.664, 0.371]
    malformed_output = '[{"bbox": [318, 526, 371, 664]], "label": "Scale"}]'
    detections_recovered = adapter.parse_output(malformed_output, (800, 600))
    assert len(detections_recovered) == 1
    det_rec = detections_recovered[0]
    assert det_rec["label"] == "Scale"
    assert abs(det_rec["bbox"][0] - 0.526) < 1e-6
    assert abs(det_rec["bbox"][1] - 0.318) < 1e-6
    assert abs(det_rec["bbox"][2] - 0.664) < 1e-6
    assert abs(det_rec["bbox"][3] - 0.371) < 1e-6

    # Test loose custom text parsing recovery (no JSON formatting)
    loose_output = "Detections: Milk Pitcher [309, 500, 423, 658], Cup: [394, 318, 488, 452]"
    detections_loose = adapter.parse_output(loose_output, (800, 600))
    assert len(detections_loose) == 2
    
    # First loose detection: Milk Pitcher [xmin=309, ymin=500, xmax=423, ymax=658]
    # Expected bbox [ymin, xmin, ymax, xmax]: [0.5, 0.309, 0.658, 0.423]
    det_loose0 = detections_loose[0]
    assert det_loose0["label"] == "Milk Pitcher"
    assert abs(det_loose0["bbox"][0] - 0.5) < 1e-6
    assert abs(det_loose0["bbox"][1] - 0.309) < 1e-6
    assert abs(det_loose0["bbox"][2] - 0.658) < 1e-6
    assert abs(det_loose0["bbox"][3] - 0.423) < 1e-6

    # Second loose detection: Cup: [xmin=394, ymin=318, xmax=488, ymax=452]
    # Expected bbox [ymin, xmin, ymax, xmax]: [0.318, 0.394, 0.452, 0.488]
    det_loose1 = detections_loose[1]
    assert det_loose1["label"] == "Cup"
    assert abs(det_loose1["bbox"][0] - 0.318) < 1e-6
    assert abs(det_loose1["bbox"][1] - 0.394) < 1e-6
    assert abs(det_loose1["bbox"][2] - 0.452) < 1e-6
    assert abs(det_loose1["bbox"][3] - 0.488) < 1e-6

    # Test short-circuiting empty or conversational "no detections"
    assert adapter.parse_output("", (800, 600)) == []
    assert adapter.parse_output("   ", (800, 600)) == []
    assert adapter.parse_output("None", (800, 600)) == []
    assert adapter.parse_output("No objects detected in the image.", (800, 600)) == []

def test_lmstudio_adapter_api_config() -> None:
    from evaluator.models.lmstudio import LMStudioAdapter
    adapter = LMStudioAdapter(
        model_name="qwen-vl", 
        device="cpu", 
        api_base="http://localhost:8000/v1", 
        api_key="secret-key-123"
    )
    assert adapter.endpoint == "http://localhost:8000/v1/chat/completions"
    assert adapter.api_key == "secret-key-123"


