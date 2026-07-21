import os
from unittest.mock import patch, MagicMock
from PIL import Image
from evaluator.visualizer import draw_boxes, create_side_by_side, save_visualization

def test_draw_boxes() -> None:
    # Create small mock image
    img = Image.new("RGB", (100, 100), color="white")
    annotations = [
        {"bbox": [0.1, 0.1, 0.5, 0.5], "label": "person", "score": 0.9},
        {"bbox": [0.2, 0.2, 0.6, 0.6], "label": "car"}
    ]
    
    drawn = draw_boxes(img, annotations, is_prediction=True)
    
    assert isinstance(drawn, Image.Image)
    assert drawn.size == (100, 100)

def test_create_side_by_side() -> None:
    img = Image.new("RGB", (100, 100), color="white")
    gts = [{"bbox": [0.1, 0.1, 0.5, 0.5], "label": "person"}]
    preds = [{"bbox": [0.12, 0.12, 0.48, 0.48], "label": "person", "score": 0.92}]
    
    combined = create_side_by_side(img, gts, preds)
    
    assert isinstance(combined, Image.Image)
    # The output width is double (100 * 2 = 200), height includes the 35px header bar (100 + 35 = 135)
    assert combined.size == (200, 135)

@patch("os.makedirs")
@patch("PIL.Image.Image.save")
def test_save_visualization(mock_save, mock_makedirs) -> None:
    img = Image.new("RGB", (200, 135), color="white")
    
    # Save call with a full file_name path (should extract base name)
    save_visualization(img, "/mock/output_dir", "/some/source/path/frame_000070.jpg")
    
    # Verify os.makedirs was called on target output dir
    mock_makedirs.assert_called_once_with("/mock/output_dir", exist_ok=True)
    
    # Verify image.save was called with correct concatenated path
    expected_path = os.path.join("/mock/output_dir", "frame_000070.jpg")
    mock_save.assert_called_once_with(expected_path)
