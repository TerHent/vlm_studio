import os
from unittest.mock import MagicMock, patch
from evaluator.dataset import DatasetLoader

@patch("evaluator.dataset.load_from_disk")
def test_dataset_loader_custom_parsing(mock_load_from_disk) -> None:
    # Set up mock Hugging Face dataset row
    mock_item = {
        "file_name": "frame_000001.jpg",
        "response": "Detections: Milk Pitcher [309,500,423,658], Cup [394,318,488,452]"
    }
    
    # Create mock dataset object
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 1
    mock_dataset.__iter__.return_value = [mock_item]
    mock_dataset.features = {"response": None}
    
    mock_load_from_disk.return_value = mock_dataset
    
    # Initialize Loader on a mock path
    with patch("os.path.exists", return_value=True):
        loader = DatasetLoader(dataset_path="/mock/path", split="validation")
        
    # Verify unique class names are successfully extracted from response strings
    assert sorted(loader.category_names) == ["Cup", "Milk Pitcher"]
    
    # Mock Image loading to bypass filesystem check
    mock_image = MagicMock()
    mock_image.size = (640, 480)
    mock_image.convert.return_value = mock_image
    
    with patch("os.path.exists", return_value=True), \
         patch("PIL.Image.open", return_value=mock_image):
        results = list(loader)
        
    img, gts, file_name = results[0]
    assert file_name == "frame_000001.jpg"
    
    assert len(gts) == 2
    # Verify first bbox: Milk Pitcher [309, 500, 423, 658] (xmin, ymin, xmax, ymax)
    # Expected bbox [ymin, xmin, ymax, xmax]: [0.500, 0.309, 0.658, 0.423]
    assert gts[0]["label"] == "Milk Pitcher"
    assert abs(gts[0]["bbox"][0] - 0.500) < 1e-6
    assert abs(gts[0]["bbox"][1] - 0.309) < 1e-6
    assert abs(gts[0]["bbox"][2] - 0.658) < 1e-6
    assert abs(gts[0]["bbox"][3] - 0.423) < 1e-6
    
    # Verify second bbox: Cup [394, 318, 488, 452] (xmin, ymin, xmax, ymax)
    # Expected bbox [ymin, xmin, ymax, xmax]: [0.318, 0.394, 0.452, 0.488]
    assert gts[1]["label"] == "Cup"
    assert abs(gts[1]["bbox"][0] - 0.318) < 1e-6
    assert abs(gts[1]["bbox"][1] - 0.394) < 1e-6
    assert abs(gts[1]["bbox"][2] - 0.452) < 1e-6
    assert abs(gts[1]["bbox"][3] - 0.488) < 1e-6

@patch("evaluator.dataset.load_from_disk")
@patch("datasets.concatenate_datasets")
def test_dataset_loader_all_split(mock_concat, mock_load_from_disk) -> None:
    # Set up mock DatasetDict with train and validation splits
    train_split = MagicMock()
    val_split = MagicMock()
    mock_dict = {"train": train_split, "validation": val_split}
    
    mock_load_from_disk.return_value = mock_dict
    
    # Initialize Loader on a mock path with split="all"
    with patch("os.path.exists", return_value=True):
        loader = DatasetLoader(dataset_path="/mock/path", split="all")
        
    # Verify concatenate_datasets was called with the splits list
    mock_concat.assert_called_once_with([train_split, val_split])
