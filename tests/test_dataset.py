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

def test_is_coco_json_file(tmp_path) -> None:
    from evaluator.dataset import is_coco_json_file
    
    # Valid COCO file
    valid_coco = tmp_path / "coco.json"
    import json
    valid_coco.write_text(json.dumps({"images": [], "annotations": [], "categories": []}))
    assert is_coco_json_file(str(valid_coco)) is True
    
    # Non-COCO json file (e.g. HuggingFace dataset_dict)
    hf_json = tmp_path / "dataset_dict.json"
    hf_json.write_text(json.dumps({"splits": ["train", "validation"]}))
    assert is_coco_json_file(str(hf_json)) is False
    
    # Non-existent file
    assert is_coco_json_file(str(tmp_path / "nonexistent.json")) is False
    
    # Non-json file
    txt_file = tmp_path / "file.txt"
    txt_file.write_text("hello")
    assert is_coco_json_file(str(txt_file)) is False

def test_coco_dataset_loader_from_file(tmp_path) -> None:
    import json
    from PIL import Image
    
    # Create dummy image
    img_path = tmp_path / "cat_01.jpg"
    img = Image.new("RGB", (640, 480), color=(255, 0, 0))
    img.save(img_path)
    
    # Create dummy COCO JSON
    coco_data = {
        "categories": [
            {"id": 1, "name": "cat"},
            {"id": 2, "name": "dog"}
        ],
        "images": [
            {"id": 101, "file_name": "cat_01.jpg", "width": 640, "height": 480}
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 101,
                "category_id": 1,
                "bbox": [100.0, 50.0, 200.0, 150.0]
            }
        ]
    }
    json_path = tmp_path / "instances_val.json"
    json_path.write_text(json.dumps(coco_data))
    
    # Load dataset directly from JSON file path
    loader = DatasetLoader(dataset_path=str(json_path))
    assert loader.is_coco is True
    assert len(loader) == 1
    assert loader.category_names == ["cat", "dog"]
    
    items = list(loader)
    assert len(items) == 1
    loaded_img, gts, file_name = items[0]
    assert file_name == "cat_01.jpg"
    assert loaded_img.size == (640, 480)
    assert len(gts) == 1
    assert gts[0]["label"] == "cat"
    
    ymin, xmin, ymax, xmax = gts[0]["bbox"]
    assert abs(ymin - (50.0 / 480.0)) < 1e-5
    assert abs(xmin - (100.0 / 640.0)) < 1e-5
    assert abs(ymax - (200.0 / 480.0)) < 1e-5
    assert abs(xmax - (300.0 / 640.0)) < 1e-5

def test_coco_dataset_loader_directory_and_images_dir(tmp_path) -> None:
    import json
    from PIL import Image
    
    coco_root = tmp_path / "coco_dir"
    coco_root.mkdir()
    annotations_dir = coco_root / "annotations"
    annotations_dir.mkdir()
    
    separate_images_dir = tmp_path / "images_folder"
    separate_images_dir.mkdir()
    
    img = Image.new("RGB", (800, 600), color=(0, 255, 0))
    img.save(separate_images_dir / "dog_01.jpg")
    
    coco_data = {
        "categories": [
            {"id": 10, "name": "dog"}
        ],
        "images": [
            {"id": 55, "file_name": "dog_01.jpg", "width": 800, "height": 600}
        ],
        "annotations": [
            {
                "id": 999,
                "image_id": 55,
                "category_id": 10,
                "bbox": [50.0, 60.0, 400.0, 300.0]
            }
        ]
    }
    json_path = annotations_dir / "instances_val2017.json"
    json_path.write_text(json.dumps(coco_data))
    
    # Test directory detection with split="validation" and external images_dir
    loader = DatasetLoader(
        dataset_path=str(coco_root),
        split="validation",
        images_dir=str(separate_images_dir)
    )
    assert loader.is_coco is True
    assert len(loader) == 1
    assert loader.category_names == ["dog"]
    
    items = list(loader)
    assert len(items) == 1
    _, gts, file_name = items[0]
    assert file_name == "dog_01.jpg"
    assert len(gts) == 1
    assert gts[0]["label"] == "dog"
    ymin, xmin, ymax, xmax = gts[0]["bbox"]
    assert abs(ymin - (60.0 / 600.0)) < 1e-5
    assert abs(xmin - (50.0 / 800.0)) < 1e-5
    assert abs(ymax - (360.0 / 600.0)) < 1e-5
    assert abs(xmax - (450.0 / 800.0)) < 1e-5
