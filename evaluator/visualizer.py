import os
from typing import List, Dict, Any, Tuple
from PIL import Image, ImageDraw, ImageFont

def get_class_color(label: str) -> Tuple[int, int, int]:
    """Generates a stable, distinct RGB color for a given label string."""
    palette = [
        (46, 204, 113),   # Emerald Green
        (52, 152, 219),   # Blue
        (155, 89, 182),   # Purple
        (241, 196, 15),   # Yellow
        (230, 126, 34),   # Orange
        (26, 188, 156),   # Turquoise
        (231, 76, 60),    # Red
        (243, 156, 18),   # Dark Yellow/Orange
        (142, 68, 173),   # Dark Wisteria Purple
        (52, 73, 94),     # Greyish Blue
    ]
    # Simple stable character sum hash to keep colors consistent across python executions
    val = sum(ord(c) for c in label)
    return palette[val % len(palette)]

def draw_boxes(
    image: Image.Image, 
    annotations: List[Dict[str, Any]], 
    is_prediction: bool = False
) -> Image.Image:
    """Draws bounding boxes and labels on a copy of the image.
    
    Uses distinct colors for different categories.
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    w, h = img_copy.size
    
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for ann in annotations:
        bbox = ann.get("bbox")
        label = ann.get("label", "unknown")
        score = ann.get("score")
        
        if bbox is None or len(bbox) != 4:
            continue
            
        ymin, xmin, ymax, xmax = bbox
        
        # Convert normalized coordinates to absolute pixels
        x0 = int(xmin * w)
        y0 = int(ymin * h)
        x1 = int(xmax * w)
        y1 = int(ymax * h)
        
        # Clamp to image size
        x0 = max(0, min(w - 1, x0))
        y0 = max(0, min(h - 1, y0))
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        
        # Get stable color for this class
        color = get_class_color(label)
        
        # Draw bounding box rectangle
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        
        # Format label text
        if is_prediction and score is not None:
            text = f"{label} ({score:.2f})"
        else:
            text = label
            
        # Draw background block for label text
        try:
            if hasattr(draw, "textbbox") and font:
                tx0, ty0, tx1, ty1 = draw.textbbox((x0, y0), text, font=font)
                tw = tx1 - tx0
                th = ty1 - ty0
            else:
                tw, th = draw.textsize(text) if hasattr(draw, "textsize") else (len(text) * 6, 10)
        except Exception:
            tw, th = len(text) * 6, 10
            
        # Draw text background box
        draw.rectangle([x0, max(0, y0 - th - 4), x0 + tw + 6, y0], fill=color)
        
        # Draw label text in white
        draw.text((x0 + 3, max(0, y0 - th - 2)), text, fill=(255, 255, 255), font=font)
        
    return img_copy

def create_side_by_side(
    image: Image.Image, 
    ground_truths: List[Dict[str, Any]], 
    predictions: List[Dict[str, Any]]
) -> Image.Image:
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
        
    gt_panel = draw_boxes(image, ground_truths, is_prediction=False)
    pred_panel = draw_boxes(image, predictions, is_prediction=True)
    
    w, h = image.size
    header_h = 35
    
    # Concatenate side-by-side, adding a top header block
    combined = Image.new("RGB", (w * 2, h + header_h), color=(30, 30, 30))
    
    # Paste panels below the header
    combined.paste(gt_panel, (0, header_h))
    combined.paste(pred_panel, (w, header_h))
    
    # Draw headers text
    draw = ImageDraw.Draw(combined)
    
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
        
    # Draw Left panel header text: "GROUND TRUTH"
    draw.text((20, 10), "GROUND TRUTH (LEFT)", fill=(46, 204, 113), font=font)
    
    # Draw Right panel header text: "MODEL PREDICTIONS"
    draw.text((w + 20, 10), "MODEL PREDICTIONS (RIGHT)", fill=(231, 76, 60), font=font)
    
    # Draw divider lines
    draw.line([(w, 0), (w, h + header_h)], fill=(0, 0, 0), width=4)
    draw.line([(0, header_h), (w * 2, header_h)], fill=(0, 0, 0), width=2)
    
    return combined

def save_visualization(image: Image.Image, output_dir: str, file_name: str) -> None:
    """Saves the composite visualization image to the target directory."""
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(file_name)
    save_path = os.path.join(output_dir, base_name)
    image.save(save_path)
