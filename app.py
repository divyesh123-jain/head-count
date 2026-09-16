import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
from sklearn.cluster import DBSCAN

# Load models - using YOLOv8x for maximum accuracy in crowded scenes
print("Loading yolov8x.pt... (this may take a moment)")
model = YOLO('yolov8x.pt')
print("Model loaded successfully.")

def sharpen_image(image):
    """Apply adaptive sharpening to reduce blur effects"""
    if image is None:
        return None

    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])

    # Convert back to RGB
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Apply unsharp masking
    gaussian = cv2.GaussianBlur(enhanced, (9,9), 10.0)
    sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)

    return sharpened

def detect_people_multi_scale(image, model):
    """
    Detect people using multi-scale approach for better coverage
    Returns list of bounding boxes [x1, y1, x2, y2]
    """
    all_boxes = []

    # Original image
    h, w = image.shape[:2]

    # Scale 1: Original size
    results1 = model(image, classes=[0], imgsz=1280, conf=0.20, iou=0.45, verbose=False)
    boxes1 = results1[0].boxes.xyxy.cpu().numpy() if len(results1[0].boxes) > 0 else np.empty((0, 4))
    for box in boxes1:
        all_boxes.append(box)

    # Scale 2: Slightly zoomed out (detect smaller people)
    scale2 = 0.7
    new_w2 = int(w * scale2)
    new_h2 = int(h * scale2)
    img_scaled2 = cv2.resize(image, (new_w2, new_h2), interpolation=cv2.INTER_AREA)
    results2 = model(img_scaled2, classes=[0], imgsz=1280, conf=0.20, iou=0.45, verbose=False)
    boxes2 = results2[0].boxes.xyxy.cpu().numpy() if len(results2[0].boxes) > 0 else np.empty((0, 4))
    # Scale boxes back to original size
    if len(boxes2) > 0:
        boxes2[:, [0, 2]] *= (w / new_w2)
        boxes2[:, [1, 3]] *= (h / new_h2)
        for box in boxes2:
            all_boxes.append(box)

    # Scale 3: Slightly zoomed in (detect people at edges)
    scale3 = 1.3
    new_w3 = int(w * scale3)
    new_h3 = int(h * scale3)
    img_scaled3 = cv2.resize(image, (new_w3, new_h3), interpolation=cv2.INTER_CUBIC)
    results3 = model(img_scaled3, classes=[0], imgsz=1280, conf=0.20, iou=0.45, verbose=False)
    boxes3 = results3[0].boxes.xyxy.cpu().numpy() if len(results3[0].boxes) > 0 else np.empty((0, 4))
    # Scale boxes back to original size
    if len(boxes3) > 0:
        boxes3[:, [0, 2]] *= (w / new_w3)
        boxes3[:, [1, 3]] *= (h / new_h3)
        for box in boxes3:
            all_boxes.append(box)

    return all_boxes

def merge_duplicate_boxes(boxes, eps=30, min_samples=1):
    """
    Use DBSCAN clustering to merge duplicate detections from multi-scale approach
    """
    if len(boxes) == 0:
        return []

    # Convert boxes to centers for clustering
    centers = []
    for box in boxes:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        centers.append([cx, cy])

    centers = np.array(centers)

    # Apply DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(centers)
    labels = db.labels_

    # Get unique clusters
    unique_labels = set(labels)
    merged_boxes = []

    for label in unique_labels:
        if label == -1:  # Noise
            continue

        # Get all boxes in this cluster
        cluster_indices = np.where(labels == label)[0]
        cluster_boxes = boxes[cluster_indices]

        # Merge by taking average
        merged_box = np.mean(cluster_boxes, axis=0)
        merged_boxes.append(merged_box)

    return merged_boxes

def count_heads(image):
    """
    Detect people in the image and count heads with advanced optimizations.
    Returns the annotated image with bounding boxes and the count.
    """
    if image is None:
        return None, 0

    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Pre-process: sharpen image to reduce blur
    print("Pre-processing image (sharpening)...")
    image_enhanced = sharpen_image(image)

    # Run multi-scale detection
    print("Running multi-scale detection...")
    all_boxes = detect_people_multi_scale(image_enhanced, model)

    # Merge duplicate detections
    print("Merging duplicate detections...")
    final_boxes = merge_duplicate_boxes(np.array(all_boxes), eps=30, min_samples=1)

    final_count = len(final_boxes)

    # Create annotated image
    annotated_image = image.copy()

    # Draw bounding boxes
    for i, box in enumerate(final_boxes):
        x1, y1, x2, y2 = map(int, box)

        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(annotated_image.shape[1], x2)
        y2 = min(annotated_image.shape[0], y2)

        # Draw box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add label with count
        label = f"Person {i+1}"
        cv2.putText(annotated_image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Add total count display
    h, w = annotated_image.shape[:2]

    # Create semi-transparent overlay
    overlay = annotated_image.copy()
    cv2.rectangle(overlay, (10, 10), (280, 65), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, annotated_image, 0.3, 0, annotated_image)

    # Add count text
    cv2.putText(annotated_image, f'Total Count: {final_count}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    return annotated_image, final_count

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 👥 AI Head Counter - Advanced Crowd Detection

    Upload an image containing people, and this AI application will detect and count all the heads/people in the image.

    **Advanced Features:**
    - Multi-scale detection for comprehensive coverage
    - Blur reduction and image enhancement
    - Duplicate detection removal using DBSCAN clustering
    - Optimized for crowded scenes (100-1000+ people)
    - Works with various lighting conditions and angles
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Upload Image", type="numpy", height=400)
            count_btn = gr.Button("🔍 Count Heads", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_image = gr.Image(label="Detection Result", type="numpy", height=400)

    with gr.Row():
        count_output = gr.Number(label="Total People Counted", precision=0)

    gr.Markdown("""
    ### How it works:
    1. Upload an image with people in it
    2. Click "Count Heads" button
    3. The AI will:
       - Enhance image quality (reduce blur)
       - Run detection at multiple scales
       - Merge duplicate detections intelligently
    4. Total count will be displayed with visual bounding boxes

    **Note:** This uses advanced computer vision to detect people. Accuracy may vary based on image quality, severe occlusions, and extreme viewing angles.
    """)

    # Connect the button to the function
    count_btn.click(
        fn=count_heads,
        inputs=input_image,
        outputs=[output_image, count_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
