import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os

# Load the YOLO model for person detection
# Using YOLOv8n - the smallest and fastest model
model = YOLO('yolov8n.pt')

def count_heads(image):
    """
    Detect people in the image and count heads.
    Returns the annotated image with bounding boxes and the count.
    """
    if image is None:
        return None, 0
    
    # Convert to RGB if needed (Gradio loads as RGB)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Run inference on the image
    results = model(image, classes=[0])  # class 0 is 'person' in COCO dataset
    
    # Get the number of detected people
    count = len(results[0].boxes)
    
    # Create annotated image with bounding boxes and labels
    annotated_image = results[0].plot()
    
    # Add a text box showing total count at the top
    h, w = annotated_image.shape[:2]
    
    # Create a semi-transparent overlay for the count display
    overlay = annotated_image.copy()
    
    # Draw rectangle for count display
    cv2.rectangle(overlay, (10, 10), (250, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, annotated_image, 0.3, 0, annotated_image)
    
    # Add count text
    cv2.putText(annotated_image, f'Total Count: {count}', (20, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    # Convert BGR to RGB for Gradio display
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    return annotated_image_rgb, count

# Create Gradio interface
with gr.Blocks(title="AI Head Counter") as demo:
    gr.Markdown("""
    # 👥 AI Head Counter
    
    Upload an image containing people, and this AI application will detect and count all the heads/people in the image.
    
    **Features:**
    - Automatic person detection using state-of-the-art YOLOv8
    - Visual bounding boxes around each detected person
    - Total count displayed prominently
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
    
    # Example images section
    gr.Examples(
        examples=[
            "https://images.unsplash.com/photo-1511632765486-a01980e01a18?w=800",
            "https://images.unsplash.com/photo-1528605248644-14dd04022da1?w=800",
            "https://images.unsplash.com/photo-1529156069898-49953e39b3ac?w=800",
        ],
        inputs=input_image,
        outputs=[output_image, count_output],
        fn=count_heads,
        cache_examples=False,
    )
    
    gr.Markdown("""
    ### How it works:
    1. Upload an image with people in it
    2. Click "Count Heads" button
    3. The AI will detect all people and draw bounding boxes
    4. Total count will be displayed
    
    **Note:** This uses computer vision to detect people. Accuracy may vary based on image quality, lighting, occlusions, and viewing angles.
    """)
    
    # Connect the button to the function
    count_btn.click(
        fn=count_heads,
        inputs=input_image,
        outputs=[output_image, count_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
