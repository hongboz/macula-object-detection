"""
Macula & Optic Disc Detector - ONNX Version
Simple inference using ONNX Runtime (no super-gradients needed)
"""
import gradio as gr
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image

# Class names from training
CLASSES = ['macula', 'optic-disc']
COLORS = [(255, 200, 0), (255, 100, 0)]  # Yellow, Orange

# Available models
MODELS = {
    "640x640 (higher detail)": ("ckpt_best.onnx", 640),
    "224x224 (training size)": ("ckpt_best_224.onnx", 224),
    "Color model": ("ckpt_best_color.onnx", 640),
}

# Load both models at startup
sessions = {}
print("Loading ONNX models...")
for name, (path, size) in MODELS.items():
    try:
        sessions[name] = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        print(f"  Loaded: {name}")
    except Exception as e:
        print(f"  Could not load {name}: {e}")

# Default model
current_model = list(MODELS.keys())[0] if sessions else None
print(f"Default model: {current_model}")


def preprocess(image, img_size=640):
    """Preprocess image for YOLO-NAS ONNX model."""
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)

    original_shape = image.shape[:2]  # (height, width)

    # Resize
    resized = cv2.resize(image, (img_size, img_size))

    # Convert grayscale to RGB if needed
    if len(resized.shape) == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

    # Keep as uint8 (model expects 0-255), transpose to NCHW
    blob = resized.astype(np.uint8)
    blob = blob.transpose(2, 0, 1)  # HWC -> CHW
    blob = np.expand_dims(blob, 0)  # Add batch dimension

    return blob, original_shape


def postprocess(outputs, original_shape, img_size=640, conf_threshold=0.5):
    """Parse ONNX model outputs to bounding boxes."""
    # SuperGradients YOLO-NAS format:
    # outputs[0]: (1, 1) - num_predictions
    # outputs[1]: (1, 1000, 4) - bboxes [x1, y1, x2, y2]
    # outputs[2]: (1, 1000) - confidence scores
    # outputs[3]: (1, 1000) - class IDs

    boxes = []
    scale_x = original_shape[1] / img_size
    scale_y = original_shape[0] / img_size

    num_preds = int(outputs[0][0, 0])
    bboxes = outputs[1][0]      # (1000, 4)
    scores = outputs[2][0]      # (1000,)
    class_ids = outputs[3][0]   # (1000,)

    for i in range(num_preds):
        conf = float(scores[i])
        if conf >= conf_threshold:
            x1, y1, x2, y2 = bboxes[i]
            class_id = int(class_ids[i])

            boxes.append({
                'x1': int(x1 * scale_x),
                'y1': int(y1 * scale_y),
                'x2': int(x2 * scale_x),
                'y2': int(y2 * scale_y),
                'confidence': conf,
                'class_id': class_id,
                'class_name': CLASSES[class_id] if class_id < len(CLASSES) else f'class_{class_id}'
            })

    return boxes


def draw_boxes(image, boxes):
    """Draw bounding boxes on image."""
    image = image.copy()

    for box in boxes:
        color = COLORS[box['class_id'] % len(COLORS)]
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{box['class_name']}: {box['confidence']:.1%}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return image


def detect(image, confidence_threshold, model_choice):
    """Run detection on image."""
    if image is None:
        return None, "No image uploaded"

    if model_choice not in sessions:
        return None, f"Model {model_choice} not loaded"

    # Get model and size
    session = sessions[model_choice]
    img_size = MODELS[model_choice][1]
    input_name = session.get_inputs()[0].name

    # Preprocess
    blob, original_shape = preprocess(image, img_size)

    # Run inference
    outputs = session.run(None, {input_name: blob})

    # Postprocess
    boxes = postprocess(outputs, original_shape, img_size, confidence_threshold)

    # Draw results
    result_image = draw_boxes(image, boxes)

    # Build summary
    if len(boxes) == 0:
        summary = "No objects detected at this confidence level."
    else:
        summary = f"**Detected {len(boxes)} object(s):**\n\n"
        for box in boxes:
            summary += f"- **{box['class_name']}**: {box['confidence']:.1%} confidence\n"

    return result_image, summary


# Create Gradio interface
with gr.Blocks(title="Macula & Optic Disc Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🔬 Macula & Optic Disc Detector

        Upload a retinal fundus image to detect the **macula** and **optic disc**.
        """
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", type="numpy")
            model_selector = gr.Dropdown(
                choices=list(MODELS.keys()),
                value=list(MODELS.keys())[0],
                label="Model Size"
            )
            confidence_slider = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.5,
                step=0.05,
                label="Confidence Threshold"
            )
            detect_btn = gr.Button("🔍 Detect", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Detection Result")
            output_text = gr.Markdown(label="Detection Details")

    detect_btn.click(
        fn=detect,
        inputs=[input_image, confidence_slider, model_selector],
        outputs=[output_image, output_text]
    )

    gr.Markdown("---")
    gr.Markdown(
        """
        ### About
        This model detects anatomical structures in retinal fundus images using **YOLO-NAS**.

        **Classes:** 🟡 Macula | 🟠 Optic Disc
        """
    )

if __name__ == "__main__":
    demo.launch()
