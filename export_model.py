"""
Export YOLO-NAS model to ONNX format for deployment.

Usage:
    python export_model.py --checkpoint checkpoints/yolo_nas_m/RUN_20251125_160616_710583/ckpt_best.pth
"""
import argparse
import os
import torch
from super_gradients.training import models
from super_gradients.conversion import ExportTargetBackend

# Classes from training
CLASSES = ['macula', 'optic-disc']


def export_to_onnx(checkpoint_path: str, output_path: str = None, img_size: int = 640):
    """
    Export the YOLO-NAS model to ONNX format.

    Args:
        checkpoint_path: Path to the .pth checkpoint file
        output_path: Path for the output ONNX file (default: same dir as checkpoint)
        img_size: Input image size (default: 640)
    """
    if output_path is None:
        output_path = checkpoint_path.replace('.pth', '.onnx')

    print(f"Loading model from: {checkpoint_path}")

    # Load the model with trained weights
    model = models.get(
        'yolo_nas_m',
        num_classes=len(CLASSES),
        checkpoint_path=checkpoint_path
    )

    # Prepare model for export
    model.eval()
    model.prep_model_for_conversion(input_size=[1, 3, img_size, img_size])

    print(f"Exporting to ONNX: {output_path}")

    # Export using SuperGradients built-in export
    model.export(
        output_path,
        batch_size=1,
        input_image_shape=[img_size, img_size]
    )

    print(f"Model exported successfully to: {output_path}")
    print(f"\nModel info:")
    print(f"  - Input size: {img_size}x{img_size}")
    print(f"  - Number of classes: {len(CLASSES)}")
    print(f"  - Classes: {CLASSES}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Export YOLO-NAS model to ONNX')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/yolo_nas_m/RUN_20251125_160616_710583/ckpt_best.pth',
        help='Path to the checkpoint file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output ONNX file path (default: same as checkpoint with .onnx extension)'
    )
    parser.add_argument(
        '--img-size',
        type=int,
        default=640,
        help='Input image size (default: 640)'
    )

    args = parser.parse_args()

    # Get absolute path
    if not os.path.isabs(args.checkpoint):
        args.checkpoint = os.path.join(os.path.dirname(__file__), args.checkpoint)

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    export_to_onnx(args.checkpoint, args.output, args.img_size)


if __name__ == '__main__':
    main()
