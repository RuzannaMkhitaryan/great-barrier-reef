"""
Inference script for Crown-of-Thorns Starfish detection.

Loads the trained YOLO model, runs inference on a user-provided image,
saves a visualization of the detections, and records detected object
coordinates and confidence scores.
"""
import argparse
import os
from ultralytics import YOLO

MODEL_PATH = "outputs/final.pt"
# Load the trained YOLO model once when the script starts.
model = YOLO(MODEL_PATH)

def run_prediction(image_path, conf_thresh=0.25):
    """
    Run YOLO inference on an image and return detected object information.
    Args:
        image_path: Path to the input image.
        conf_thresh: Minimum confidence threshold for detections.
    Returns:
        List of detected objects with coordinates and confidence scores.
    """
    results = model(image_path, conf=conf_thresh)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_name = f"visual_result_{base_name}.jpg"
    # Save an image with predicted bounding boxes.
    results[0].save(filename=output_name)
    print(f"Visualized prediction saved as '{output_name}'")

    # Store center coordinates and confidence for each detected COTS.
    predictions = []
    for r in results:
        for box in r.boxes:
            x_center, y_center, w, h = box.xywh[0].tolist()
            conf = box.conf[0].item()
            predictions.append({
                "x": round(x_center, 1),
                "y": round(y_center, 1),
                "confidence": round(conf, 3)
            })
    return predictions


def format_output(predictions):
    """Print prediction results in a readable format."""
    if not predictions:
        print("No objects detected.")
    else:
        print(f"Found {len(predictions)} objects.")
        for i, pred in enumerate(predictions, start=1):
            print(
                f"  {i}. x={pred['x']}, y={pred['y']}, confidence={pred['confidence']}"
            )


def save_output(predictions, image_path):
    """Append prediction results to a local text log."""
    # Keep prediction logs locally; prediction_results.txt is ignored by Git.
    output_filename = "prediction_results.txt"
    with open(output_filename, "a") as f:
        f.write(f"Image: {image_path}\n")
        for pred in predictions:
            f.write(
                f"  x={pred['x']}, y={pred['y']}, confidence={pred['confidence']}\n"
            )
        f.write("\n")


def main():
    """Parse command-line arguments and run inference."""
    parser = argparse.ArgumentParser(description="Predict on an image")
    parser.add_argument(
        "--image", type=str, required=True, help="Path to the image file"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.1,
        help="Confidence threshold (e.g. 0.05, 0.1, 0.25)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: file wasn't found in the path {args.image}")
        return

    predictions = run_prediction(args.image, conf_thresh=args.conf)
    format_output(predictions)
    save_output(predictions, args.image)


if __name__ == "__main__":
    main()
