# import argparse          # reading args from command line 
# import cv2                # computer vision library, used here to read images
# import os            
# from ultralytics import YOLO


# MODEL_PATH = "outputs/final.pt"
# model = YOLO(MODEL_PATH)


# def preprocess_image(image):
#     """
#     PLACEHOLDER. Later, this section will contain the code
#     that converts the image into the format expected by the model 
#     (e.g., resizing, color normalization, etc. 
#     — B will specify these requirements when they provide the architecture).
#     For now, we simply return the image as is.
#     """
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     return image_rgb


# def run_prediction(image, conf_thresh=0.25):
#     """
#     this section contains the code
#     that loads the model weights and
#     obtains the actual prediction (star coordinates)
#     """
#     results = model(image, conf=conf_thresh)

#     #later
#     results[0].save(filename="visual_result.jpg")
#     print("Visualized prediction saved as 'visual_result.jpg'")


#     predictions = []
#     for r in results:
#         for box in r.boxes:
#             x_center, y_center, w, h = box.xywh[0].tolist()
#             conf = box.conf[0].item()
#             predictions.append({
#                 "x": round(x_center, 1),
#                 "y": round(y_center, 1),
#                 "confidence": round(conf, 3)
#             })
#     return predictions


# def format_output(predictions):
#     """
#     This section contains the code
#     that formats the prediction results for display.
#     """
#     if not predictions:
#         print("No starfish detected.")
#     else:
#         print(f"Found {len(predictions)} starfish.")
#         for i, pred in enumerate(predictions, start=1):
#             print(f"  {i}. x={pred['x']}, y={pred['y']}, confidence={pred['confidence']}")


# def save_output(predictions, image_path):
#     """
#     Saving result in txt file near the script
#     We will use this to detect errors (Pillar 5) — will be the history 
#     of model using on dif images
#     """
#     output_filename = "prediction_results.txt"
#     with open(output_filename, "a") as f:
#         f.write(f"Image: {image_path}\n")
#         for pred in predictions:
#             f.write(f"  x={pred['x']}, y={pred['y']}, confidence={pred['confidence']}\n")
#         f.write("\n")
#     print(f"result also saved in {output_filename}")


# def main():
#     # This section handles command-line arguments. The user can specify the image file to process
#     # by providing it as an argument when running the script, e.g., python predict.py --image foto.jpg
#     parser = argparse.ArgumentParser(description="Predict starfish on an image")
#     parser.add_argument("--image", type=str, required=True, help="Path to the image file")

#     # Добавили параметр порога уверенности (по умолчанию 0.25)
#     parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (e.g. 0.1, 0.25)")

#     args = parser.parse_args()

#     #checking if file excisits 
#     if not os.path.exists(args.image):
#         print(f"Error: file wasnt find in the path {args.image}")
#         return

#     # Read the image from the specified file path using OpenCV. If the image cannot be opened (e.g., if the file does not exist or is not a valid image), an error message is printed and the program exits.
#     image = cv2.imread(args.image)

#     if image is None:
#         print(f"Error: Could not open file {args.image}")
#         return

#     processed = preprocess_image(image)
#     predictions = run_prediction(processed)
#     format_output(predictions)
#     save_output(predictions, args.image)



# if __name__ == "__main__":
#     main()


import argparse
import os
from ultralytics import YOLO

MODEL_PATH = "outputs/final.pt"
model = YOLO(MODEL_PATH)

print("КЛАССЫ МОДЕЛИ:", model.names)

def run_prediction(image_path, conf_thresh=0.25):
    results = model(image_path, conf=conf_thresh)

    # уникальное имя на основе исходного файла, чтобы не перезаписывать предыдущие результаты
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_name = f"visual_result_{base_name}.jpg"
    results[0].save(filename=output_name)
    print(f"Visualized prediction saved as '{output_name}'")

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
    if not predictions:
        print("No objects detected.")
    else:
        print(f"Found {len(predictions)} objects.")
        for i, pred in enumerate(predictions, start=1):
            print(
                f"  {i}. x={pred['x']}, y={pred['y']}, confidence={pred['confidence']}"
            )


def save_output(predictions, image_path):
    output_filename = "prediction_results.txt"
    with open(output_filename, "a") as f:
        f.write(f"Image: {image_path}\n")
        for pred in predictions:
            f.write(
                f"  x={pred['x']}, y={pred['y']}, confidence={pred['confidence']}\n"
            )
        f.write("\n")


def main():
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

    # Передаем путь к файлу напрямую
    predictions = run_prediction(args.image, conf_thresh=args.conf)
    format_output(predictions)
    save_output(predictions, args.image)


if __name__ == "__main__":
    main()