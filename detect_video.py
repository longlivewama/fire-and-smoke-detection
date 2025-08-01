import cv2
from ultralytics import YOLO
import argparse

def detect_on_video(input_path, output_path, model_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("❌ Error opening video file.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("🚀 Running YOLO detection...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inference and annotate
        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()
    print(f"✅ Done! Saved at: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", default="output.mp4", help="Output video path")
    parser.add_argument("--model", default="models/best.pt", help="Model path")
    args = parser.parse_args()
    
    detect_on_video(args.input, args.output, args.model)
