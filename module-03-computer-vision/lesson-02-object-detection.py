"""
Module 03 -- Lesson 2: Object Detection with YOLO
===================================================
YOLO (You Only Look Once) is a neural network that looks at an image and
tells you WHAT objects are in it and WHERE they are.

How it works (simplified):
  1. Image goes in as a grid of pixels (just like Lesson 1)
  2. The neural network (millions of trained weights) processes it
  3. Output: a list of detections, each with:
     - Class name (person, car, dog, etc.)
     - Confidence score (0.0 to 1.0 -- how sure the model is)
     - Bounding box (x, y, width, height -- where in the image)

YOLOv8 was trained on the COCO dataset: 80 object categories including
person, car, dog, cat, chair, bottle, phone, laptop, etc.

The model weights file (yolov8n.pt) is ~6MB -- "n" means "nano" (smallest).
Sizes: nano < small < medium < large < extra-large (bigger = more accurate, slower)

Library: ultralytics -- the official YOLO Python package

What we'll do:
  1. Run detection on a generated test image
  2. Run detection on your webcam (real-time)
  3. Show how to filter by class and confidence
"""

import cv2
import numpy as np
import os
from ultralytics import YOLO

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def save_img(name: str, image: np.ndarray):
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    cv2.imwrite(path, image)
    print(f"  Saved: {name}.png")


# ==============================================================================
# PART 1: Run YOLO on a test image
# ==============================================================================

def part1_detect_objects():
    print("=" * 60)
    print("  PART 1: Object Detection on an Image")
    print("=" * 60)

    # Load the YOLO model -- downloads weights automatically on first run (~6MB)
    # yolov8n = nano (fastest, least accurate)
    # yolov8s, yolov8m, yolov8l, yolov8x = progressively bigger/better
    print("\n  Loading YOLOv8 nano model...")
    model = YOLO("yolov8n.pt")

    # Create a test image with objects YOLO can detect
    # We'll use a real photo from your webcam in Part 2
    # For now, let's grab a sample image from the internet
    print("  Running detection on a sample image...")

    # Use a built-in ultralytics test -- detect on a bus image
    results = model("https://ultralytics.com/images/bus.jpg", verbose=False)

    # Process results
    result = results[0]  # first (only) image

    print(f"\n  Detections found: {len(result.boxes)}")
    print(f"  Image size: {result.orig_shape}")
    print(f"  Model: {result.speed}")
    print()

    # Loop through each detection
    print("  {:<12} {:<10} {:<20}".format("Class", "Conf", "Bounding Box"))
    print("  " + "-" * 45)

    for box in result.boxes:
        # Class name
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        # Confidence (0.0 to 1.0)
        confidence = float(box.conf[0])

        # Bounding box coordinates [x1, y1, x2, y2]
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        print(f"  {class_name:<12} {confidence:.2f}       ({int(x1)},{int(y1)}) -> ({int(x2)},{int(y2)})")

    # Save the annotated image (YOLO draws boxes automatically)
    annotated = result.plot()  # returns image with boxes drawn
    save_img("05_yolo_bus", annotated)

    print()
    return model


# ==============================================================================
# PART 2: Real-time webcam detection
# ==============================================================================

def part2_webcam(model: YOLO):
    print("=" * 60)
    print("  PART 2: Real-Time Webcam Object Detection")
    print("=" * 60)
    print("\n  Opening webcam... (press 'q' to quit)")
    print("  Point your camera at objects: people, phones, bottles, etc.\n")

    cap = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        print("  ERROR: Could not open webcam. Skipping Part 2.")
        return

    # Set resolution (lower = faster processing)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0
    saved_one = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO on this frame
        results = model(frame, verbose=False, conf=0.4)  # conf=0.4 filters low-confidence
        annotated = results[0].plot()

        # Show detection count on screen
        det_count = len(results[0].boxes)
        cv2.putText(annotated, f"Objects: {det_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated, "Press 'q' to quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Display the frame
        cv2.imshow("YOLO Real-Time Detection", annotated)

        # Save one frame as an example
        if det_count > 0 and not saved_one:
            save_img("05_yolo_webcam", annotated)
            saved_one = True

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n  Processed {frame_count} frames.")


# ==============================================================================
# PART 3: Filtering detections
# ==============================================================================

def part3_filtering(model: YOLO):
    print("=" * 60)
    print("  PART 3: Filtering Detections")
    print("=" * 60)

    print("\n  Running detection on bus image again with filters...\n")

    results = model("https://ultralytics.com/images/bus.jpg", verbose=False)
    result = results[0]

    # Filter: only show "person" detections with confidence > 0.5
    print("  FILTER: Only 'person' with confidence > 0.5:")
    print("  " + "-" * 40)

    person_count = 0
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        if class_name == "person" and confidence > 0.5:
            person_count += 1
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(f"  Person #{person_count}: conf={confidence:.2f}, "
                  f"pos=({int(x1)},{int(y1)})->({int(x2)},{int(y2)})")

    print(f"\n  Total people found: {person_count}")
    print(f"  Total objects (all classes): {len(result.boxes)}")

    # Show all 80 classes the model can detect
    print(f"\n  All {len(model.names)} classes YOLO can detect:")
    classes = list(model.names.values())
    for i in range(0, len(classes), 8):
        row = classes[i:i+8]
        print(f"    {', '.join(row)}")

    print()


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("\n" + "=" * 60)
    print("  MODULE 03 -- LESSON 2: OBJECT DETECTION WITH YOLO")
    print("=" * 60 + "\n")

    model = part1_detect_objects()
    part3_filtering(model)

    # Ask user if they want to try webcam
    try:
        choice = input("  Try real-time webcam detection? (y/n): ").strip().lower()
        if choice == 'y':
            part2_webcam(model)
    except (KeyboardInterrupt, EOFError):
        pass

    print("=" * 60)
    print("  Done! Check the saved images in module-03-computer-vision/")
    print("=" * 60)


if __name__ == "__main__":
    main()
