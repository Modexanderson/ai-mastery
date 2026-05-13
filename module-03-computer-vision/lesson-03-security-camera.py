"""
Module 03 -- Lesson 3: Real-Time Security Camera
==================================================
We combine everything from Lessons 1 & 2 into a practical project:
a security camera system that:

  1. Captures live video from your webcam
  2. Runs YOLO object detection on every frame
  3. Detects people specifically
  4. Draws bounding boxes and labels
  5. Counts people in frame
  6. Logs detection events with timestamps
  7. Saves a screenshot when a person is first detected
  8. Shows an alert when the scene changes (someone enters/leaves)

This is a real product you could sell -- smart security cameras
use exactly this pipeline, just with better models and hardware.

Controls:
  q     - quit
  s     - save screenshot manually
  r     - reset person counter
  space - pause/unpause

Model: YOLOv8n (nano) for speed on webcam
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime
from ultralytics import YOLO

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
SCREENSHOTS_DIR = os.path.join(OUTPUT_DIR, "screenshots")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)


def draw_dashboard(frame, stats):
    """Draw a semi-transparent info panel on the frame."""
    h, w = frame.shape[:2]

    # Semi-transparent black bar at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Title
    cv2.putText(frame, "SECURITY CAMERA", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    # Timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (w - 250, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Stats row
    people = stats.get("people", 0)
    total = stats.get("total_objects", 0)
    fps = stats.get("fps", 0)
    alerts = stats.get("alerts", 0)

    # People count (green if 0, red if detected)
    color = (0, 255, 0) if people == 0 else (0, 0, 255)
    cv2.putText(frame, f"People: {people}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(frame, f"Objects: {total}", (180, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.putText(frame, f"FPS: {fps:.0f}", (350, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.putText(frame, f"Alerts: {alerts}", (470, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

    # Status
    status = "PERSON DETECTED" if people > 0 else "ALL CLEAR"
    status_color = (0, 0, 255) if people > 0 else (0, 255, 0)
    cv2.putText(frame, status, (10, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

    # Controls help at bottom
    cv2.putText(frame, "q:quit  s:screenshot  r:reset  space:pause", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

    return frame


def save_screenshot(frame, reason="manual"):
    """Save a timestamped screenshot."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"alert_{reason}_{timestamp}.png"
    path = os.path.join(SCREENSHOTS_DIR, filename)
    cv2.imwrite(path, frame)
    print(f"  [SCREENSHOT] {filename}")
    return filename


def main():
    print("=" * 60)
    print("  MODULE 03 -- LESSON 3: SECURITY CAMERA SYSTEM")
    print("=" * 60)

    # Load YOLO model
    print("\n  Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")
    print("  Model loaded.")

    # Open webcam
    print("  Opening webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("  ERROR: Could not open webcam!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("  Webcam ready. Starting security monitor...\n")
    print("  Controls: q=quit, s=screenshot, r=reset, space=pause\n")

    # State tracking
    prev_people_count = 0
    total_alerts = 0
    paused = False
    frame_count = 0
    fps = 0
    fps_start_time = time.time()

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO detection
            results = model(frame, verbose=False, conf=0.4)
            result = results[0]

            # Count detections by class
            people_count = 0
            total_objects = len(result.boxes)

            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                if class_name == "person":
                    people_count += 1
                    # Draw person boxes in red (more prominent)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"PERSON {confidence:.0%}"
                    cv2.putText(frame, label, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    # Other objects in green
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    label = f"{class_name} {confidence:.0%}"
                    cv2.putText(frame, label, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Detect scene changes (someone entered or left)
            if people_count > 0 and prev_people_count == 0:
                total_alerts += 1
                print(f"  [ALERT #{total_alerts}] Person entered frame! ({people_count} detected)")
                save_screenshot(frame, "person_entered")

            elif people_count == 0 and prev_people_count > 0:
                print(f"  [INFO] Scene cleared - no people detected")

            elif people_count != prev_people_count and people_count > 0:
                print(f"  [INFO] People count changed: {prev_people_count} -> {people_count}")

            prev_people_count = people_count

            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = time.time()

            # Draw the dashboard overlay
            stats = {
                "people": people_count,
                "total_objects": total_objects,
                "fps": fps,
                "alerts": total_alerts,
            }
            frame = draw_dashboard(frame, stats)

            # Display
            cv2.imshow("Security Camera", frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            save_screenshot(frame, "manual")
        elif key == ord('r'):
            total_alerts = 0
            print("  [RESET] Alert counter reset")
        elif key == ord(' '):
            paused = not paused
            print(f"  [{'PAUSED' if paused else 'RESUMED'}]")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    print(f"\n  Session ended.")
    print(f"  Total alerts: {total_alerts}")
    print(f"  Screenshots saved to: {SCREENSHOTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
