"""
Module 03 — Lesson 1: Image Basics & Processing
=================================================
KEY INSIGHT: An image is just a 3D NumPy array of shape (height, width, 3).
Every computer vision operation — filters, detection, recognition — is just
math performed on this array. Same idea as the neural net in Module 00.

What we'll do:
  1. Create images from scratch using NumPy (pixels are just numbers)
  2. Load a real image and inspect its data
  3. Apply transformations: grayscale, blur, edge detection, thresholding
  4. Draw on images (boxes, text) — this is how detection results are displayed

Library: OpenCV (cv2) — the industry standard for computer vision.
  - Used in self-driving cars, security cameras, medical imaging, robotics
  - Written in C++ for speed, Python bindings for ease of use

NOTE: OpenCV uses BGR (Blue-Green-Red) not RGB. Historical quirk.
"""

import cv2
import numpy as np
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def save_and_show(name: str, image: np.ndarray):
    """Save image to file and print its info."""
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    cv2.imwrite(path, image)
    h, w = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    print(f"  Saved: {name}.png  ({w}x{h}, {channels}ch, dtype={image.dtype})")


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: Images are just number arrays
# ══════════════════════════════════════════════════════════════════════════════

def part1_images_are_arrays():
    print("=" * 60)
    print("  PART 1: Images Are Just Number Arrays")
    print("=" * 60)

    # Create a blank black image: 200 rows, 300 columns, 3 color channels
    # np.zeros = fill with 0s. All zeros = black (no red, no green, no blue)
    black = np.zeros((200, 300, 3), dtype=np.uint8)
    print(f"\n  Black image shape: {black.shape}")
    print(f"  That means: {black.shape[0]} rows x {black.shape[1]} cols x {black.shape[2]} channels")
    print(f"  Total pixels: {black.shape[0] * black.shape[1]:,}")
    print(f"  dtype: {black.dtype} (unsigned 8-bit int = 0 to 255)")
    save_and_show("01_black", black)

    # Paint it red — OpenCV uses BGR, so red = [0, 0, 255]
    red = black.copy()
    red[:] = [0, 0, 255]      # set ALL pixels to blue=0, green=0, red=255
    save_and_show("01_red", red)

    # Create a gradient — values go from 0 (black) to 255 (white)
    gradient = np.zeros((200, 300), dtype=np.uint8)
    for col in range(300):
        gradient[:, col] = int(col / 300 * 255)  # left=dark, right=bright
    save_and_show("01_gradient", gradient)

    # Create colored stripes — showing RGB channels
    stripes = np.zeros((200, 300, 3), dtype=np.uint8)
    stripes[:, :100]   = [255, 0, 0]     # left third = blue (BGR!)
    stripes[:, 100:200] = [0, 255, 0]     # middle = green
    stripes[:, 200:]    = [0, 0, 255]     # right = red
    save_and_show("01_stripes", stripes)

    # Draw shapes — this is how object detection results are displayed
    canvas = np.zeros((300, 400, 3), dtype=np.uint8)

    # Rectangle: (image, top-left, bottom-right, color_BGR, thickness)
    cv2.rectangle(canvas, (20, 20), (180, 130), (0, 255, 0), 2)

    # Circle: (image, center, radius, color_BGR, thickness)  -1 = filled
    cv2.circle(canvas, (300, 150), 60, (0, 0, 255), -1)

    # Line: (image, start, end, color_BGR, thickness)
    cv2.line(canvas, (20, 250), (380, 250), (255, 255, 0), 3)

    # Text: (image, text, position, font, scale, color_BGR, thickness)
    cv2.putText(canvas, "OpenCV", (100, 280), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2)

    save_and_show("01_shapes", canvas)
    print()


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: Load and inspect a real image
# ══════════════════════════════════════════════════════════════════════════════

def part2_load_image():
    print("=" * 60)
    print("  PART 2: Create & Inspect a Sample Image")
    print("=" * 60)

    # Create a sample image with interesting features for processing
    img = np.zeros((400, 600, 3), dtype=np.uint8)

    # Background gradient
    for row in range(400):
        val = int(row / 400 * 100) + 50
        img[row, :] = [val, val // 2, val // 3]

    # Add some shapes to make processing interesting
    cv2.rectangle(img, (50, 50), (250, 200), (200, 150, 50), -1)    # filled rect
    cv2.rectangle(img, (50, 50), (250, 200), (255, 255, 255), 2)    # white border
    cv2.circle(img, (450, 150), 80, (50, 50, 200), -1)              # filled circle
    cv2.circle(img, (450, 150), 80, (255, 255, 255), 2)             # white border
    cv2.putText(img, "COMPUTER", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(img, "VISION", (170, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 255), 3)

    save_and_show("02_sample", img)

    # Inspect pixel values
    pixel = img[100, 150]     # row 100, column 150
    print(f"\n  Pixel at (100, 150): B={pixel[0]}, G={pixel[1]}, R={pixel[2]}")

    # Slice a region (like cropping)
    roi = img[50:200, 50:250]   # rows 50-200, cols 50-250
    save_and_show("02_cropped", roi)
    print(f"  Cropped from (50,50) to (250,200) — just array slicing!")
    print()

    return img


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: Image processing operations
# ══════════════════════════════════════════════════════════════════════════════

def part3_processing(img: np.ndarray):
    print("=" * 60)
    print("  PART 3: Image Processing Operations")
    print("=" * 60)
    print()

    # ── Grayscale ────────────────────────────────────────────────────────
    # Convert 3 channels (BGR) to 1 channel (intensity)
    # Formula: gray = 0.114*B + 0.587*G + 0.299*R (human eye sensitivity)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    save_and_show("03_grayscale", gray)
    print(f"  Shape went from {img.shape} -> {gray.shape} (3 channels -> 1)")

    # ── Blur (Gaussian) ─────────────────────────────────────────────────
    # Smooths the image by averaging nearby pixels
    # Used to reduce noise before edge detection
    # (15, 15) = kernel size — bigger = more blur
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    save_and_show("03_blurred", blurred)

    # ── Edge Detection (Canny) ───────────────────────────────────────────
    # Finds sharp transitions in brightness = edges
    # This is how self-driving cars detect lane lines, objects, etc.
    # Two thresholds control sensitivity: lower = more edges
    edges = cv2.Canny(gray, 50, 150)
    save_and_show("03_edges", edges)
    print(f"  Edges are binary: only 0 (no edge) and 255 (edge)")

    # ── Thresholding ─────────────────────────────────────────────────────
    # Convert grayscale to pure black & white
    # Every pixel above 127 becomes 255 (white), below becomes 0 (black)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    save_and_show("03_threshold", thresh)

    # ── Color space: HSV ─────────────────────────────────────────────────
    # BGR is how screens display color
    # HSV (Hue, Saturation, Value) is how humans think about color
    # HSV makes it easy to detect objects by color (e.g., "find all red things")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    save_and_show("03_hsv", hsv)
    print(f"  HSV = Hue (color), Saturation (intensity), Value (brightness)")

    # ── Resize ───────────────────────────────────────────────────────────
    # Scale images up or down — essential for feeding into neural networks
    # Most models expect a fixed input size (e.g., 224x224, 640x640)
    small = cv2.resize(img, (160, 120))
    big = cv2.resize(img, (1200, 800))
    save_and_show("03_resize_small", small)
    save_and_show("03_resize_big", big)

    # ── Rotate ───────────────────────────────────────────────────────────
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, 45, 1.0)  # 45 degrees
    rotated = cv2.warpAffine(img, matrix, (w, h))
    save_and_show("03_rotated", rotated)

    print()
    return gray


# ══════════════════════════════════════════════════════════════════════════════
# PART 4: Contour detection — finding shapes
# ══════════════════════════════════════════════════════════════════════════════

def part4_contours(img: np.ndarray, gray: np.ndarray):
    print("=" * 60)
    print("  PART 4: Finding Shapes (Contour Detection)")
    print("=" * 60)
    print()

    # Step 1: Detect edges
    edges = cv2.Canny(gray, 50, 150)

    # Step 2: Find contours (outlines of shapes)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"  Found {len(contours)} contours (shape outlines)")

    # Step 3: Draw contours on a copy of the original image
    result = img.copy()
    for i, contour in enumerate(contours):
        # Get bounding box around each contour
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        # Only draw significant contours (filter noise)
        if area > 500:
            # Draw the contour outline in green
            cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
            # Draw bounding box in yellow
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 255), 2)
            # Label with area
            cv2.putText(result, f"area:{int(area)}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            print(f"  Shape at ({x},{y}) size {w}x{h}, area={int(area)}")

    save_and_show("04_contours", result)
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  MODULE 03 — LESSON 1: IMAGE BASICS & PROCESSING")
    print("  Library: OpenCV (cv2) + NumPy")
    print("=" * 60 + "\n")

    part1_images_are_arrays()
    img = part2_load_image()
    gray = part3_processing(img)
    part4_contours(img, gray)

    print("=" * 60)
    print("  All images saved to the module-03-computer-vision/ folder.")
    print("  Open them to see the results of each operation!")
    print("=" * 60)


if __name__ == "__main__":
    main()
