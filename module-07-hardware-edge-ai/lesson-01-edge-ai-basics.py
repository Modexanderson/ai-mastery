"""
Module 07 -- Lesson 1: Edge AI & Hardware Basics
==================================================
So far all our AI ran on a PC. But AI is moving to the EDGE:
  - Security cameras that detect people WITHOUT a server
  - Smart doorbells that recognize faces locally
  - Factory sensors that detect defects in real-time
  - Drones that navigate autonomously
  - Cars that drive themselves

What is Edge AI?
  Running AI models DIRECTLY on the device (camera, phone, sensor)
  instead of sending data to a cloud server.

Why it matters:
  - SPEED: No network delay (real-time decisions)
  - PRIVACY: Data never leaves the device
  - COST: No cloud server bills
  - RELIABILITY: Works without internet

Hardware for Edge AI:
  - Raspberry Pi + camera (~$50)
  - NVIDIA Jetson Nano (~$150) -- GPU on a tiny board
  - Arduino + TinyML (~$30)
  - Google Coral USB (~$60) -- TPU accelerator
  - Your phone! (has NPU/Neural Engine)
  - Your PC's NPU! (Ryzen AI 9 HX 370 has a built-in NPU)

This lesson: Optimize and export models for edge deployment.
We'll take a PyTorch model, optimize it, and measure performance.
"""

import torch
import torch.nn as nn
import time
import os
import sys

print("\n" + "=" * 60)
print("  MODULE 07 -- LESSON 1: EDGE AI & HARDWARE")
print("  Running AI on Devices, Not Servers")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n  Device: {device}", end="")
if device.type == "cuda":
    print(f" ({torch.cuda.get_device_name(0)})")
else:
    print(" (CPU)")


# ============================================================
# PART 1: Model Size Matters on Edge Devices
# ============================================================
print("\n" + "=" * 60)
print("  PART 1: Why Model Size Matters")
print("=" * 60)

print("""
  On a cloud server: model size barely matters (tons of RAM/GPU)
  On edge devices: every MB counts!

  Device          | RAM    | What fits
  ----------------+--------+---------------------------
  Arduino         | 256 KB | Tiny models only (TinyML)
  Raspberry Pi    | 1-8 GB | Small models (MobileNet)
  NVIDIA Jetson   | 4-8 GB | Medium models (YOLOv8s)
  Your PC (GPU)   | 8 GB   | Large models (7B LLM)
  Cloud server    | 80+ GB | Anything (70B+ LLMs)

  The goal: make models SMALLER without losing accuracy.
  Techniques: quantization, pruning, distillation, ONNX export.
""")


# ============================================================
# PART 2: Build a Model to Optimize
# ============================================================
print("=" * 60)
print("  PART 2: Building a Model for Edge Deployment")
print("=" * 60)


class ImageClassifier(nn.Module):
    """
    A small CNN (Convolutional Neural Network) for image classification.
    This simulates a model you'd deploy on a security camera or drone.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


model = ImageClassifier().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n  Model: Image Classifier CNN")
print(f"  Parameters: {total_params:,}")
print(f"  Input: 32x32 RGB images (like a small camera feed)")
print(f"  Output: 10 classes")


# ============================================================
# PART 3: Quantization -- Shrink the Model
# ============================================================
print("\n" + "=" * 60)
print("  PART 3: Quantization -- Making Models Smaller & Faster")
print("=" * 60)

print("""
  Normal model:   weights stored as FP32 (32 bits per number)
  Quantized:      weights stored as INT8 (8 bits per number)

  Result: 4x smaller, 2-4x faster, minimal accuracy loss!

  This is how:
    - Ollama runs 7B models on your 8GB GPU (Q4 quantization)
    - Phones run AI camera features
    - Smart cameras detect people in real-time
""")

# Save original model size
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_fp32.pt")
torch.save(model.state_dict(), save_path)
original_size = os.path.getsize(save_path) / 1024
print(f"  Original model (FP32): {original_size:.1f} KB")

# Quantize to INT8 (CPU only -- this is how edge devices work)
model_cpu = ImageClassifier()
model_cpu.load_state_dict(model.cpu().state_dict())
model_cpu.eval()

quantized_model = torch.quantization.quantize_dynamic(
    model_cpu,
    {nn.Linear},  # Quantize linear layers
    dtype=torch.qint8,
)

# Save quantized model
quant_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_int8.pt")
torch.save(quantized_model.state_dict(), quant_path)
quantized_size = os.path.getsize(quant_path) / 1024
print(f"  Quantized model (INT8): {quantized_size:.1f} KB")
print(f"  Size reduction: {original_size / quantized_size:.1f}x smaller!")


# ============================================================
# PART 4: Speed Benchmark -- FP32 vs INT8
# ============================================================
print("\n" + "=" * 60)
print("  PART 4: Speed Benchmark -- FP32 vs Quantized")
print("=" * 60)

# Create fake camera frames (batch of 32x32 images)
dummy_input = torch.randn(1, 3, 32, 32)
num_runs = 100

# Benchmark original model (CPU for fair comparison)
model_cpu.eval()
start = time.time()
with torch.no_grad():
    for _ in range(num_runs):
        _ = model_cpu(dummy_input)
fp32_time = (time.time() - start) / num_runs * 1000

# Benchmark quantized model
start = time.time()
with torch.no_grad():
    for _ in range(num_runs):
        _ = quantized_model(dummy_input)
int8_time = (time.time() - start) / num_runs * 1000

print(f"\n  Inference speed ({num_runs} runs average):")
print(f"  FP32 (original):  {fp32_time:.2f} ms per image")
print(f"  INT8 (quantized): {int8_time:.2f} ms per image")
print(f"  Speedup: {fp32_time / int8_time:.1f}x faster")
print(f"\n  At this speed on edge device:")
print(f"    FP32: {1000/fp32_time:.0f} FPS (frames per second)")
print(f"    INT8: {1000/int8_time:.0f} FPS (frames per second)")


# ============================================================
# PART 5: ONNX Export -- Universal Model Format
# ============================================================
print("\n" + "=" * 60)
print("  PART 5: ONNX Export -- Run Anywhere")
print("=" * 60)

print("""
  ONNX = Open Neural Network Exchange
  It's a universal format that runs on ANY device:
    - Raspberry Pi (ONNX Runtime)
    - Web browsers (ONNX.js)
    - Mobile phones (ONNX Runtime Mobile)
    - NVIDIA Jetson (TensorRT from ONNX)
    - Intel CPUs (OpenVINO from ONNX)
""")

onnx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.onnx")
try:
    torch.onnx.export(
        model_cpu,
        dummy_input,
        onnx_path,
        input_names=["image"],
        output_names=["prediction"],
        dynamic_axes={"image": {0: "batch"}, "prediction": {0: "batch"}},
    )
    onnx_size = os.path.getsize(onnx_path) / 1024
    print(f"  Exported to ONNX: {onnx_size:.1f} KB")
    print(f"  This file can now run on Raspberry Pi, phones, browsers!")
except Exception as e:
    print(f"  ONNX export skipped: {e}")
    print(f"  (Install with: pip install onnx)")


# ============================================================
# PART 6: Edge AI Product Ideas
# ============================================================
print("\n" + "=" * 60)
print("  PART 6: Edge AI Product Ideas You Can Build")
print("=" * 60)

print("""
  With what you've learned in this course, you can build:

  1. SMART SECURITY CAMERA ($99-299/unit)
     - Raspberry Pi + Camera + YOLOv8 (you built this in Module 03!)
     - Person/car/animal detection
     - Alerts via WiFi, no cloud needed
     - Sell hardware + monthly monitoring subscription

  2. RETAIL PEOPLE COUNTER ($199/unit)
     - Camera at store entrance
     - Counts customers, tracks peak hours
     - Dashboard for store owner
     - Monthly analytics subscription

  3. QUALITY INSPECTION SYSTEM ($500-2000/unit)
     - Camera on factory production line
     - Detects defects in products (scratches, dents)
     - Fine-tune model on client's specific products
     - Huge market: manufacturing, food, electronics

  4. SMART DOORBELL ($79-149/unit)
     - Face recognition (knows family vs strangers)
     - Package delivery detection
     - Runs locally (privacy selling point)

  5. PARKING LOT MONITOR ($299/unit)
     - Detect empty/occupied spaces
     - Display availability on screen
     - API for mobile app
     - Sell to malls, hospitals, airports

  Your hardware advantage:
    - RTX 4070: train and optimize models
    - Ryzen AI NPU: test edge inference locally
    - Export to ONNX: deploy on any device
""")


# ============================================================
# PART 7: Your PC's NPU
# ============================================================
print("=" * 60)
print("  BONUS: Your Ryzen AI NPU")
print("=" * 60)

print("""
  Your AMD Ryzen AI 9 HX 370 has a built-in NPU
  (Neural Processing Unit) rated at 50 TOPS.

  What is an NPU?
    A chip designed ONLY for AI inference.
    More power-efficient than GPU for small models.

  TOPS = Tera Operations Per Second
    Your NPU: 50 TOPS
    Apple M3 NPU: 18 TOPS
    Snapdragon phone: 45 TOPS

  The NPU is perfect for:
    - Always-on AI features (background tasks)
    - Camera effects, noise cancellation
    - Running small models without draining battery
    - Windows AI features (Copilot, Recall)

  To use it: AMD provides the Ryzen AI SDK
  (we'll explore this more if you want to go deeper)
""")

# Cleanup temp files
for f in ["model_fp32.pt", "model_int8.pt", "model.onnx"]:
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), f)
    if os.path.exists(p):
        os.remove(p)

print("=" * 60)
print("  KEY TAKEAWAYS:")
print("  1. Edge AI = running models ON the device, not in the cloud")
print("  2. Quantization: FP32 -> INT8 = 4x smaller, 2-4x faster")
print("  3. ONNX: universal format, runs on any device")
print("  4. Your RTX 4070 trains models, edge devices RUN them")
print("  5. Hardware AI products = recurring revenue (device + subscription)")
print("  6. Your Ryzen NPU can run small models efficiently")
print("=" * 60)
