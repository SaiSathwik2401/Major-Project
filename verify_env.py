import sys
print("Verifying imports...")
try:
    import flask
    print(f"[OK] Flask installed")
except ImportError as e:
    print(f"[FAIL] Flask failed: {e}")

try:
    import tensorflow as tf
    print(f"[OK] TensorFlow {tf.__version__}")
except ImportError as e:
    print(f"[FAIL] TensorFlow failed: {e}")

try:
    import cv2
    print(f"[OK] OpenCV {cv2.__version__}")
except ImportError as e:
    print(f"[FAIL] OpenCV failed: {e}")

print("Verification complete.")
