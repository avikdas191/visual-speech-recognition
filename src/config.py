"""
Shared path configuration for the lip reading codebase.

Environment note
----------------
All training and dataset generation ran under WSL2 (Ubuntu 22.04) with GPU.
Paths in those notebooks point at /home/admins/ and reflect that environment.
Phase 4 onward runs on Windows, CPU only - inference and demo work, no training.
Paths here point at D:\\lipreading_workbench\\.
Both environments verified equivalent: same model, same test set, 0.7150 and
0.7556 to four decimal places.
albumentations does not install on Windows Python 3.11, so dataset generation
is WSL-only. See docs/environment/ for exact versions.
"""

import os
import sys

# if getattr(sys, "frozen", False):
#     # running from a PyInstaller executable
#     ROOT = os.path.dirname(sys.executable)
# else:
#     # running from source: config.py sits in src/, so root is one level up
#     ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# config.py sits in src/, so the project root is one level up
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RUNTIME = os.path.join(ROOT, "runtime")
FRAMES = os.path.join(RUNTIME, "extracted_frames")
CROPPED = os.path.join(RUNTIME, "cropped_frames")
GRIDS = os.path.join(RUNTIME, "grids")

MODELS = os.path.join(ROOT, "models")
CNN_MODEL_PATH = os.path.join(MODELS, "model_rebuilt_data.h5")
T5_MODEL_DIR = os.path.join(MODELS, "t5_fine_tuned_local")
PREDICTOR_PATH = os.path.join(MODELS, "shape_predictor_68_face_landmarks.dat")
LABELS_PATH = os.path.join(MODELS, "class_labels_cl10.json")

SOURCE_VIDEOS = r"D:\lipreading_archive\Data for lip reading model\Videos of 10 selected words named"


def ensure_runtime_dirs():
    for d in (FRAMES, CROPPED, GRIDS):
        os.makedirs(d, exist_ok=True)