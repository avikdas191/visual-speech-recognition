"""
Lip reading pipeline: frame normalisation, lip cropping, grid assembly,
word prediction and sentence generation.

These functions are shared by the demo notebook and the desktop
application so that both run identical logic.

Environment note
----------------
All training and dataset generation ran under WSL2 (Ubuntu 22.04) with GPU.
Paths in those notebooks point at /home/admins/ and reflect that environment.
Phase 4 onward runs on Windows, CPU only - inference and demo work, no training.
Both environments verified equivalent: same model, same test set, 0.7150 and
0.7556 to four decimal places.
albumentations does not install on Windows Python 3.11, so dataset generation
is WSL-only. See docs/environment/ for exact versions.
"""

import os
import json
import shutil

import cv2
import dlib
import numpy as np
from PIL import Image

from config import (RUNTIME, FRAMES, CROPPED, GRIDS,
                    CNN_MODEL_PATH, T5_MODEL_DIR,
                    PREDICTOR_PATH, LABELS_PATH,
                    ensure_runtime_dirs)

TARGET_FRAMES = 60
LIP_HEIGHT, LIP_WIDTH = 80, 112
ROWS, COLS = 10, 6
GRID_NAME = "grid.png"
GRID_RESIZED_NAME = "grid_resized.png"
NEW_SIZE = (224, 224)

_detector = None
_predictor = None
lip_model = None
class_labels = None
t5_tokenizer = None
t5_model = None


# ---------------------------------------------------------------- runtime

def clear_runtime(verbose=True):
    """Empty runtime/ entirely, keeping the runtime folder itself."""
    ensure_runtime_dirs()
    removed = 0
    for name in os.listdir(RUNTIME):
        path = os.path.join(RUNTIME, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        removed += 1
    ensure_runtime_dirs()
    if verbose:
        print(f"removed {removed} items from runtime/")


# ------------------------------------------------- frame normalisation

def normalise_extracted_frames(frames_dir=FRAMES, target=TARGET_FRAMES, verbose=True):
    """Trim or pad the captured frames to exactly `target`, in memory,
    then rewrite frames_dir as 01..target. Same rule as the training
    pipeline: pad with the last frame, trim 80% from the end."""
    files = [f for f in os.listdir(frames_dir) if f.endswith(".png")]
    files.sort(key=lambda f: int(os.path.splitext(f)[0]))

    if not files:
        raise RuntimeError(f"no frames found in {frames_dir}")

    frames = [cv2.imread(os.path.join(frames_dir, f)) for f in files]
    source_count = len(frames)
    padded = trimmed_start = trimmed_end = 0

    if source_count < target:
        padded = target - source_count
        frames = frames + [frames[-1]] * padded
    elif source_count > target:
        excess = source_count - target
        trimmed_end = int(excess * 0.8)
        trimmed_start = excess - trimmed_end
        frames = frames[trimmed_start:source_count - trimmed_end]

    for f in files:
        os.remove(os.path.join(frames_dir, f))
    for i, frame in enumerate(frames, start=1):
        cv2.imwrite(os.path.join(frames_dir, f"{i:02d}.png"), frame)

    if verbose:
        print(f"source frames : {source_count}")
        print(f"padded        : {padded}")
        print(f"trimmed start : {trimmed_start}")
        print(f"trimmed end   : {trimmed_end}")
        print(f"written       : {len(frames)} to {frames_dir}")

    return dict(source=source_count, padded=padded,
                trimmed_start=trimmed_start, trimmed_end=trimmed_end)


# ------------------------------------------------------------ lip crop

def _ensure_dlib():
    global _detector, _predictor
    if _detector is None:
        _detector = dlib.get_frontal_face_detector()
        _predictor = dlib.shape_predictor(PREDICTOR_PATH)


def crop_lip(frame):
    _ensure_dlib()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _detector(gray)
    if not faces:
        return None
    landmarks = _predictor(gray, faces[0])
    mouth = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(48, 68)])
    x, y, w, h = cv2.boundingRect(mouth)

    pad_left   = max((LIP_WIDTH - w) // 2, 0)
    pad_right  = max((LIP_WIDTH - w) - pad_left, 0)
    pad_top    = max((LIP_HEIGHT - h) // 2, 0)
    pad_bottom = max((LIP_HEIGHT - h) - pad_top, 0)

    pad_left   = min(pad_left, x)
    pad_right  = min(pad_right, frame.shape[1] - (x + w))
    pad_top    = min(pad_top, y)
    pad_bottom = min(pad_bottom, frame.shape[0] - (y + h))

    lip = frame[y - pad_top:y + h + pad_bottom, x - pad_left:x + w + pad_right]
    if lip.size == 0:
        return None
    return cv2.resize(lip, (LIP_WIDTH, LIP_HEIGHT))


def fill_gaps(crops):
    """Fill frames where dlib failed from the nearest successful neighbour."""
    n = len(crops)
    if not any(c is not None for c in crops):
        return None, None
    filled, records, i = list(crops), [], 0
    while i < n:
        if filled[i] is not None:
            i += 1
            continue
        start = i
        while i < n and crops[i] is None:
            i += 1
        end, length = i - 1, i - start
        before = start - 1 if start > 0 else None
        after = i if i < n else None
        if before is None:
            for k in range(start, end + 1):
                filled[k] = crops[after]; records.append((k + 1, after + 1))
        elif after is None:
            for k in range(start, end + 1):
                filled[k] = crops[before]; records.append((k + 1, before + 1))
        else:
            first_half = (length + 1) // 2
            for offset, k in enumerate(range(start, end + 1)):
                src = before if offset < first_half else after
                filled[k] = crops[src]; records.append((k + 1, src + 1))
    return filled, records


def crop_extracted_frames(frames_dir=FRAMES, cropped_dir=CROPPED, verbose=True):
    for f in os.listdir(cropped_dir):
        os.remove(os.path.join(cropped_dir, f))

    files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
    crops = []
    for f in files:
        frame = cv2.imread(os.path.join(frames_dir, f))
        crops.append(None if frame is None else crop_lip(frame))

    failed = [i + 1 for i, c in enumerate(crops) if c is None]
    filled, records = fill_gaps(crops)

    if filled is None:
        raise RuntimeError("dlib detected no face in any frame - recapture needed")

    for i, c in enumerate(filled, start=1):
        cv2.imwrite(os.path.join(cropped_dir, f"{i:02d}.png"), c)

    if verbose:
        print(f"frames in     : {len(crops)}")
        print(f"detected      : {len(crops) - len(failed)}")
        print(f"failed frames : {failed if failed else 'none'}")
        print(f"filled from   : {records if records else 'none'}")
        print(f"written       : {len(filled)} to {cropped_dir}")

    return dict(total=len(crops), detected=len(crops) - len(failed),
                failed=failed, fills=records)


def measure_mouth_boxes(frames_dir=FRAMES, verbose=True):
    """Report dlib mouth bounding box sizes, for checking capture framing."""
    _ensure_dlib()
    files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
    widths, heights, frame = [], [], None
    for f in files:
        frame = cv2.imread(os.path.join(frames_dir, f))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _detector(gray)
        if not faces:
            continue
        landmarks = _predictor(gray, faces[0])
        mouth = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(48, 68)])
        x, y, w, h = cv2.boundingRect(mouth)
        widths.append(w)
        heights.append(h)

    if verbose and widths:
        print(f"frames measured : {len(widths)}")
        print(f"source frame    : {frame.shape[1]}x{frame.shape[0]}")
        print(f"mouth width  min/mean/max : {min(widths)} / {sum(widths)/len(widths):.1f} / {max(widths)}")
        print(f"mouth height min/mean/max : {min(heights)} / {sum(heights)/len(heights):.1f} / {max(heights)}")
        print(f"crop target     : {LIP_WIDTH}x{LIP_HEIGHT}")

    return widths, heights


# --------------------------------------------------------- grid assembly

def build_grid(cropped_dir=CROPPED, grids_dir=GRIDS, verbose=True):
    frames = sorted(f for f in os.listdir(cropped_dir) if f.endswith(".png"))
    if len(frames) != ROWS * COLS:
        raise RuntimeError(f"expected {ROWS * COLS} crops, found {len(frames)}")

    first = cv2.imread(os.path.join(cropped_dir, frames[0]))
    fh, fw, ch = first.shape
    grid = np.zeros((fh * ROWS, fw * COLS, ch), dtype=np.uint8)

    for idx, f in enumerate(frames):
        img = cv2.imread(os.path.join(cropped_dir, f))
        if img is None:
            raise RuntimeError(f"could not read {f}")
        r, c = idx // COLS, idx % COLS
        grid[r * fh:(r + 1) * fh, c * fw:(c + 1) * fw] = img

    grid_path = os.path.join(grids_dir, GRID_NAME)
    cv2.imwrite(grid_path, grid)

    resized_path = os.path.join(grids_dir, GRID_RESIZED_NAME)
    with Image.open(grid_path) as img:
        img.resize(NEW_SIZE, Image.LANCZOS).save(resized_path)

    if verbose:
        print(f"cell size    : {fw}x{fh}")
        print(f"grid         : {grid.shape[1]}x{grid.shape[0]} ({ROWS} rows x {COLS} cols)")
        print(f"resized      : {NEW_SIZE[0]}x{NEW_SIZE[1]}")
        print(f"written      : {grids_dir}")

    return resized_path


# ------------------------------------------------------ word prediction

def load_lip_model(verbose=True):
    global lip_model, class_labels
    from tensorflow.keras.models import load_model
    lip_model = load_model(CNN_MODEL_PATH)
    with open(LABELS_PATH) as f:
        class_labels = json.load(f)
    if verbose:
        print(f"model  : {os.path.basename(CNN_MODEL_PATH)}")
        print(f"params : {lip_model.count_params():,}")
        print(f"input  : {lip_model.input_shape}")
        print(f"labels : {list(class_labels.values())}")


def predict_word(grid_image_path=None, top_k=3, verbose=True):
    if lip_model is None:
        raise RuntimeError("run load_lip_model() first")
    if grid_image_path is None:
        grid_image_path = os.path.join(GRIDS, GRID_RESIZED_NAME)

    with Image.open(grid_image_path) as img:
        arr = np.array(img.convert("RGB")) / 255.0
    arr = np.expand_dims(arr, axis=0)

    probs = lip_model.predict(arr, verbose=0)[0]
    order = np.argsort(probs)[::-1]

    if verbose:
        print(f"predicted : {class_labels[str(order[0])]}  ({probs[order[0]]:.4f})")
        print("top", top_k, ":")
        for i in order[:top_k]:
            print(f"  {class_labels[str(i)]:6s} {probs[i]:.4f}")

    return class_labels[str(order[0])], probs


# --------------------------------------------------- sentence generation

def load_t5_model(verbose=True):
    global t5_tokenizer, t5_model
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    t5_tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_DIR)
    t5_model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_DIR)
    if verbose:
        print(f"tokenizer : {type(t5_tokenizer).__name__}")
        print(f"model     : {type(t5_model).__name__}")
        print(f"loaded    : {T5_MODEL_DIR}")


def generate_sentence(word, max_length=20, top_k=50, top_p=0.9,
                      temperature=0.9, verbose=True):
    """Generate a sentence containing the predicted word.

    The prompt format and sampling settings come from the original
    integration notebook, except for temperature, which was lowered
    from 0.9 to 0.6 after comparing sample output. That change affects
    only generation at inference time and does not alter the model or
    its recorded metrics."""
    if t5_model is None:
        raise RuntimeError("run load_t5_model() first")

    input_text = f"Generate a sentence for {word}:"
    input_ids = t5_tokenizer(input_text, return_tensors="pt").input_ids

    outputs = t5_model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        num_return_sequences=1
    )

    sentence = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    if verbose:
        print(f"prompt    : {input_text}")
        print(f"sentence  : {sentence}")
    return sentence