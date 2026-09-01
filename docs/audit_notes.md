# Codebase Audit — Lip-to-Text Recognition System

**Purpose:** Record of the codebase review conducted before reconstruction. Documents the pipeline as built, defects identified, and decisions on what was retained.

---

## 1. System Overview

A lip-reading system that classifies spoken words from silent video and generates a sentence containing the predicted word.

**Pipeline:**

1. Hand gesture (MediaPipe) starts and stops webcam recording
2. Video trimmed to 60 frames
3. Lip region cropped from each frame using dlib 68-point landmarks (points 48–68), output 112×80
4. 60 crops tiled into a single 6 columns by 10 rows grid image, resized to 224×224
5. 2D CNN classifies the grid into one of 10 word classes
6. Fine-tuned T5-small generates a sentence containing the predicted word
7. Flask web page displays status; camera feed runs in a separate OpenCV window

**Word classes:** bat, cup, drop, eat, fish, hot, jump, milk, pen, red

**Deployed CNN:** `model2811_36_21_d130.h5` — 2 convolutional layers (32, 64), Flatten, Dense 256 → 128 → 10. 51.4M parameters.

**Dataset:** `7_final_images_130` — 130 images per word (60 original recording sessions, 70 augmented). Split 90/20/20 per word = 900 train, 200 validation, 200 test.

**T5:** 5000 sentences (500 per word), assembled from published dialogue corpora and hand-written sentences, then expanded by text augmentation. Fine-tuned 5 epochs, seed 42.

---

## 2. Development History

Three build directories, developed partly in parallel rather than strictly in sequence.

**Build-1** — Frame extraction and Haar Cascade lip cropping. Two hybrid architectures attempted (3D-CNN feeding ResNet50 and EfficientNetB0, both ImageNet-pretrained and frozen). Neither reached training. Established the 60-frame target and zero-padded filenames. Abandoned; Haar Cascade cropping was insufficiently accurate.

**Build-2** — 3D-CNN era. Switched to dlib landmark-based cropping. Data stored as `data.txt` JSON dumps. Model trained on 5 classes, 100 samples. Failed at chance level (see §3.1). Abandoned.

**Build-3** — Grid-image era. Reformulated the problem as 2D image classification by tiling 60 frames into a single image. This is the approach that reached a working system.

---

## 3. Defects Identified

### 3.1 3D-CNN architecture — 554M parameters on 80 training samples

`3DCNN.ipynb` flattened a 3D feature map directly into a Dense(512) layer, producing 1,081,344 input features and ~553M weights in that single layer — 99.9% of the model. Training log shows loss converging to 1.6902 against ln(5) = 1.6094, i.e. the model defaulted to uniform prediction across five classes. Accuracy remained at ~0.20 (chance).

Two independent causes: the parameter count relative to sample size, and the fact that 18 of 20 training sets were augmentations derived from only 2 real recordings.

**Resolution:** Approach abandoned in favour of the 2D grid-image formulation. Retained in `src/archive/` as documentation of the design decision.

### 3.2 Evaluation shuffle bug — invalidates all per-class metrics

In `final_data_ready.ipynb` and `test.ipynb`, the test generator was created without `shuffle=False`:

```python
test = IDG.flow_from_directory(...)   # shuffle defaults to True
```

Predictions were then obtained via `model.predict(test)` and compared against `test.classes`. `predict` iterates the generator in shuffled order; `test.classes` returns labels in directory order. The two are misaligned, so every metric computed this way compares each prediction against a label belonging to a different image.

Observed effect: `model.evaluate()` reported 98.5% test accuracy, while the per-class table computed in the same notebook on the same data showed chance-level figures across all 10 classes.

`model.evaluate()` is unaffected because it pairs predictions with labels internally.

**Scope:** The evaluation block was written once and copy-pasted between notebooks. All previously computed confusion matrices, per-class precision/recall tables, and WER figures in this project are invalid.

**Resolution:** `shuffle=False` was set on the test generator and metrics regenerated.

### 3.3 Non-reproducible data generation

Neither `pre_augment.py`, `post_augment.py`, nor the augmentation code in `initial_data_ready.ipynb` sets a random seed. The train/validation/test split in `initial_data_ready.ipynb` cell 22 shuffles indices without a seed; the resulting index lists were hardcoded afterwards.

The augmentation scripts printed which original recording session each augmented sample derived from, but this output was never written to a file.

**Consequence:** The mapping from augmented image to source recording session is unrecoverable. A session-based split cannot be constructed from the existing dataset; it would require regenerating the dataset from the original videos.

**Resolution:** Seeds were added in the rebuilt pipeline. The original dataset is retained with the limitation documented.

---

## 4. Known Limitations

### 4.1 Train/test split leakage

Images 001–060 correspond to 60 recording sessions; 061–130 are augmentations derived from them. The split was applied by image index, uniformly across classes. Augmented images therefore appear in the test set while their source recordings appear in training, and images from the same recording session appear on both sides of the split.

Test accuracy from this split should be read as an upper bound, not as a generalisation estimate.

### 4.2 Training/inference preprocessing mismatch

Frame trimming ratios differ across the codebase: 20/80 in the training data pipeline (`initial_data_ready.ipynb`), 80/20 in the web application, 90/10 in `integration.ipynb`. The training and deployment paths therefore extract different portions of the recorded video.

Additionally, `process_image.py` (sharpening plus fixed +5% brightness/contrast) is applied in `integration.ipynb` but not in `web_app.ipynb`, and was never applied to training data.

**Observed effect:** Live predictions in `integration.ipynb` produced top-class confidences of 0.36 ("cup", with "pen" at 0.31) and 0.55 ("jump"), against a model reporting 98.5% on its own test set. Live webcam input is substantially out of distribution relative to the phone-recorded training data.

### 4.3 Dataset composition

All 60 original recordings are of a single speaker, captured across approximately five locations and five head positions. The model's reported accuracy reflects performance on one face under conditions represented in training.

### 4.4 Stale frame directory in the web application

`cropped_frames_dir` is not cleared between runs in `web_app.ipynb`. If a run produces fewer than 60 cropped frames due to detection failure, frames from the previous run remain and the grid assembly proceeds with mixed data.

### 4.5 LSTM sequence-to-sequence text generation

An alternative to T5 was implemented in `txtdata_1st.ipynb` using an encoder-decoder LSTM with `AdditiveAttention`. The encoder input has shape `(None, 1)` — a single token — so the attention softmax operates over one position and returns 1.0 unconditionally. The attention mechanism cannot function given this input shape.

Not used in the final system. T5 was selected instead.

---

## 5. Text Generation — Metrics Status

T5 evaluation in `txtdata_1st.ipynb` does not share the defect in §3.2; each generated sentence is paired with its own reference. These figures are valid as recorded:

- Test loss: 0.5564
- Perplexity: 8.87
- Distinct-1: 0.0120
- Distinct-2: 0.0163
- BLEU: predominantly 0.02–0.15
- METEOR: predominantly 0.05–0.30

**Interpretation:** BLEU is a weak metric for this task — the objective is to generate *a* valid sentence containing the target word, and there are 500 valid references per word, of which BLEU compares against one. The Distinct-1 figure of 0.012 is more informative: generated output is highly repetitive across the test set, indicating the model learned sentence templates rather than diverse constructions. Consistent with 5 epochs over 400 sentences per word.

This is the only training run in the project with a fixed seed (42) and is therefore reproducible.

---

## 6. Retention Decisions

**Retained in `src/`:**

| File                             | Location  | Role                             |
| -------------------------------- | --------- | -------------------------------- |
| `crop_lip.py`                    | data_prep | dlib lip cropping, batch         |
| `initial_data_ready.ipynb`       | data_prep | Full training-data pipeline      |
| `file_handling.ipynb`            | data_prep | Dataset organisation             |
| `final_data_ready.ipynb`         | training  | CNN training and evaluation      |
| `txtdata_1st.ipynb`              | training  | T5 fine-tuning, LSTM experiments |
| `txtdata_2nd.ipynb`              | training  | Text corpus construction         |
| `web_app.ipynb`                  | app       | Deployed Flask application       |
| `index.html`, `original_UI.html` | app       | Front end and source template    |

Locations in this table reflect the layout at audit time. Several of these files were later superseded by rebuilt equivalents and moved to `src/archive/`; see `README.md` for the current layout.

**Retained in `src/archive/`** — not part of the pipeline, kept as development record: `3DCNN.ipynb`, `pre_augment.py`, `post_augment.py`, `augment_test.ipynb`, `camera_test.ipynb`

**Excluded:**

- Build-1 in full (`v1.ipynb`, `v2.ipynb`) — superseded, Haar Cascade approach abandoned
- Build-2 except `3DCNN.ipynb` — superseded by build-3 equivalents
- `test.py`, `test.ipynb` — scratch files
- `integration.ipynb`, `web_app_test.ipynb`, `process_image.py` — superseded by `web_app.ipynb`
- `src_4_pro_dis_2/` — complete duplicate of retained content
- 16 of 17 CNN checkpoints — only the deployed model retained
- All dataset variants other than 130 — experimental intermediates
- `t5_fine_tuned/` — Trainer checkpoint with optimizer state; weights identical to `t5_fine_tuned_local/`, which additionally contains the tokenizer

---

## 7. Work Arising From This Audit

Everything listed here was addressed during the reconstruction.

1. CNN evaluation metrics regenerated with `shuffle=False` (§3.2). Results in `results_rebuilt_data/`.
2. Dataset rebuilt from the original video with session-aware splitting and fixed seeds, removing the leakage described in §4.1. Method in `results_rebuilt_data/split_method.txt`, outcome in `results_comparison.md`.
3. Licence terms for the text corpora and the Bootstrap template confirmed. Neither the corpora nor the derived dataset are redistributed, so no redistribution terms apply.
4. Model file sizes resolved by keeping the models outside the repository. See `models/README.md`.
5. The preprocessing mismatches in §4.2 and the stale frame directory in §4.4 are resolved in the rebuilt pipeline. Details in `deployment_live_inference/live_inference_pipeline.txt`.

---

This audit records the state of the codebase before reconstruction. The work that followed is documented in the results and method documents listed in `README.md`.
