# Evaluation Results: Original Split vs Rebuilt Dataset

## The two measurements

|                    | Original dataset | Rebuilt dataset |
| ------------------ | ---------------- | --------------- |
| Test accuracy      | **98.50%**       | **71.50%**      |
| Macro-F1           | 0.9850           | 0.7130          |
| Weighted-F1        | 0.9850           | 0.7130          |
| Per-class F1 range | 0.9744 – 1.0000  | 0.5556 – 0.8837 |
| Errors             | 3 of 200         | 57 of 200       |
| Test loss          | 0.1230           | 0.8225          |

Both figures come from the same architecture (2 conv layers, 51,434,058 parameters), the same hyperparameters (Adam 1e-4, batch 16, 15 epochs), the same input size, and the same test set size of 20 images per class. The only variable that differs is how images were assigned to training and test.

A third measurement exists for the rebuilt dataset only: **75.56%** on the 9 real recordings per class, with augmented images excluded entirely. No equivalent exists for the original dataset, because the mapping from augmented images back to their source recordings was never saved and cannot be reconstructed. The comparison above therefore uses full test sets on both sides.

Word Error Rate is not reported. For isolated single-word classification it reduces to `1 − accuracy` with concatenation artifacts, and adds nothing over the accuracy figure. The original notebook's WER of 0.86 was computed through the misaligned evaluation described below and is invalid in any case.

## What each number measures

**98.50% — the original dataset.** Images were assigned to train, validation and test by index number. Images 001–060 are the 60 real recording sessions; 061–130 are augmented images generated from them. Because the assignment was by index and the augmentation happened before splitting, a test image can be a transformed copy of a recording the model trained on, and images from the same recording session appear on both sides of the split.

This figure answers: *how well does the model classify images from recording sessions it has already seen?*

**71.50% — the rebuilt dataset.** The 60 recording sessions were assigned to splits first, stratified by recording location. Augmentation was then applied within each split only, so no augmented image derives from a session in a different split. This was verified after assembly by tracing every one of the 1,300 final images back to its origin session: zero sessions appear in more than one split.

This figure answers: *how well does the model classify recordings it has never encountered?*

The second question is the one that matters for any real use of the system.

## The gap

27 percentage points.

The training curves make the mechanism visible. On the rebuilt data, training accuracy reaches 1.0000 by epoch 9 and training loss continues falling to 0.0085, while validation accuracy flattens at 0.70–0.74 from epoch 7 onward. The model memorises its 42 training sessions completely and then stops learning anything that transfers.

The same architecture on the original split reached roughly 98% validation accuracy. That did not happen because the model generalised better — it happened because the validation set contained material derived from the training sessions.

Put plainly: a substantial part of what the original 98.50% measured was the model's ability to recognise recording conditions — a particular lighting setup, camera angle, and framing — rather than mouth shapes. When those conditions are held out, roughly a quarter of the apparent accuracy disappears.

## Do the confusion patterns support this?

This was the diagnostic question. If the model had learned nothing but session appearance, its errors on unseen sessions would scatter arbitrarily. If it learned genuine articulatory features, errors should cluster on words that genuinely look alike.

Four confusions appear in both the full test set and the real-recordings-only subset:

| Confusion  | Full test (of 20) | Real only (of 9) |
| ---------- | ----------------- | ---------------- |
| bat ↔ pen  | 4 and 6           | 2 and 2          |
| milk → pen | 7                 | 3                |
| red → eat  | 10                | 4                |
| cup → jump | 5                 | 2                |

**bat, pen and milk all begin with a bilabial closure** — the lips pressed fully together before release. The visual gesture is close to identical across the three; what separates the words is largely acoustic. This is a documented limitation of visual-only speech recognition, not a defect specific to this model. That the errors concentrate here is evidence the model learned real lip features.

**red → eat** is the largest single confusion and is one-directional: `eat` is never mistaken for `red`. This produces the asymmetry in `eat`'s scores — recall 0.9000 against precision 0.6207. No clear articulatory explanation is offered here; it is recorded as an open observation.

The overall picture is that the model did learn something real about mouth shapes, and the errors it makes are largely the errors lip reading is expected to make. The original evaluation simply could not show this, because near-perfect scores leave no error structure to examine.

## The live performance gap

The original project's `integration.ipynb` recorded live webcam predictions with top-class confidences of 0.364 (for `cup`, with `pen` at 0.305) and 0.553 (for `jump`). A model genuinely operating at 98.50% would not produce a near-tie on a correct prediction.

The rebuilt figure of 71.50% is far more consistent with that observed behaviour, but it does not explain the gap entirely. Two documented pipeline inconsistencies contribute:

**Frame trimming differs between training and deployment.** The training pipeline removes excess frames 20% from the start and 80% from the end. The web application uses 80% from the start and 20% from the end; `integration.ipynb` uses 90/10. The deployment path therefore extracts a different portion of the recording than the model was trained on.

**Preprocessing differs.** `process_image.py` applies sharpening and a fixed +5% brightness and contrast adjustment. It runs in `integration.ipynb`, does not run in `web_app.ipynb`, and never touched the training data.

**Capture device differs.** Training data was recorded on a phone; live input comes from a webcam, with different sensor characteristics, resolution and colour response.

These are cheap to align and are the first things to test in any follow-up work.

## Limitations that both figures share

**Single speaker.** All 60 recording sessions are of one person. Neither figure says anything about performance on a different face. This is the largest limitation of the project and no amount of splitting addresses it.

**Small test set.** 200 images across 10 classes for the full test, 90 for the real-only subset. On the real-only figure, one image is 11 percentage points of a single class's accuracy. Class-level numbers should be read as indicative.

**Severe downscaling.** The 6 columns by 10 rows grid of 112×80 lip crops is resized to 224×224, reducing each crop to roughly 37×22 pixels. Whatever distinguishes one mouth shape from another must survive that reduction. This is inherent to the original design and is a plausible ceiling on achievable accuracy.

**Augmentation distribution differs between train and evaluation** in the rebuilt dataset. Training uses all 16 augmentation types; validation and test use 11. This affects the 71.50% figure but not the 75.56% real-only figure, which contains no augmented images.

**Uncontrolled version differences.** The rebuild used dlib 20.0.1 against the original's 19.24.2, and a current OpenCV. The original's intermediate cropped frames were not retained, so the rebuilt crops could not be compared directly against the originals.

## Which figure to quote

**71.50%**, with the original 98.50% shown alongside as context and clearly labelled as measured on a split with session overlap.

The 75.56% real-recordings-only figure is the cleanest estimate the project can produce and should be reported too, with its sample size stated.

Quoting 98.50% alone would not be false — it is a correctly computed number — but it would be misleading about what the system does, and the gap between it and observed live behaviour would be unexplained.

---
