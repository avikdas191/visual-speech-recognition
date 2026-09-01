# Documentation index

The documents here are grouped by the question they answer. Most are plain text
and can be read in any order, though the dataset documents make more sense
before the experiment ones.

---

## Start here

**[`architecture_rationale.md`](architecture_rationale.md)**
Why the system is built the way it is. Explains the grid-image representation,
why a 2D convolutional network is used for what is fundamentally a temporal
problem, and what evidence supports each design decision. The single most
useful document for understanding the project.

**[`final_results.md`](final_results.md)**
Consolidated results across the whole reconstruction.

**[`results_comparison.md`](results_comparison.md)**
The 98.50% against 71.50% analysis: why the original split produced a figure 27
points higher, and what that figure was actually measuring.

---

## How the dataset was built

**[`results_rebuilt_data/split_method.txt`](results_rebuilt_data/split_method.txt)**
How recordings were divided into training, validation and test sets by session
rather than by image, and why that distinction is the whole point.

**[`results_rebuilt_data/frame_extraction_method.txt`](results_rebuilt_data/frame_extraction_method.txt)**
How a variable-length recording becomes exactly 60 frames.

**[`results_rebuilt_data/lip_cropping_method.txt`](results_rebuilt_data/lip_cropping_method.txt)**
How the mouth region is located and cropped to 112 x 80.

**[`results_rebuilt_data/augmentation_method.txt`](results_rebuilt_data/augmentation_method.txt)**
The sixteen augmentation types and how they are applied within splits.

**[`results_rebuilt_data/dataset_assembly_method.txt`](results_rebuilt_data/dataset_assembly_method.txt)**
How crops are tiled into grid images and assembled into the final dataset.

**[`results_rebuilt_data/dataset_diagnostics.txt`](results_rebuilt_data/dataset_diagnostics.txt)**
Verification that no session appears in more than one split.

**[`project_setup/rebuild_workspace_structure.txt`](project_setup/rebuild_workspace_structure.txt)**
The folder layout the dataset generation produces.

**[`text_corpus_construction/text_dataset_method.txt`](text_corpus_construction/text_dataset_method.txt)**
How the sentence generation dataset was built: sources, augmentation, cleaning,
and what its construction explains about the model's low diversity scores.

---

## Results

**[`results_rebuilt_data/rebuilt_data_summary.txt`](results_rebuilt_data/rebuilt_data_summary.txt)**
Results on the rebuilt dataset, with confusion matrices, classification reports
and training curves alongside it.

**[`results_rebuilt_data/training_and_evaluation_method.txt`](results_rebuilt_data/training_and_evaluation_method.txt)**
How the model was trained and evaluated.

**[`results_existing_data/existing_data_summary.txt`](results_existing_data/existing_data_summary.txt)**
Results on the original dataset, retained for comparison. This is where the
98.50% figure comes from.

Both results folders contain the raw prediction arrays, so any metric can be
recomputed without retraining.

---

## Improvement work and its outcomes

Six stages of experiments, roughly 117 configurations. Nothing beat the
baseline on test. These documents record what was tried and what can be
concluded from the failures.

**[`results_architecture/architecture_method.txt`](results_architecture/architecture_method.txt)**
How to narrow architecture decisions by arithmetic before training anything.
Parameters per sample, receptive field, and what a test set of a given size can
actually resolve. The most reusable document here.

**[`results_architecture/architecture_experiments.txt`](results_architecture/architecture_experiments.txt)**
Twelve configurations across three rounds. Includes a correction appended
later: a hypothesis recorded as refuted was in fact correct, and the experiment
that appeared to refute it had failed for an unrelated reason.

**[`results_architecture/training_refinements.txt`](results_architecture/training_refinements.txt)**
Early stopping, batch normalisation and dropout. No effect above noise, and
batch normalisation was substantially harmful.

**[`results_input_representation/input_representation.txt`](results_input_representation/input_representation.txt)**
Aspect ratio and grayscale input.

**[`results_augmentation/augmentation_experiments.txt`](results_augmentation/augmentation_experiments.txt)**
Augmentation composition and dataset volume from 90 to 330 images per word.

**[`results_pretrained/pretrained_backbone_experiments.txt`](results_pretrained/pretrained_backbone_experiments.txt)**
MobileNetV2, EfficientNetB0 and VGG16. Contains the cleanest demonstration in
the project that global average pooling destroys the signal on grid input, and
the clearest case of validation evidence reversing on test.

**[`results_buildup/systematic_buildup.txt`](results_buildup/systematic_buildup.txt)**
689 training runs across 65 configurations, building upward from a minimal
model. Produced the one configuration that matches the baseline on test at half
the parameter count.

---

## The working system

**[`deployment_live_inference/live_inference_pipeline.txt`](deployment_live_inference/live_inference_pipeline.txt)**
The complete path from webcam to generated sentence. Includes the finding that
capture resolution and camera distance trade off directly, because the lip crop
depends only on the mouth's size in pixels.

**[`deployment_demonstrations/desktop_application.txt`](deployment_demonstrations/desktop_application.txt)**
The single-window application: how it was built, why the interface is arranged
as it is, and the finding that ambient lighting affects capture frame rate
through sensor exposure time.

**[`deployment_demonstrations/web_application.txt`](deployment_demonstrations/web_application.txt)**
The two-window arrangement, reproducing how the project was originally
presented, and why it is kept alongside the desktop application rather than
replaced by it.

---

## Environment and codebase

**[`project_setup/environment_setup.txt`](project_setup/environment_setup.txt)**
Why training ran under Linux, and how the exact package versions were
recovered.

**[`environment/`](environment/)**
Exact package versions for both platforms. Files ending in `_wsl` describe the
Linux environment where training and dataset generation ran; those ending in
`_windows` describe the Windows environment used for inference and the
demonstrations.

**[`audit_notes.md`](audit_notes.md)**
An audit of the original codebase and the defects it found.

---

## Media

**[`media/`](media/)**
Screenshots and a recording of the two demonstrations, plus two example grid
images showing what the model actually receives as input. These are referenced
from the main README.
