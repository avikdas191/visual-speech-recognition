# Training run records

Raw output from every training run in the improvement work. No model
weights - those were deleted after the results were recorded. What
remains is the evidence behind the figures quoted in docs/.

Each folder holds three kinds of file:

| File            | Contents                                                                                          |
| --------------- | ------------------------------------------------------------------------------------------------- |
| `result_*.json` | one per run: config, dataset, seed, parameter count, final and best accuracy, best epoch, runtime |
| `history_*.csv` | per-epoch training and validation curves for that run                                             |
| `*_log.csv`     | the aggregated summary across all runs in that folder                                             |

Any figure in the documents below can be traced back to these files.

## Folder map

**`models_comparison/`**
Architecture search. Depth 2-6 blocks, global average pooling against
Flatten, 1x1 convolution bottleneck, relu against tanh. Three rounds,
3-5 seeds each.
→ `docs/results_architecture/architecture_experiments.txt`

**`models_refinements/`**
Training refinements on the baseline architecture. Early stopping,
batch normalisation, dropout at two rates in two positions. Seven
configurations, five seeds.
→ `docs/results_architecture/training_refinements.txt`

**`models_input_representation/`**
Aspect-ratio preservation and grayscale conversion, against the
unmodified dataset. Three variants, five seeds.
→ `docs/results_input_representation/input_representation.txt`

**`models_augmentation/`**
Augmentation composition and volume. Temporal against photometric
replacements, full 16-type evaluation coverage, and a volume ladder
from 90 to 330 images per word. Ten configurations, five seeds.
→ `docs/results_augmentation/augmentation_experiments.txt`

**`models_pretrained/`**
Frozen ImageNet backbones - MobileNetV2, EfficientNetB0, VGG16. Twenty
configurations covering head type, feature layer depth, unfreezing,
learning rate and epoch count. Validation only.
→ `docs/results_pretrained/pretrained_backbone_experiments.txt`

**`models_pretrained_test/`**
Test evaluation of the two strongest pretrained configurations.
Includes raw prediction arrays.
→ same document

**`models_buildup/`**
Systematic build-up from a minimal architecture. Conv depth, dense
width and depth, kernel size, five dataset volumes, four epoch
settings, and L2 regularisation at four strengths. 65 configurations,
689 runs - the largest set here.
→ `docs/results_buildup/systematic_buildup.txt`

**`models_buildup_test/`**
Test evaluation of the strongest build-up configuration, three seeds.
Contains `y_prob_*.npy` and `y_true_*.npy`, the raw predictions behind
the pooled confusion matrix in `docs/results_buildup/`.
→ same document

## Naming

Result and history filenames encode the run:

`result_{config}_{dataset}_ep{epochs}_seed{seed}.json`

Some earlier folders omit the epoch tag, in which case the epoch count
is recorded inside the JSON. Dataset keys d90 through d330 refer to
images per word - see `docs/rebuild_workspace_structure.txt`.

## What is not here

Model weights. Every configuration in these folders was trained,
evaluated and discarded. Only two trained models are retained, in
`models/`:

| File                      | Description                           |
| ------------------------- | ------------------------------------- |
| `model2811_36_21_d130.h5` | the original deployed model           |
| `model_rebuilt_data.h5`   | the model behind the reported results |

All runs were executed under WSL2 with GPU acceleration. See
`docs/environment/` for the exact package versions.
