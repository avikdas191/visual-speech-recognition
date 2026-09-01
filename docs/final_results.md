# Final Results

Consolidated results for the reconstructed lip-reading system. Read `results_comparison.md` for the detail behind the two headline figures.

## The headline figures

|              | Full test (200 images) | Real recordings only (90) |
| ------------ | ---------------------- | ------------------------- |
| **Accuracy** | **71.50%**             | **75.56%**                |
| Loss         | 0.8225                 | 0.6907                    |
| Macro-F1     | 0.7130                 | 0.7537                    |

Model: `model_rebuilt_data.h5` — 2 conv layers (32, 64), Flatten, Dense(256, tanh), Dense(128, tanh), Dense(10, softmax). 51,434,058 parameters. Adam 1e-4, batch 16, 15 epochs, seed 202.

Dataset: rebuilt with a session-based split. 42 recording sessions in training, 9 in validation, 9 in test, with augmented images generated only from sessions within their own split. Verified after assembly: no recording session contributes to more than one split.

The real-recordings-only figure is the primary measurement. Every one of its 90 images is an independent recording the model has never encountered in any form.

## Why the evaluation was rebuilt

The original evaluation could not be relied on. The test generator was created without `shuffle=False`, so predictions and labels were compared in different orders and every per-class metric was meaningless.

The dataset split was the larger problem. Images were assigned by index after augmentation had been applied, so a test image could be a transformed copy of a recording the model had trained on. Both the evaluation code and the split needed rebuilding before any figure could be trusted.

## Two measurements, one model

|               | Original dataset          | Rebuilt dataset         |
| ------------- | ------------------------- | ----------------------- |
| Test accuracy | 98.50%                    | 71.50%                  |
| Macro-F1      | 0.9850                    | 0.7130                  |
| Errors        | 3 of 200                  | 57 of 200               |
| Model         | `model2811_36_21_d130.h5` | `model_rebuilt_data.h5` |

Same architecture, same hyperparameters, same test set size. The only difference is how images were assigned to splits.

The original split was by image index, and augmentation happened before splitting — so a test image could be a transformed copy of a training recording, and images from one session appeared on both sides. That figure measures performance on recording sessions the model has already seen.

27 percentage points of the original 98.50% was measuring recording conditions rather than mouth shapes.

## Error structure

Pooled across three seeds of the equivalent alternative architecture described below — 600 predictions, 174 errors, enough to resolve the pattern that the baseline's 3 errors could not.

**Largest confusions:**

| Pair       | Count |
| ---------- | ----- |
| milk → pen | 25    |
| red → eat  | 18    |
| jump → cup | 16    |
| bat → pen  | 11    |
| pen → bat  | 11    |
| pen → red  | 11    |
| red → pen  | 10    |

**Per-class F1, strongest to weakest:** fish 0.9333, hot 0.8889, drop 0.8397, cup 0.7344, eat 0.7132, jump 0.6869, milk 0.6596, bat 0.6549, red 0.5246, pen 0.4898.

**The bilabial cluster.** `bat`, `pen` and `milk` all begin with the lips pressed fully together. Four of the seven largest confusions occur between them. `pen` has the worst precision of any class (0.4138) because it absorbs errors from both of the others: when the model sees a bilabial closure it often guesses pen.

This is a documented limitation of visual-only speech recognition — what distinguishes these words is largely acoustic, not visual. That the errors concentrate here is evidence the model learned genuine articulatory features rather than session appearance.

**The asymmetries confirm it.** `milk` has precision 0.9118 against recall 0.5167 — when the model says milk it is usually right, but it misses half of them to pen. `jump` has the same shape (0.8718 / 0.5667). `pen` is the inverse, catching its own but absorbing others'.

**red → eat at 18** is the second-largest confusion and has no clear articulatory explanation. It appeared in the baseline's evaluation too. Recorded as an open observation.

## What was tried to improve it

Six stages, roughly 117 configurations, four test evaluations.

| Stage            | Scope                                                     | Configs | Best validation | Test outcome                     |
| ---------------- | --------------------------------------------------------- | ------- | --------------- | -------------------------------- |
| 1 — Architecture | GAP, 1×1 bottleneck, depth 2–6, relu vs tanh              | 12      | 0.745           | **0.6750 / 0.7148** — lost 4 pts |
| 2 — Training     | early stopping, batch norm, dropout ×2 positions ×2 rates | 7       | 0.721           | not tested                       |
| 3 — Input        | aspect ratio, grayscale                                   | 3       | 0.713           | not tested                       |
| 4 — Augmentation | composition, volume 90→330 per word                       | 10      | 0.758           | not tested                       |
| 5 — Pretrained   | MobileNetV2, EfficientNetB0, VGG16                        | 20      | 0.787           | **0.6533 / 0.6815** — lost 6 pts |
| 6 — Build-up     | depth, dense width, kernel size, volume, L2 ×4 strengths  | 65      | 0.775           | **0.7100 / 0.7296** — tied       |

**Nothing improved the test result.** The baseline's 71.50% / 75.56% stands.

### The one alternative worth recording

`b2_2conv_d128_64` on the 138-images-per-word dataset reaches 0.7100 / 0.7296 with **25.7M parameters against the baseline's 51.4M**. Statistically tied on both measures, at half the size.

Not adopted as the primary result — the baseline's figures are already documented, it was evaluated before any configuration search, and adopting b2 would require the deployment pipeline to generate 138 images per word instead of 90 for no measured gain. Recorded as an equivalent alternative and as the third independent demonstration that the deployed architecture is over-parameterised.

## The findings that hold

These are comparisons between configurations under identical conditions, not selections of a winner, so they are not subject to the selection failure described below.

**On architecture:**

- Two convolutional layers is optimal. Three and four are worse at every dense configuration tested.
- Two dense layers beat one, and beat three. Useful width is 64–128; below 64 loses information, above 128 adds parameters without benefit.
- Global average pooling destroys the signal on grid images. Using identical frozen MobileNetV2 features, a GAP head gives 0.310 validation and cannot fit the training set (train accuracy 0.297); a Flatten head gives 0.618 and memorises it (0.999). In a mosaic, spatial position *is* frame index.
- Kernel 5 performs identically to kernel 3 despite a substantially wider receptive field. Kernel 7 destabilises training.
- Wide first conv layers (64 filters) cause complete training failure on some initialisations — 13 of 689 runs in Stage 6 reached chance-level training accuracy of 0.08–0.10.
- `tanh` beats `relu` in the dense layers by 8.5 points at 5 conv blocks, with non-overlapping ranges across 5 seeds. The original project's choice, made on a supervisor's recommendation under time pressure, was correct.

**On transfer learning:**

- EfficientNetB0 > MobileNetV2 > VGG16 at every epoch count, matching their ImageNet ranking. Architectural simplicity does not aid transfer to out-of-distribution input.
- Frozen ImageNet features reach roughly 0.65 on test at best.
- 15 epochs — the value inherited from the original project — undertrains these models. VGG16 gains 12.5 points from 15 to 80 epochs.

**On data:**

- 234 images per word is the best dataset volume for every architecture tested. 330 is worse than 234, so augmented volume saturates between them.
- Augmentation *composition* has no measurable effect at equal volume. At higher volume, temporal augmentation beat photometric by 4.6 points with non-overlapping ranges — the reverse of what pixel-change magnitude predicted. Magnitude does not predict usefulness.
- Neither aspect-ratio correction nor grayscale conversion helps. Grayscale — the supervisor's suggestion and the 28 November logbook entry — came in 1.9 points below RGB.

**On memorisation, the unsolved problem:**

- 666 of 689 Stage 6 runs reached exactly 1.0000 training accuracy.
- Dropout at two rates in two positions: no effect. Training accuracy stayed at 1.0000 even at 50% dropout on 200,704 flattened features.
- Reducing parameters fourteen-fold across the b-series: no effect.
- L2 at 1e-5, 1e-4, 1e-3 and 1e-2 across ten architectures, 360 runs: **training accuracy exactly 1.0000 in every one**. The epoch at which validation loss bottoms does not move across four orders of magnitude of regularisation strength.

Three independent mechanisms against memorisation, all ineffective.

## The methodological failure, twice

Two configurations were promoted to test evaluation on validation evidence and both reversed.

**Stage 1.** Twelve configurations, best gained 2.8 validation points, lost 4 on test.

**Stage 5.** Sixteen configurations. Two independent mechanisms — feature adaptation and longer training — converged on the same 5.9-point gain with seed ranges that did not overlap the baseline's. Both lost 6 to 11 points on test.

The mechanism: with 200 validation images, one image is 0.5 percentage points. A 5.9-point gain is twelve images. Finding a configuration among sixteen that gets twelve particular images right is not difficult, and nothing about that transfers to a different set of nine recording sessions.

**The specific error was treating non-overlapping seed ranges as evidence of transfer.** Seed variance measures sensitivity to weight initialisation. It says nothing about whether an advantage on one set of 200 images generalises to different recordings.

After Stage 1 the response was to run fewer configurations per stage. That was insufficient — Stage 5's pool was larger, its apparent evidence stronger, and its reversal worse.

Stage 6's candidate was selected differently: not as the highest scorer, but as representative of a configuration family that sat at or near the top across three independent sweeps under different epoch counts and regularisation settings. It was the only candidate that did not collapse.

A later configuration, EfficientNetB0 at 80 epochs, reached 0.787 with the tightest seed spread of anything tested. It was deliberately **not** tested, being one point above two configurations that had already failed by the same measure.

## Limitations

**Single speaker.** All 60 recording sessions are of one person. Neither figure says anything about performance on a different face. This is the largest limitation of the project and no amount of splitting, augmentation or architecture addresses it.

**Small test set.** 200 images for the full test, 90 for the real-only subset. On the latter, one image is 11 percentage points of a single class's accuracy.

**Severe downscaling.** The 6 columns by 10 rows grid of 112×80 crops is resized to 224×224, reducing each crop to roughly 37×22 pixels and stretching it horizontally by 19%. Inherent to the original design, and a plausible ceiling on achievable accuracy. Aspect correction was tested and made no difference.

**Deployment mismatch.** The original deployment path trimmed frames differently from training, and applied sharpening and a brightness adjustment at inference that never touched the training data. The rebuilt pipeline resolves both, and the details are recorded in `deployment_live_inference/live_inference_pipeline.txt`. What remains is the capture device: training used phone recordings and the demonstrations use a webcam. This affects live behaviour rather than test metrics.

**One degraded test session.** `set_34` carries 51 padded frames across its 10 videos and is one of the 9 test sessions. The split was not redrawn in response — it was assigned by a documented rule with a fixed seed before any data-quality inspection.

**Uncontrolled version differences.** dlib 20.0.1 against the original's 19.24.2; current OpenCV. The original's intermediate cropped frames were not retained, so the rebuilt crops could not be compared directly.

## Text generation

The T5 component is secondary to the word-recognition model but is reported for completeness. Its evaluation paired each generated sentence with its own reference, so it was unaffected by the ordering problem in the CNN evaluation and these figures stand as originally recorded.

**T5-small, fine-tuned on 5,000 sentences (500 per word), 5 epochs, seed 42:**

| Metric               | Value                                      |
| -------------------- | ------------------------------------------ |
| Test loss            | 0.5564                                     |
| Perplexity           | 8.8750                                     |
| BLEU (500 sentences) | mean 0.0541, median 0.0287, range 0–0.5247 |
| METEOR               | mean 0.1776, median 0.1266, range 0–0.7923 |
| Distinct-1           | 0.0120                                     |
| Distinct-2           | 0.0163                                     |

**Interpretation.** BLEU is a weak metric here: the task is to generate *a* valid sentence containing the target word, and there are 500 valid references per word of which BLEU compares against one. Low scores are expected by construction.

The medians are roughly half the means for both BLEU and METEOR, so both distributions are heavily right-skewed — most generations score very low and a few score well. Quoting the mean alone would overstate typical performance.

Distinct-1 at 0.0120 is the more informative figure: only about 1.2% of generated unigrams are unique across the test set. The model learned sentence templates rather than diverse constructions, which is what 5 epochs over 400 sentences per word would produce.

This is the only training run in the original project with a fixed seed, and therefore the only one that is exactly reproducible.

**The alternative that was not used.** An encoder-decoder LSTM with additive attention was also trained, reaching test accuracy 0.6317 and test loss 2.0143. Its encoder input has shape `(None, 1)` — a single token — so the attention softmax operates over one position and returns 1.0 unconditionally. The attention mechanism could not function given that input shape. T5 was selected instead.

## Conclusion

The figures for the lip-reading model are **71.50%** on the full test set and **75.56%** on real recordings alone.

Across six stages of improvement work — 117 configurations spanning architecture depth, dense width, kernel size, regularisation, input representation, augmentation composition, augmentation volume, three pretrained backbones and four L2 strengths — nothing beat that baseline on test.

Every stable configuration from 2.0M to 51.6M parameters lands between 0.70 and 0.78 on validation, and every one that reached test landed between 0.60 and 0.72. The band is flat.

**The constraint is not the model.** Sixty recording sessions of a single speaker — 42 of them available for training — contain a fixed amount of information about how that speaker's mouth moves, and every reasonable configuration recovers approximately all of it. The training accuracy of 1.0000 that appears in 666 of 689 runs, unmoved by dropout, parameter reduction or L2 at any strength, is the same fact stated differently.

The directions that would plausibly matter are outside what this reconstruction could test: more speakers, more recording sessions, or an architecture that models time explicitly rather than as spatial layout. The last was attempted and abandoned on cost grounds; it remains the most substantive untried direction.

---
