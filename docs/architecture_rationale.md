

# CNN Architecture Selection: Method and Empirical Results

Written for this project, but the method generalises. It is the answer to a question that has no good answer in most tutorials: faced with a classification problem, how many convolutional layers, how many dense layers, how wide?

The common approach is to try combinations until something works. The claim here is that most of the search space can be eliminated by arithmetic before any training happens, and that what remains is a handful of informed candidates rather than thousands.

The claim is also partly wrong, and the evidence for where it fails is included. Sixty-five architectures were tested empirically over 689 training runs, and some of the reasoning below predicted the outcome while some did not.

---

## Part 1 — Compute these three numbers first

### 1. Parameters per training sample

The original model in this project:

```
Total parameters:        51,434,058
Training images:                900
Parameters per sample:       57,149
```

Where they sit:

| Component             | Parameters | Share  |
| --------------------- | ---------- | ------ |
| Conv layers (2)       | 19,392     | 0.04%  |
| Flatten → Dense(256)  | 51,380,480 | 99.90% |
| Dense(128), Dense(10) | 34,186     | 0.07%  |

Flatten produces 56 × 56 × 64 = 200,704 features. A `Dense(256)` on top is 200,704 × 256 = 51,380,480 weights. Effectively the entire network is one layer, and the part that looks at the image is under a twentieth of one percent of it.

The symptom is visible in the training curve: accuracy reaches 1.0000 by epoch 9 while validation flattens around 0.73.

The same error, larger, appears in an abandoned 3D-CNN from earlier work: a flattened 3D feature map into `Dense(512)` gave 554 million parameters against 80 training samples. That model converged to a loss of 1.6902 against ln(5) = 1.6094 — it defaulted to uniform prediction and learned nothing.

**Compute this first.** If nearly all parameters sit in one dense layer, that is the finding, and it precedes any question about depth or width.

### 2. Receptive field

How many input pixels one neuron can see. Two quantities are tracked together:

- **RF** — input pixels visible to the neuron
- **jump** — distance in input pixels between adjacent neurons' centres

```
RF_out   = RF_in + (kernel_size − 1) × jump_in
jump_out = jump_in × stride
```

Starting at RF = 1, jump = 1, for Conv 3×3 (stride 1) followed by MaxPool 2×2 (stride 2):

| After block | RF     | jump | Feature map |
| ----------- | ------ | ---- | ----------- |
| 1           | 4      | 2    | 112 × 112   |
| 2           | **10** | 4    | 56 × 56     |
| 3           | 22     | 8    | 28 × 28     |
| 4           | 46     | 16   | 14 × 14     |
| 5           | 94     | 32   | 7 × 7       |

Note the mechanism: as the feature map shrinks, each surviving neuron's view widens. The map getting smaller is the cause; the view getting wider is the effect. RF roughly doubles per block while parameters stay cheap — which is why depth is an efficient way to buy reach.

### 3. What your test set can resolve

This project's real-only test set holds 9 images per class, 90 total. One image is 1.1 percentage points overall and 11 points for its class. The validation set holds 200; one image is 0.5 points.

Measured seed variance — the same configuration retrained with different weight initialisation — is **3 to 4 percentage points**.

So a difference of 2 points cannot be distinguished from noise, and any experiment whose expected gain is below roughly 5 points is not measurable.

This is a hard constraint. It argues for few large changes rather than many small ones, and it means results must be reported as a mean across several seeds with the spread stated.

---

## Part 2 — Apply them to the input representation

This project's input is 60 lip crops tiled into one grid image. Cell size in the final 224 × 224:

```
width  = 224 / 6 columns = 37.3 px
height = 224 / 10 rows   = 22.4 px
```

The original model has RF = 10 px — 27% of a cell's width, 45% of its height. **No neuron in the convolutional stack has ever seen a complete frame, let alone two.** The conv layers act as a per-frame texture detector, and every temporal comparison is left to the dense layer.

Layout matters here. In a grid filled left to right:

```
horizontal neighbour = next frame        (t, t+1)
vertical neighbour   = six frames later  (t, t+6)
```

To span two cells: horizontally 2 × 37.3 = 74.7 px, vertically 2 × 22.4 = 44.8 px.

| Blocks | RF  | Sees t, t+6? | Sees t, t+1? |
| ------ | --- | ------------ | ------------ |
| 2      | 10  | no           | no           |
| 3      | 22  | no           | no           |
| 4      | 46  | yes          | no           |
| 5      | 94  | yes          | **yes**      |

The reasoning that follows: two blocks cannot perceive motion at all, and five blocks can. **Therefore depth should help.**

Hold that prediction. Part 4 reports what happened.

### On the grid layout itself

A transpose to 10 columns × 6 rows was considered and rejected:

```
Lip crop 112 × 80, aspect ratio 1.40 (wider than tall)

6 cols × 10 rows →  672 × 800 → 224 × 224
  cell 37.3 × 22.4, aspect 1.67   (19% horizontal stretch)

10 cols × 6 rows → 1120 × 480 → 224 × 224
  cell 22.4 × 37.3, aspect 0.60   (mouth taller than wide)
```

The transpose inverts the mouth's proportions. And 6 × 10 is provably the best available: for square output, `112 × cols = 80 × rows` with `cols × rows = 60` gives `cols = 6.55, rows = 9.16`. The nearest factor pair of 60 is 6 × 10.

| cols × rows | Dimensions    | Ratio    |
| ----------- | ------------- | -------- |
| 5 × 12      | 560 × 960     | 0.58     |
| **6 × 10**  | **672 × 800** | **0.84** |
| 10 × 6      | 1120 × 480    | 2.33     |
| 12 × 5      | 1344 × 400    | 3.36     |

Separately: **CNNs do not require square input.** `Conv2D` accepts any height and width. The 19% distortion could be removed by scaling 672 × 800 by 0.333 in both directions to 224 × 267. This was tested and made no measurable difference — see Part 4.

---

## Part 3 — What pooling does to a mosaic

Global average pooling collapses a feature map into channel means. For an ordinary photograph this is sensible: position is largely nuisance information, and a cat is a cat wherever it sits.

**For a mosaic of video frames, spatial position is frame index.** The top-left cell is frame 1; the bottom-right is frame 60. Averaging destroys which frame showed what, and a model that cannot tell frame 3 from frame 47 cannot read lips.

This was tested directly using a frozen MobileNetV2 backbone — identical weights, identical features, changing only the head:

| Head                   | Validation | Training accuracy |
| ---------------------- | ---------- | ----------------- |
| Global average pooling | 0.310      | 0.297             |
| Flatten                | **0.618**  | 0.999             |

Averaging halved performance and prevented the model from fitting even the training set. Taking features from an earlier layer and pooling those was worse still, at 0.121 — chance level.

The pattern held across all twenty pretrained configurations: every GAP configuration sat near chance, every Flatten configuration cleared 0.42.

**Use Flatten, or any head that preserves spatial layout, for mosaic input.**

One caution learned the hard way: an earlier experiment tested a 1×1 convolution bottleneck as a position-preserving alternative to GAP. It failed, and that was read as refuting the position argument. It had failed for an unrelated reason — a narrow bottleneck feeding `tanh` dense layers prevented training entirely, visible in training accuracy of 0.09 on four of five seeds. **A failed test of a hypothesis is only evidence against it if the test itself was sound.**

---

## Part 4 — What the evidence actually showed

Sixty-five architectures, 689 runs, three seeds each, five dataset volumes, four epoch settings.

### Confirmed

**Dense layers matter more than conv depth.** The three worst architectures had no dense layer at all. Adding a single `Dense(128)` to the same two conv layers gained 2.7 points.

**Two dense layers beat one, and beat three.**

| Head               | Best validation |
| ------------------ | --------------- |
| Dense(128, 64)     | 0.7717          |
| Dense(64, 32)      | 0.7617          |
| Dense(64)          | 0.7200          |
| Dense(128, 64, 32) | 0.7283          |
| Dense(64, 32, 16)  | 0.6967          |

**Width has a floor and a ceiling.** `Dense(32)` was the worst configuration in its series — below about 64 units, information is lost. And `Dense(128, 64)` at 25.7M parameters matched `Dense(256, 128)` at 51.4M — doubling the width buys nothing.

**Parameters can be halved with no cost.** The 25.7M model reached 0.7100 / 0.7296 on test against the 51.4M baseline's 0.7150 / 0.7556 — statistically tied at half the size.

### Refuted

**Depth did not help, despite the receptive field argument.**

Part 2 predicted that two blocks cannot perceive motion and five can, so depth should help. It did not:

| Conv depth | Best validation |
| ---------- | --------------- |
| 1          | 0.7283          |
| **2**      | **0.7717**      |
| 3          | 0.7650          |
| 4          | 0.7433          |
| 5          | 0.7483          |
| 6          | 0.7550          |

Two conv layers is optimal. Everything deeper is equal or worse at the same dense configuration.

**The kernel-size test isolates this cleanly.** Changing kernel size alters the receptive field without altering depth:

| Kernel | RF after 2 blocks | Best validation |
| ------ | ----------------- | --------------- |
| 3      | 10                | 0.7717          |
| 5      | 16                | 0.7517          |
| 7      | 22                | destabilises    |

A 5×5 kernel gives 60% more receptive field at the same depth and performs the same. A 7×7 gives 120% more and breaks training.

**So receptive field arithmetic did not predict performance here.** It correctly describes what the network *can* see. It does not follow that the network learns to use it — and with 900 training examples derived from 42 recording sessions, apparently it does not.

The arithmetic remains worth computing: it told us the original model was seeing less than half a frame, which is genuine information about the model. It just wasn't the binding constraint.

### Also refuted

**Aspect-ratio correction:** −0.2 points. The 19% stretch is uniform across every image, and consistent distortion is learnable.

**Grayscale:** −1.9 points. The argument was invariance — removing colour removes a cue that could identify recording sessions rather than mouth shapes. Not supported by the data.

**Augmentation composition:** no measurable effect at equal volume.

### Failure modes worth knowing

**Wide first conv layers cause complete training failure on some initialisations.** Configurations with 64 filters in the first layer produced 13 runs at 0.08–0.10 training accuracy — chance for a ten-class problem. Other seeds of the same configuration reached 1.0000 and scored normally. Initialisation determined whether they trained at all.

**Activation choice at the dense layers is load-bearing.** `tanh` beat `relu` by 8.5 points at 5 conv blocks with non-overlapping ranges across 5 seeds. `relu`'s training accuracy was also lower — it fits *less*, consistent with dead units. This architecture proved unusually sensitive to anything inserted at the conv-to-dense boundary: GAP, 1×1 bottlenecks, batch normalisation and relu all caused large failures there.

---

## Part 5 — When architecture is not the answer

Across 689 runs, **666 reached exactly 1.0000 training accuracy.** Of the 23 that did not, 13 were complete training failures.

Three independent mechanisms were tried against this:

| Mechanism                               | Result                          |
| --------------------------------------- | ------------------------------- |
| Dropout 0.3 and 0.5, two positions      | Training accuracy stayed 1.0000 |
| Parameter reduction 51.4M → 3.6M        | Training accuracy stayed 1.0000 |
| L2 at 1e-5, 1e-4, 1e-3, 1e-2 (360 runs) | Training accuracy stayed 1.0000 |

The L2 result is the most telling. Across four orders of magnitude of regularisation strength, the epoch at which validation loss bottoms does not move:

| Config | 1e-5 | 1e-4 | 1e-3 | 1e-2 |
| ------ | ---- | ---- | ---- | ---- |
| A      | 12.7 | 12.7 | 12.7 | 12.7 |
| B      | 12.0 | 12.0 | 12.0 | 12.0 |
| C      | 9.0  | 8.7  | 8.7  | 8.7  |

Identical to one decimal place. Mean validation moved 0.3 points, with no monotonic trend.

**Every stable configuration from 2.0M to 51.6M parameters landed between 0.70 and 0.78 on validation.** The best result at each of four epoch settings falls in a one-point band, 0.765 to 0.775 — narrower than the seed spread within any single configuration.

That is the signature of a data limit rather than a modelling one. Sixty recording sessions of a single speaker contain a fixed amount of information, and every reasonable configuration recovers approximately all of it.

**When your training accuracy is 1.0000 regardless of what you do to the model, stop changing the model.**

---

## Part 6 — The selection trap

This matters more than any architectural finding, because it invalidated two results before it was understood.

**Twice, a configuration was chosen on validation evidence and lost on test.**

| Stage                | Pool size  | Validation gain | Test outcome     |
| -------------------- | ---------- | --------------- | ---------------- |
| Architecture search  | 12 configs | +2.8 points     | −4 points        |
| Pretrained backbones | 16 configs | +5.9 points     | −6 to −11 points |

The second is the instructive one. Two independent mechanisms — feature adaptation and longer training — converged on the same 5.9-point gain, with seed ranges that did not overlap the baseline's at all. It looked like strong evidence. It reversed completely.

**The arithmetic:** with 200 validation images, one image is 0.5 percentage points. A 5.9-point gain is twelve images. Finding a configuration among sixteen that gets twelve particular images right is not difficult, and nothing about that transfers to a different set of recordings.

**The specific error was treating non-overlapping seed ranges as evidence of transfer.** Seed variance measures sensitivity to weight initialisation. It says nothing about whether an advantage on one validation set generalises to different data.

After the first failure the response was to run fewer configurations per stage. That was insufficient — the second pool was larger and the reversal worse.

**What worked instead:** the one candidate that did not collapse was selected differently. Not as the highest scorer, but as representative of a configuration family — two conv layers, two dense layers at 128/64 — that sat at or near the top across three independent sweeps, under different epoch counts and different regularisation settings. Consistency across independent experiments is a better signal than any single number.

A later configuration reached the highest validation figure of the entire project with the tightest seed spread. It was deliberately not tested, being one point above two configurations that had already failed by the same measure.

---

## The method, stated generally

1. **Diagnose before searching.** A training curve showing 1.0000 accuracy against flat validation says the problem is memorisation, not insufficient capacity. That eliminates every experiment aimed at adding capacity.

2. **Compute what can be computed.** Parameter counts, receptive fields, parameters per sample — all arithmetic, all available before training. They will not always predict performance, but they will tell you what your model is structurally capable of.

3. **Know what your test set can resolve.** If one image is 1.1 percentage points, differences under 5 points are not measurable, and no amount of experimentation will make them so.

4. **One variable per experiment, several seeds each, report the spread.** Report both the best epoch's result and the final epoch's — they diverge, and the difference is informative.

5. **Predict before running.** Write the expected direction and magnitude down first. A result that contradicts a recorded prediction is informative; a result that contradicts an unrecorded one gets rationalised.

6. **Select on consistency across independent experiments, not on the best single number.** The size of your selection pool relative to your validation set determines how much of any apparent gain is selection noise.

7. **Hold out a test set and touch it rarely.** Necessary, but not sufficient — see Part 6.

8. **Stop when the measurement instrument cannot read the difference.**
