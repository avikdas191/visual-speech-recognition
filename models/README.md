# Models

`model_rebuilt_data.h5` is the word recognition model and produces the reported
results. Two convolutional blocks, two dense layers with tanh activation, ten
softmax outputs. 51,434,058 parameters. Input is a 224 x 224 three-channel
image.

`model2811_36_21_d130.h5` is the model deployed in the original project, kept
for reference. It is not used by anything here.

`t5_fine_tuned_local/` is the sentence generation model, a fine-tuned T5-small.
It expects the prompt form "Generate a sentence for {word}:" and will not
behave correctly with a different phrasing, because that is the form it was
fine-tuned on.

`shape_predictor_68_face_landmarks.dat` is the dlib facial landmark predictor
used to locate the mouth. Points 48 to 68 are the mouth region.

`class_labels_cl10.json` maps the model's ten output indices to words.

## Substituting a model

Paths are set in `src/config.py`, so a model of the same kind can be swapped by
changing one line. The limits are worth knowing. The word recognition path
loads a Keras model in HDF5 format, feeds it a 224 x 224 three-channel image
and expects ten class probabilities. The sentence generation path loads a
T5-compatible directory and calls it with the prompt form above. A different
architecture, such as a convolutional network with a recurrent layer, or a
transformer other than T5, needs changes to the surrounding code and not only
to the path.

## What is in this repository

Only `class_labels_cl10.json` is included. The model files are too large to
distribute through a repository: the two Keras models and the fine-tuned T5
directory come to well over 1 GB between them, and the T5 directory alone
exceeds GitHub's per-file limit.

`shape_predictor_68_face_landmarks.dat` is a third-party file from the dlib
project and is available from dlib's own distribution at
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

The two Keras models and the fine-tuned T5 model are not published. Both
applications will fail at startup without them, reporting which file is
missing. The code is included so that the method can be read and the pipeline
understood; running it end to end requires the models.
