# Source code

Two shared modules sit at the top level and are imported by everything else.

`config.py` holds every path used by the project. Paths are derived from the
project root rather than hardcoded, so the codebase can be moved without
editing anything.

`pipeline.py` holds the processing chain: frame normalisation, lip cropping and
gap filling, grid assembly, word recognition and sentence generation. Both
demonstrations import from here, so both run identical logic by construction
rather than by inspection.

## Folders

`data_prep/` generates the datasets. `generate_rebuilt_dataset.ipynb` is the
authoritative one and produces the dataset the reported results come from. The
other notebooks generate the variants used in the experiments: different aspect
ratios and grayscale, different augmentation compositions, and different
dataset volumes.

These notebooks require albumentations, which does not install on Windows under
Python 3.11, so they run on Linux only.

`training/` holds model training and evaluation. `cnn_evaluation_rebuilt_data.ipynb` 
produces the reported figures. The `cnn_*` notebooks each correspond to one 
experimental stage and each has a matching `_worker.py` that runs a single training 
as a separate process, which is necessary because TensorFlow does not release GPU 
memory between models in one process.

`app/` holds the two demonstrations. `lip_reading_inference_demo.ipynb` runs the
pipeline step by step with printed output at each stage, which is the clearest
place to see how the system works. `lip_reading_app.py` is the desktop
application and `lip_reading_web_app.py` is the web version.

`archive/` holds superseded code from the original project, kept unchanged as a
record of how that work ran. Nothing here is on the execution path. It is
useful for seeing what the deployment looked like before the reconstruction,
and for the sentence generation dataset notebooks.
