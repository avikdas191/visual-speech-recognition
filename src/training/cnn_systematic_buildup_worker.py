import os, sys, time, json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Input, Conv2D, MaxPool2D, Flatten, Dense,
                                     Dropout, BatchNormalization,
                                     GlobalAveragePooling2D)
from tensorflow.keras.callbacks import Callback

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

W   = "/home/admins/rebuild_workspace"
OUT = os.path.join(W, "models_buildup")
word_classes = ['bat','cup','drop','eat','fish','hot','jump','milk','pen','red']


DATASETS = {
    "d90":  "06_dataset_final",
    "d138": "06v_dataset_volume",
    "d186": "06h_dataset_vol144",
    "d234": "06i_dataset_vol192",
    "d330": "06j_dataset_vol288",
}


VARIANTS = {
    "a0_1conv16":      dict(conv=[16],         dense=[],         lr=1e-4),
    "a1_1conv32":      dict(conv=[32],         dense=[],         lr=1e-4),
    "a2_2conv":        dict(conv=[32,64],      dense=[],         lr=1e-4),
    "a3_2conv_d128":   dict(conv=[32,64],      dense=[128],      lr=1e-4),
    "a4_baseline":     dict(conv=[32,64],      dense=[256,128],  lr=1e-4),
    "a5_3conv":        dict(conv=[32,64,128],  dense=[256,128],  lr=1e-4),
    "a6_3conv_narrow": dict(conv=[16,32,64],   dense=[256,128],  lr=1e-4),

    # dense width sweep, 2 conv
    "b0_2conv_d64":       dict(conv=[32,64],     dense=[64],      lr=1e-4),
    "b1_2conv_d32":       dict(conv=[32,64],     dense=[32],      lr=1e-4),
    "b2_2conv_d128_64":   dict(conv=[32,64],     dense=[128,64],  lr=1e-4),
    "b3_2conv_d64_32":    dict(conv=[32,64],     dense=[64,32],   lr=1e-4),

    # dense width sweep, 3 conv
    "b4_3conv_d64":       dict(conv=[32,64,128], dense=[64],      lr=1e-4),
    "b5_3conv_d128_64":   dict(conv=[32,64,128], dense=[128,64],  lr=1e-4),
    "b6_3conv_d64_32":    dict(conv=[32,64,128], dense=[64,32],   lr=1e-4),

    # 4 conv - shrinks the flatten further
    "b7_4conv_d128":      dict(conv=[32,64,128,256], dense=[128], lr=1e-4),
    "b8_4conv_d64":       dict(conv=[32,64,128,256], dense=[64],  lr=1e-4),

        # three dense layers
    "c0_2conv_d128_64_32":  dict(conv=[32,64], dense=[128,64,32], lr=1e-4),
    "c1_2conv_d64_32_16":   dict(conv=[32,64], dense=[64,32,16],  lr=1e-4),

    # wider conv, best two dense configs
    "c2_2conv_wide_d128_64": dict(conv=[64,128], dense=[128,64], lr=1e-4),
    "c3_2conv_wide_d64_32":  dict(conv=[64,128], dense=[64,32],  lr=1e-4),

    # kernel size, on b2's dense config
    "c4_2conv_k5_d128_64":  dict(conv=[32,64], dense=[128,64], kernel=5, lr=1e-4),
    "c5_2conv_k7_d128_64":  dict(conv=[32,64], dense=[128,64], kernel=7, lr=1e-4),

    # kernel size, on b3's dense config
    "c6_2conv_k5_d64_32":   dict(conv=[32,64], dense=[64,32],  kernel=5, lr=1e-4),

    # wide + large kernel combined
    "c7_2conv_wide_k5":     dict(conv=[64,128], dense=[128,64], kernel=5, lr=1e-4),

    # interpolation between b2 and b3
    "c8_2conv_d96_48":      dict(conv=[32,64], dense=[96,48], lr=1e-4),
}

BEST10 = {
    "b2": dict(conv=[32,64],          dense=[128,64]),
    "b3": dict(conv=[32,64],          dense=[64,32]),
    "a4": dict(conv=[32,64],          dense=[256,128]),
    "c8": dict(conv=[32,64],          dense=[96,48]),
    "c4": dict(conv=[32,64],          dense=[128,64], kernel=5),
    "b5": dict(conv=[32,64,128],      dense=[128,64]),
    "a5": dict(conv=[32,64,128],      dense=[256,128]),
    "b6": dict(conv=[32,64,128],      dense=[64,32]),
    "b8": dict(conv=[32,64,128,256],  dense=[64]),
    "a6": dict(conv=[16,32,64],       dense=[256,128]),
}

L2_STRENGTHS = {"1e5": 1e-5, "1e4": 1e-4, "1e3": 1e-3, "1e2": 1e-2}

for base, spec in BEST10.items():
    for tag, val in L2_STRENGTHS.items():
        VARIANTS[f"L2_{base}_{tag}"] = dict(spec, lr=1e-4, l2reg=val)

DEFAULTS = dict(conv=[32,64], dense=[256,128], act="tanh", head="flatten",
                dropout=0.0, batchnorm=False, lr=1e-4, epochs=15, batch=16,
                kernel=3)


def build(v):
    layers = [Input(shape=(224,224,3))]
    # for f in v['conv']:
    #     layers.append(Conv2D(f, (3,3), activation='relu', padding='same'))
    for f in v['conv']:
        layers.append(Conv2D(f, (v['kernel'], v['kernel']), activation='relu', padding='same'))
        if v['batchnorm']:
            layers.append(BatchNormalization())
        layers.append(MaxPool2D((2,2), strides=2))
    layers.append(Flatten() if v['head'] == "flatten" else GlobalAveragePooling2D())
    if v['dropout'] > 0:
        layers.append(Dropout(v['dropout']))
    for u in v['dense']:
        layers.append(Dense(u, activation=v['act']))
    layers.append(Dense(10, activation='softmax'))
    return Sequential(layers)


class EpochPrinter(Callback):
    """Print epoch 1, then every 5th, then the last."""
    def __init__(self, total):
        super().__init__()
        self.total = total
    def on_epoch_end(self, epoch, logs=None):
        e = epoch + 1
        if e == 1 or e % 5 == 0 or e == self.total:
            print(f"    ep {e:>3}/{self.total}  "
                  f"loss {logs['loss']:.4f}  acc {logs['accuracy']:.4f}  "
                  f"val_loss {logs['val_loss']:.4f}  val_acc {logs['val_accuracy']:.4f}",
                  flush=True)


# name, dkey, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
# v = dict(DEFAULTS)
# v.update(VARIANTS[name])
# data = os.path.join(W, DATASETS[dkey])
name, dkey, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
v = dict(DEFAULTS)
v.update(VARIANTS[name])
if len(sys.argv) > 4:
    v['epochs'] = int(sys.argv[4])
tag = f"{name}_{dkey}_ep{v['epochs']}_seed{seed}"
data = os.path.join(W, DATASETS[dkey])


os.makedirs(OUT, exist_ok=True)
tf.keras.utils.set_random_seed(seed)

IDG = ImageDataGenerator(rescale=1/255)
def gen(sub, shuffle):
    return IDG.flow_from_directory(os.path.join(data, sub), target_size=(224,224),
        classes=word_classes, batch_size=v['batch'], class_mode='categorical',
        shuffle=shuffle)

train, valid = gen("Train", True), gen("Validation", False)

model = build(v)
model.compile(optimizer=Adam(learning_rate=v['lr']),
              loss='categorical_crossentropy', metrics=['accuracy'])

print(f"\n{name} | {dkey} | seed {seed} | params {model.count_params():,} | "
      f"train {train.samples}", flush=True)

t0 = time.time()
hist = model.fit(train, validation_data=valid, epochs=v['epochs'],
                 callbacks=[EpochPrinter(v['epochs'])], verbose=0)
elapsed = time.time() - t0

h = hist.history
# pd.DataFrame(h).to_csv(f"{OUT}/history_{name}_{dkey}_seed{seed}.csv", index=False)
pd.DataFrame(h).to_csv(f"{OUT}/history_{tag}.csv", index=False)


row = dict(config=name, dataset=dkey, seed=seed, params=model.count_params(),
           train_samples=train.samples, epochs=v['epochs'], batch=v['batch'],
           lr=v['lr'], final_train_acc=h['accuracy'][-1],
           final_val_acc=h['val_accuracy'][-1],
           best_val_acc=max(h['val_accuracy']),
           best_val_loss=min(h['val_loss']),
           best_epoch=int(np.argmin(h['val_loss']))+1, seconds=round(elapsed,1))

# with open(f"{OUT}/result_{name}_{dkey}_seed{seed}.json", "w") as f:
with open(f"{OUT}/result_{tag}.json", "w") as f:
    json.dump(row, f)

print(f"  -> final_val {row['final_val_acc']:.4f}  best_val {row['best_val_acc']:.4f}  "
      f"({row['seconds']}s)", flush=True)