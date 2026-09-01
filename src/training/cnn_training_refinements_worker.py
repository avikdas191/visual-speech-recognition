import os, sys, time, json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Input, Conv2D, MaxPool2D, Flatten,
                                     Dense, BatchNormalization, Dropout)
from tensorflow.keras.callbacks import EarlyStopping

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

DATA = "/home/admins/rebuild_workspace/06_dataset_final"
OUT  = "/home/admins/rebuild_workspace/models_refinements"
word_classes = ['bat','cup','drop','eat','fish','hot','jump','milk','pen','red']

CONFIGS = {
    "r0_baseline":  dict(kw={},                early_stop=False),
    "r1_earlystop": dict(kw={},                early_stop=True),
    "r2_batchnorm": dict(kw={"batchnorm":True},early_stop=False),
    "r3_dropout30": dict(kw={"dropout":0.3},   early_stop=False),
    "r4_dropout50": dict(kw={"dropout":0.5},   early_stop=False),
    "r5_dropout_preflat30": dict(kw={"dropout_preflatten":0.3}, early_stop=False),
    "r6_dropout_preflat50": dict(kw={"dropout_preflatten":0.5}, early_stop=False),
}

def build_base(batchnorm=False, dropout=0.0, dropout_preflatten=0.0):
    layers = [Input(shape=(224,224,3))]
    for f in [32, 64]:
        layers.append(Conv2D(f, (3,3), activation='relu', padding='same'))
        if batchnorm:
            layers.append(BatchNormalization())
        layers.append(MaxPool2D((2,2), strides=2))
    layers.append(Flatten())
    if dropout_preflatten > 0:
        layers.append(Dropout(dropout_preflatten))
    layers.append(Dense(256, activation='tanh'))
    if dropout > 0:
        layers.append(Dropout(dropout))
    layers.append(Dense(128, activation='tanh'))
    layers.append(Dense(10, activation='softmax'))
    return Sequential(layers)

name, seed = sys.argv[1], int(sys.argv[2])
cfg = CONFIGS[name]

tf.keras.utils.set_random_seed(seed)
IDG = ImageDataGenerator(rescale=1/255)
train = IDG.flow_from_directory(os.path.join(DATA,"Train"), target_size=(224,224),
        classes=word_classes, batch_size=16, class_mode='categorical')
valid = IDG.flow_from_directory(os.path.join(DATA,"Validation"), target_size=(224,224),
        classes=word_classes, batch_size=16, class_mode='categorical', shuffle=False)

model = build_base(**cfg['kw'])
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [EarlyStopping(monitor='val_loss', patience=4,
                           restore_best_weights=True)] if cfg['early_stop'] else []

t0 = time.time()
hist = model.fit(train, validation_data=valid, epochs=15,
                 callbacks=callbacks, verbose=0)
elapsed = time.time() - t0

h = hist.history
pd.DataFrame(h).to_csv(f"{OUT}/history_{name}_seed{seed}.csv", index=False)

row = dict(config=name, seed=seed, epochs_run=len(h['accuracy']),
           final_train_acc=h['accuracy'][-1], final_val_acc=h['val_accuracy'][-1],
           best_val_acc=max(h['val_accuracy']), best_val_loss=min(h['val_loss']),
           best_epoch=int(np.argmin(h['val_loss'])) + 1, seconds=round(elapsed,1))

with open(f"{OUT}/result_{name}_seed{seed}.json", "w") as f:
    json.dump(row, f)

print(f"{name:18s} seed {seed}  train {row['final_train_acc']:.4f}  "
      f"best_val {row['best_val_acc']:.4f}  ep {row['epochs_run']}  ({row['seconds']}s)")