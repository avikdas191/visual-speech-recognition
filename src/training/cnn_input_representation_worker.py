import os, sys, time, json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

WORKSPACE = "/home/admins/rebuild_workspace"
OUT = os.path.join(WORKSPACE, "models_input_representation")
word_classes = ['bat','cup','drop','eat','fish','hot','jump','milk','pen','red']

VARIANTS = {
    "v0_rgb_square": dict(
        data=os.path.join(WORKSPACE, "06_dataset_final"),
        real=os.path.join(WORKSPACE, "06_test_real_only"),
        target=(224,224), color='rgb', channels=3),
    "v1_rgb_aspect": dict(
        data=os.path.join(WORKSPACE, "06b_dataset_aspect_preserved"),
        real=os.path.join(WORKSPACE, "06b_test_real_only_aspect"),
        target=(267,224), color='rgb', channels=3),
    "v2_gray_square": dict(
        data=os.path.join(WORKSPACE, "06g_dataset_gray"),
        real=os.path.join(WORKSPACE, "06g_test_real_only_gray"),
        target=(224,224), color='grayscale', channels=1),
}

def build(h, w, c):
    return Sequential([
        Input(shape=(h, w, c)),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        MaxPool2D((2,2), strides=2),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPool2D((2,2), strides=2),
        Flatten(),
        Dense(256, activation='tanh'),
        Dense(128, activation='tanh'),
        Dense(10, activation='softmax'),
    ])

name, seed = sys.argv[1], int(sys.argv[2])
v = VARIANTS[name]
os.makedirs(OUT, exist_ok=True)

tf.keras.utils.set_random_seed(seed)
IDG = ImageDataGenerator(rescale=1/255)

def gen(path, shuffle):
    return IDG.flow_from_directory(path, target_size=v['target'],
        classes=word_classes, batch_size=16, class_mode='categorical',
        color_mode=v['color'], shuffle=shuffle)

train = gen(os.path.join(v['data'], "Train"), True)
valid = gen(os.path.join(v['data'], "Validation"), False)

model = build(v['target'][0], v['target'][1], v['channels'])
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy', metrics=['accuracy'])

t0 = time.time()
hist = model.fit(train, validation_data=valid, epochs=15, verbose=0)
elapsed = time.time() - t0

h = hist.history
pd.DataFrame(h).to_csv(f"{OUT}/history_{name}_seed{seed}.csv", index=False)

row = dict(config=name, seed=seed, params=model.count_params(),
           final_train_acc=h['accuracy'][-1], final_val_acc=h['val_accuracy'][-1],
           best_val_acc=max(h['val_accuracy']), best_val_loss=min(h['val_loss']),
           best_epoch=int(np.argmin(h['val_loss']))+1, seconds=round(elapsed,1))

with open(f"{OUT}/result_{name}_seed{seed}.json", "w") as f:
    json.dump(row, f)

print(f"{name:16s} seed {seed}  train {row['final_train_acc']:.4f}  "
      f"best_val {row['best_val_acc']:.4f}  params {row['params']:,}  ({row['seconds']}s)")