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

W = "/home/admins/rebuild_workspace"
OUT = os.path.join(W, "models_augmentation")
word_classes = ['bat','cup','drop','eat','fish','hot','jump','milk','pen','red']

GROUPS = {
    "A_original":      "06_dataset_final",
    "B_temporal":      "06t_dataset_temporal",
    "C_photogeo":      "06p_dataset_photogeo",
    "D_volume":        "06v_dataset_volume",
    "E_full":          "06f_dataset_full",
    "F_full_temporal": "06ft_dataset_full_temporal",
    "G_full_photogeo": "06fp_dataset_full_photogeo",
    "H_vol144":        "06h_dataset_vol144",
    "I_vol192":        "06i_dataset_vol192",
    "J_vol288":        "06j_dataset_vol288",
}

def build():
    return Sequential([
        Input(shape=(224,224,3)),
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
data = os.path.join(W, GROUPS[name])
os.makedirs(OUT, exist_ok=True)

tf.keras.utils.set_random_seed(seed)
IDG = ImageDataGenerator(rescale=1/255)

def gen(sub, shuffle):
    return IDG.flow_from_directory(os.path.join(data, sub), target_size=(224,224),
        classes=word_classes, batch_size=16, class_mode='categorical', shuffle=shuffle)

train, valid = gen("Train", True), gen("Validation", False)

model = build()
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy', metrics=['accuracy'])

t0 = time.time()
hist = model.fit(train, validation_data=valid, epochs=15, verbose=0)
elapsed = time.time() - t0

h = hist.history
pd.DataFrame(h).to_csv(f"{OUT}/history_{name}_seed{seed}.csv", index=False)

row = dict(config=name, seed=seed, train_samples=train.samples, val_samples=valid.samples,
           final_train_acc=h['accuracy'][-1], final_val_acc=h['val_accuracy'][-1],
           best_val_acc=max(h['val_accuracy']), best_val_loss=min(h['val_loss']),
           best_epoch=int(np.argmin(h['val_loss']))+1, seconds=round(elapsed,1))

with open(f"{OUT}/result_{name}_seed{seed}.json", "w") as f:
    json.dump(row, f)

print(f"{name:17s} seed {seed}  train {row['final_train_acc']:.4f}  "
      f"best_val {row['best_val_acc']:.4f}  n={train.samples}  ({row['seconds']}s)")