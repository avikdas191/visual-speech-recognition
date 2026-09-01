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

W    = "/home/admins/rebuild_workspace"
DATA = os.path.join(W, "06v_dataset_volume")          # d138
REAL = os.path.join(W, "06v_test_real_only_volume")
OUT  = os.path.join(W, "models_buildup_test")
word_classes = ['bat','cup','drop','eat','fish','hot','jump','milk','pen','red']

seed = int(sys.argv[1])
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 30
os.makedirs(OUT, exist_ok=True)
tf.keras.utils.set_random_seed(seed)

IDG = ImageDataGenerator(rescale=1/255)
def gen(root, sub, shuffle):
    path = os.path.join(root, sub) if sub else root
    return IDG.flow_from_directory(path, target_size=(224,224), classes=word_classes,
        batch_size=16, class_mode='categorical', shuffle=shuffle)

train     = gen(DATA, "Train", True)
valid     = gen(DATA, "Validation", False)
test      = gen(DATA, "Test", False)
test_real = gen(REAL, None, False)

model = Sequential([
    Input(shape=(224,224,3)),
    Conv2D(32, (3,3), activation='relu', padding='same'),
    MaxPool2D((2,2), strides=2),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPool2D((2,2), strides=2),
    Flatten(),
    Dense(128, activation='tanh'),
    Dense(64,  activation='tanh'),
    Dense(10,  activation='softmax'),
])
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy', metrics=['accuracy'])

t0 = time.time()
hist = model.fit(train, validation_data=valid, epochs=epochs, verbose=0)
elapsed = time.time() - t0

lf, af = model.evaluate(test, verbose=0)
lr, ar = model.evaluate(test_real, verbose=0)

test.reset()
y_prob = model.predict(test, verbose=0)
np.save(f"{OUT}/y_prob_b2_seed{seed}.npy", y_prob)
np.save(f"{OUT}/y_true_b2_seed{seed}.npy", test.classes)
pd.DataFrame(hist.history).to_csv(f"{OUT}/history_b2_seed{seed}.csv", index=False)

row = dict(config="b2_2conv_d128_64", dataset="d138", seed=seed, epochs=epochs,
           params=model.count_params(),
           final_val=hist.history['val_accuracy'][-1],
           best_val=max(hist.history['val_accuracy']),
           test_full_acc=af, test_full_loss=lf,
           test_real_acc=ar, test_real_loss=lr, seconds=round(elapsed,1))
with open(f"{OUT}/result_b2_seed{seed}.json", "w") as f:
    json.dump(row, f)

print(f"seed {seed}  best_val {row['best_val']:.4f}  final_val {row['final_val']:.4f}  "
      f"test_full {af:.4f}  test_real {ar:.4f}  ({row['seconds']}s)")