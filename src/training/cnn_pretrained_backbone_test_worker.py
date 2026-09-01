import os, sys, time, json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

W    = "/home/admins/rebuild_workspace"
DATA = os.path.join(W, "06_dataset_final")
REAL = os.path.join(W, "06_test_real_only")
OUT  = os.path.join(W, "models_pretrained_test")
word_classes = ['bat','cup','drop','eat','fish','hot','jump','milk','pen','red']

CONFIGS = {
    "m13_eff_unfreeze20": dict(unfreeze=20, epochs=15),
    "m14_eff_epochs40":   dict(unfreeze=0,  epochs=40),
}

name, seed = sys.argv[1], int(sys.argv[2])
c = CONFIGS[name]
os.makedirs(OUT, exist_ok=True)
tf.keras.utils.set_random_seed(seed)

IDG = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input)

def gen(root, sub, shuffle):
    path = os.path.join(root, sub) if sub else root
    return IDG.flow_from_directory(path, target_size=(224,224), classes=word_classes,
        batch_size=16, class_mode='categorical', shuffle=shuffle)

train = gen(DATA, "Train", True)
valid = gen(DATA, "Validation", False)
test  = gen(DATA, "Test", False)
test_real = gen(REAL, None, False)

base = tf.keras.applications.EfficientNetB0(
    include_top=False, weights='imagenet', input_shape=(224,224,3))
base.trainable = False
if c['unfreeze'] > 0:
    base.trainable = True
    for layer in base.layers[:-c['unfreeze']]:
        layer.trainable = False

inp = Input(shape=(224,224,3))
x = base(inp, training=(c['unfreeze'] > 0))
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(256, activation='tanh')(x)
out = Dense(10, activation='softmax')(x)
model = Model(inp, out)
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy', metrics=['accuracy'])

t0 = time.time()
hist = model.fit(train, validation_data=valid, epochs=c['epochs'], verbose=0)
elapsed = time.time() - t0

lf, af = model.evaluate(test, verbose=0)
lr, ar = model.evaluate(test_real, verbose=0)

test.reset()
y_prob = model.predict(test, verbose=0)
np.save(f"{OUT}/y_prob_{name}_seed{seed}.npy", y_prob)
np.save(f"{OUT}/y_true_{name}_seed{seed}.npy", test.classes)

row = dict(config=name, seed=seed,
           val_acc=hist.history['val_accuracy'][-1],
           test_full_acc=af, test_full_loss=lf,
           test_real_acc=ar, test_real_loss=lr, seconds=round(elapsed,1))
with open(f"{OUT}/result_{name}_seed{seed}.json", "w") as f:
    json.dump(row, f)

pd.DataFrame(hist.history).to_csv(f"{OUT}/history_{name}_seed{seed}.csv", index=False)
model.save(f"{OUT}/{name}_seed{seed}.keras")

print(f"{name:20s} seed {seed}  val {row['val_acc']:.4f}  "
      f"test_full {af:.4f}  test_real {ar:.4f}  ({row['seconds']}s)")