import os, sys, time, json
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, TimeDistributed, Conv2D, MaxPool2D,
                                     Flatten, Dense, Dropout, LSTM, GRU,
                                     Bidirectional, GlobalAveragePooling1D,
                                     BatchNormalization)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

W = "/home/admins/rebuild_workspace"
CROPPED = os.path.join(W, "02_frames_cropped")
AUGMENTED = os.path.join(W, "03_frames_augmented")
LOGS = os.path.join(W, "logs")
OUT = os.path.join(W, "models_sequence")
SPLIT_CSV = "/home/admins/lip_codebase_clean/docs/results_rebuilt_data/session_split_assignment.csv"

WORDS = ['bat','cup','drop','eat','fish','hot','jump','milk','pen','red']
N_FRAMES, FH, FW = 60, 80, 112


# ---------- data ----------
def build_index(split):
    """Return [(folder_path, label_index), ...] for one split."""
    split_df = pd.read_csv(SPLIT_CSV)
    plan = pd.read_csv(os.path.join(LOGS, "augmentation_plan_log.csv"))
    items = []
    for folder in split_df.loc[split_df.split == split, "folder"]:
        for i, w in enumerate(WORDS):
            items.append((os.path.join(CROPPED, folder, w), i))
    for aug in plan.loc[plan.split == split, "aug_set_name"]:
        for i, w in enumerate(WORDS):
            items.append((os.path.join(AUGMENTED, aug, w), i))
    return items


class SequenceLoader(tf.keras.utils.Sequence):
    def __init__(self, items, batch_size, shuffle, seed=0):
        super().__init__()
        self.items = list(items)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.items) / self.batch_size))

    def on_epoch_end(self):
        self.order = np.arange(len(self.items))
        if self.shuffle:
            self.rng.shuffle(self.order)

    def __getitem__(self, idx):
        sel = self.order[idx*self.batch_size : (idx+1)*self.batch_size]
        X = np.empty((len(sel), N_FRAMES, FH, FW, 3), dtype=np.float32)
        y = np.zeros((len(sel), len(WORDS)), dtype=np.float32)
        for j, k in enumerate(sel):
            path, label = self.items[k]
            for f in range(N_FRAMES):
                img = cv2.imread(os.path.join(path, f"{f+1:02d}.png"))
                X[j, f] = img[:, :, ::-1] / 255.0      # BGR -> RGB, scale
            y[j, label] = 1.0
        return X, y

    @property
    def classes(self):
        return np.array([lab for _, lab in
                         (self.items[i] for i in self.order)])


# ---------- model ----------
VARIANTS = {
    "s0_cnn_lstm128": dict(filters=[32,64], rnn="lstm", units=128, bi=False,
                           dense=0, dropout=0.3, lr=1e-4, epochs=15, batch=16),
    "s1_cnn_lstm256": dict(filters=[32,64], rnn="lstm", units=256, bi=False,
                           dense=0, dropout=0.3, lr=1e-4, epochs=15, batch=16),
    "s2_cnn3_lstm128": dict(filters=[32,64,128], rnn="lstm", units=128, bi=False,
                            dense=0, dropout=0.3, lr=1e-4, epochs=15, batch=16),
    "s3_bilstm128":   dict(filters=[32,64], rnn="lstm", units=128, bi=True,
                           dense=0, dropout=0.3, lr=1e-4, epochs=15, batch=16),
    "s4_gru128":      dict(filters=[32,64], rnn="gru", units=128, bi=False,
                           dense=0, dropout=0.3, lr=1e-4, epochs=15, batch=16),
}

def build(v):
    inp = Input(shape=(N_FRAMES, FH, FW, 3))
    x = inp
    for f in v['filters']:
        x = TimeDistributed(Conv2D(f, (3,3), activation='relu', padding='same'))(x)
        if v.get('batchnorm'):
            x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(MaxPool2D((2,2), strides=2))(x)
    x = TimeDistributed(Flatten())(x)
    if v.get('frame_dense'):
        x = TimeDistributed(Dense(v['frame_dense'], activation='relu'))(x)
    x = Dropout(v['dropout'])(x)

    rnn_cls = LSTM if v['rnn'] == "lstm" else GRU
    layer = rnn_cls(v['units'], return_sequences=False)
    x = Bidirectional(layer)(x) if v['bi'] else layer(x)

    x = Dropout(v['dropout'])(x)
    if v['dense']:
        x = Dense(v['dense'], activation='tanh')(x)
    out = Dense(len(WORDS), activation='softmax')(x)
    return Model(inp, out)


# ---------- run ----------
name, seed = sys.argv[1], int(sys.argv[2])
v = VARIANTS[name]
os.makedirs(OUT, exist_ok=True)
tf.keras.utils.set_random_seed(seed)

train = SequenceLoader(build_index("train"), v['batch'], True, seed)
valid = SequenceLoader(build_index("validation"), v['batch'], False)

model = build(v)
model.compile(optimizer=Adam(learning_rate=v['lr']),
              loss='categorical_crossentropy', metrics=['accuracy'])

t0 = time.time()
hist = model.fit(train, validation_data=valid, epochs=v['epochs'], verbose=0)
elapsed = time.time() - t0

h = hist.history
pd.DataFrame(h).to_csv(f"{OUT}/history_{name}_seed{seed}.csv", index=False)

row = dict(config=name, seed=seed, params=model.count_params(),
           train_samples=len(train.items), val_samples=len(valid.items),
           epochs_run=len(h['accuracy']), batch=v['batch'],
           final_train_acc=h['accuracy'][-1], final_val_acc=h['val_accuracy'][-1],
           best_val_acc=max(h['val_accuracy']), best_val_loss=min(h['val_loss']),
           best_epoch=int(np.argmin(h['val_loss']))+1, seconds=round(elapsed,1))

with open(f"{OUT}/result_{name}_seed{seed}.json", "w") as f:
    json.dump(row, f)

print(f"{name:18s} seed {seed}  train {row['final_train_acc']:.4f}  "
      f"best_val {row['best_val_acc']:.4f}  params {row['params']:,}  ({row['seconds']}s)")