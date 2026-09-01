import os, sys, time, json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Flatten

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

W    = "/home/admins/rebuild_workspace"
DATA = os.path.join(W, "06_dataset_final")
OUT  = os.path.join(W, "models_pretrained")
word_classes = ['bat','cup','drop','eat','fish','hot','jump','milk','pen','red']

BACKBONES = {
    "mobilenetv2": dict(
        cls=tf.keras.applications.MobileNetV2,
        preprocess=tf.keras.applications.mobilenet_v2.preprocess_input),
    "vgg16": dict(
        cls=tf.keras.applications.VGG16,
        preprocess=tf.keras.applications.vgg16.preprocess_input),
    "efficientnetb0": dict(
        cls=tf.keras.applications.EfficientNetB0,
        preprocess=tf.keras.applications.efficientnet.preprocess_input),
}

name, seed = sys.argv[1], int(sys.argv[2])
# cfg = BACKBONES[name]
os.makedirs(OUT, exist_ok=True)

tf.keras.utils.set_random_seed(seed)

# backbone-specific preprocessing, not rescale=1/255
# IDG = ImageDataGenerator(preprocessing_function=cfg['preprocess'])
# IDG = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)

# train, valid = gen("Train", True), gen("Validation", False)

# base = cfg['cls'](include_top=False, weights='imagenet', input_shape=(224,224,3))
# base.trainable = False

# inp = Input(shape=(224,224,3))
# x = base(inp, training=False)
# x = GlobalAveragePooling2D()(x)
# x = Dropout(0.2)(x)
# out = Dense(10, activation='softmax')(x)
# model = Model(inp, out)

# model.compile(optimizer=Adam(learning_rate=0.0001),
#               loss='categorical_crossentropy', metrics=['accuracy'])

VARIANTS = {
    "m0_baseline":     dict(head="gap",     layer=None,               unfreeze=0,  extra_dense=0, lr=1e-4),
    "m1_flatten":      dict(head="flatten", layer=None,               unfreeze=0,  extra_dense=0, lr=1e-4),
    "m2_midlayer":     dict(head="gap",     layer="block_6_expand_relu", unfreeze=0, extra_dense=0, lr=1e-4),
    "m3_densehead":    dict(head="gap",     layer=None,               unfreeze=0,  extra_dense=256, lr=1e-4),
    "m4_unfreeze20":   dict(head="gap",     layer=None,               unfreeze=20, extra_dense=0, lr=1e-4),
    "m5_lr1e3":        dict(head="gap",     layer=None,               unfreeze=0,  extra_dense=0, lr=1e-3),
    "m6_flat_lr1e3":      dict(head="flatten", layer=None, unfreeze=0,  extra_dense=0,   lr=1e-3),
    "m7_flat_dense":      dict(head="flatten", layer=None, unfreeze=0,  extra_dense=256, lr=1e-4),
    "m8_flat_dense_lr1e3":dict(head="flatten", layer=None, unfreeze=0,  extra_dense=256, lr=1e-3),
    "m9_flat_unfreeze20": dict(head="flatten", layer=None, unfreeze=20, extra_dense=0,   lr=1e-4),
    "m10_flat_dense_drop": dict(head="flatten", layer=None, unfreeze=0, extra_dense=256, lr=1e-4, dropout=0.5),
    "m11_flat_dense64":    dict(head="flatten", layer=None, unfreeze=0, extra_dense=64,  lr=1e-4, dropout=0.2),
    "m12_efficientnet":    dict(head="flatten", layer=None, unfreeze=0, extra_dense=256, lr=1e-4, dropout=0.2, 
                                backbone="efficientnetb0"),
    "m13_eff_unfreeze20": dict(head="flatten", layer=None, unfreeze=20, extra_dense=256, lr=1e-4,
                               dropout=0.2, backbone="efficientnetb0"),
    "m14_eff_epochs40":   dict(head="flatten", layer=None, unfreeze=0,  extra_dense=256, lr=1e-4,
                               dropout=0.2, backbone="efficientnetb0", epochs=40),
    "m15_eff_volume":     dict(head="flatten", layer=None, unfreeze=0,  extra_dense=256, lr=1e-4,
                               dropout=0.2, backbone="efficientnetb0", data="06v_dataset_volume"),
    "m16_vgg16":          dict(head="flatten", layer=None, unfreeze=0, extra_dense=256, lr=1e-4,
                               dropout=0.2, backbone="vgg16"),
    "m17_vgg16_epochs40": dict(head="flatten", layer=None, unfreeze=0, extra_dense=256, lr=1e-4,
                               dropout=0.2, backbone="vgg16", epochs=40),
    "m18_eff_epochs80":   dict(head="flatten", layer=None, unfreeze=0, extra_dense=256, lr=1e-4,
                               dropout=0.2, backbone="efficientnetb0", epochs=80),
    "m19_vgg16_epochs80": dict(head="flatten", layer=None, unfreeze=0, extra_dense=256, lr=1e-4,
                               dropout=0.2, backbone="vgg16", epochs=80),
}

v = VARIANTS[name]
v.setdefault('dropout', 0.2)
v.setdefault('backbone', 'mobilenetv2')
v.setdefault('epochs', 15)
v.setdefault('data', "06_dataset_final")

IDG = ImageDataGenerator(preprocessing_function=BACKBONES[v['backbone']]['preprocess'])

# def gen(sub, shuffle):
#     return IDG.flow_from_directory(os.path.join(DATA, sub), target_size=(224,224),
#         classes=word_classes, batch_size=16, class_mode='categorical', shuffle=shuffle)

def gen(sub, shuffle):
    return IDG.flow_from_directory(os.path.join(W, v['data'], sub), target_size=(224,224),
        classes=word_classes, batch_size=16, class_mode='categorical', shuffle=shuffle)

train, valid = gen("Train", True), gen("Validation", False)



# base = tf.keras.applications.MobileNetV2(
#     include_top=False, weights='imagenet', input_shape=(224,224,3))
base = BACKBONES[v['backbone']]['cls'](
    include_top=False, weights='imagenet', input_shape=(224,224,3))



if v['layer']:
    base = Model(base.input, base.get_layer(v['layer']).output)

base.trainable = False
if v['unfreeze'] > 0:
    base.trainable = True
    for layer in base.layers[:-v['unfreeze']]:
        layer.trainable = False

inp = Input(shape=(224,224,3))
x = base(inp, training=(v['unfreeze'] > 0))
x = GlobalAveragePooling2D()(x) if v['head'] == "gap" else Flatten()(x)
x = Dropout(v['dropout'])(x)
if v['extra_dense']:
    x = Dense(v['extra_dense'], activation='tanh')(x)
out = Dense(10, activation='softmax')(x)
model = Model(inp, out)

model.compile(optimizer=Adam(learning_rate=v['lr']),
              loss='categorical_crossentropy', metrics=['accuracy'])



trainable = int(np.sum([np.prod(w.shape) for w in model.trainable_weights]))
total = model.count_params()

t0 = time.time()
# hist = model.fit(train, validation_data=valid, epochs=15, verbose=0)
hist = model.fit(train, validation_data=valid, epochs=v['epochs'], verbose=0)
elapsed = time.time() - t0

h = hist.history
pd.DataFrame(h).to_csv(f"{OUT}/history_{name}_seed{seed}.csv", index=False)

row = dict(config=name, seed=seed, total_params=total, trainable_params=trainable,
           epochs_run=len(h['accuracy']), train_samples=train.samples,
           final_train_acc=h['accuracy'][-1], final_val_acc=h['val_accuracy'][-1],
           best_val_acc=max(h['val_accuracy']), best_val_loss=min(h['val_loss']),
           best_epoch=int(np.argmin(h['val_loss']))+1, seconds=round(elapsed,1))

with open(f"{OUT}/result_{name}_seed{seed}.json", "w") as f:
    json.dump(row, f)

print(f"{name:15s} seed {seed}  train {row['final_train_acc']:.4f}  "
      f"best_val {row['best_val_acc']:.4f}  trainable {trainable:,}  ({row['seconds']}s)")