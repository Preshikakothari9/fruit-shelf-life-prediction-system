"""
FruitFresh training pipeline.

Trains a MobileNetV2-based classifier on a directory of fruit images organised as:

    dataset_root/
        fresh_apple/    *.jpg
        rotten_apple/   *.jpg
        fresh_banana/   *.jpg
        rotten_banana/  *.jpg
        ...

Outputs:
    app/ml_model/fruit_model.h5
    app/ml_model/classes.json

Usage:
    python train.py <path_to_dataset_directory>
"""

import json
import os
import sys

IMG_SIZE = 224
BATCH_SIZE = 32
HEAD_EPOCHS = 12
FINETUNE_EPOCHS = 8
MIN_IMAGES_PER_CLASS = 10


def _validate_dataset(dataset_dir: str):
    """Sanity-check the dataset directory before spending time on training."""
    if not os.path.isdir(dataset_dir):
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")

    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    classes = []
    for entry in sorted(os.listdir(dataset_dir)):
        sub = os.path.join(dataset_dir, entry)
        if not os.path.isdir(sub):
            continue
        n = sum(
            1 for f in os.listdir(sub)
            if os.path.splitext(f)[1].lower() in valid_ext
        )
        if n < MIN_IMAGES_PER_CLASS:
            print(f"  [skip] '{entry}' has only {n} images (need >= {MIN_IMAGES_PER_CLASS})")
            continue
        classes.append((entry, n))

    if len(classes) < 2:
        raise SystemExit(
            "Need at least 2 class folders with images. "
            "Expected layout: dataset/<class_name>/*.jpg"
        )

    print(f"Found {len(classes)} usable class folders:")
    for name, n in classes:
        print(f"  - {name:30s} {n} images")
    return [c[0] for c in classes]


def start_training(dataset_dir: str):
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    _validate_dataset(dataset_dir)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "app", "ml_model")
    os.makedirs(model_dir, exist_ok=True)

    # MobileNetV2 expects pixel values in [-1, 1]; preprocess_input handles that.
    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.20,
        shear_range=0.10,
        horizontal_flip=True,
        brightness_range=(0.7, 1.3),
        validation_split=0.2,
    )
    val_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,
    )

    train_flow = train_gen.flow_from_directory(
        dataset_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42,
    )
    val_flow = val_gen.flow_from_directory(
        dataset_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42,
    )

    # class_indices is {label: idx}. We need the inverse, ordered by idx.
    inv = {v: k for k, v in train_flow.class_indices.items()}
    class_list = [inv[i] for i in range(len(inv))]
    with open(os.path.join(model_dir, "classes.json"), "w", encoding="utf-8") as f:
        json.dump({"classes": class_list}, f, indent=2)
    print(f"Classes ({len(class_list)}): {class_list}")

    # Balance classes so a tiny 'rotten_kiwi' folder doesn't get ignored.
    y_train = train_flow.classes
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(len(class_list)),
        y=y_train,
    )
    class_weights = {i: float(w) for i, w in enumerate(weights)}

    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(len(class_list), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6
        ),
    ]

    print("\n=== Stage 1: training classifier head ===")
    model.fit(
        train_flow,
        epochs=HEAD_EPOCHS,
        validation_data=val_flow,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    print("\n=== Stage 2: fine-tuning top of MobileNetV2 ===")
    base_model.trainable = True
    for layer in base_model.layers[:-40]:
        layer.trainable = False
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.fit(
        train_flow,
        epochs=FINETUNE_EPOCHS,
        validation_data=val_flow,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    out_path = os.path.join(model_dir, "fruit_model.h5")
    model.save(out_path)
    print(f"\nTraining complete. Model saved to {out_path}")
    print("Restart the Flask app so the new weights are loaded.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        start_training(sys.argv[1])
    else:
        print("Usage: python train.py <path_to_dataset_directory>")
