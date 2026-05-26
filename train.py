import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------- CONFIGURATION ---------------- #

CONFIG = {

    "image_size": (128, 128),

    "batch_size": 32,

    "epochs": 35,

    "learning_rate": 0.001,

    "num_classes": 3,

    "dataset_path": r"C:\Users\NALUBALA ARJUN\Dropbox\PC\Desktop\Orthodontic1\RawImage\TrainingData",

    "model_save_path": "models/tsynet_ortho_predictor.h5"
}

# Create model folder if not exists
os.makedirs(
    os.path.dirname(CONFIG["model_save_path"]),
    exist_ok=True
)

# ---------------- CREATE CNN MODEL ---------------- #

def create_tsynet_model(input_shape, num_classes):

    model = Sequential([

        tf.keras.layers.Input(shape=input_shape),

        # First CNN Block
        Conv2D(
            64,
            (3, 3),
            activation='relu',
            padding='same'
        ),

        MaxPooling2D((2, 2)),

        # Second CNN Block
        Conv2D(
            64,
            (3, 3),
            activation='relu',
            padding='same'
        ),

        MaxPooling2D((2, 2)),

        # Third CNN Block
        Conv2D(
            128,
            (3, 3),
            activation='relu',
            padding='same'
        ),

        MaxPooling2D((2, 2)),

        # Flatten Layer
        Flatten(),

        # Dense Layer
        Dense(
            64,
            activation='relu'
        ),

        # Dropout
        Dropout(0.3),

        # Output Layer
        Dense(
            num_classes,
            activation='softmax'
        )
    ])

    # Compile Model
    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=CONFIG["learning_rate"]
        ),

        loss='categorical_crossentropy',

        metrics=['accuracy']
    )

    return model

# ---------------- DATA PREPARATION ---------------- #

def prepare_data():

    train_datagen = ImageDataGenerator(

        rescale=1./255,

        rotation_range=15,

        zoom_range=0.1,

        width_shift_range=0.1,

        height_shift_range=0.1,

        brightness_range=[0.8, 1.2],

        horizontal_flip=False,

        validation_split=0.2
    )

    # Training Generator
    train_generator = train_datagen.flow_from_directory(

        CONFIG["dataset_path"],

        target_size=CONFIG["image_size"],

        color_mode='grayscale',

        batch_size=CONFIG["batch_size"],

        class_mode='categorical',

        subset='training'
    )

    # Validation Generator
    val_generator = train_datagen.flow_from_directory(

        CONFIG["dataset_path"],

        target_size=CONFIG["image_size"],

        color_mode='grayscale',

        batch_size=CONFIG["batch_size"],

        class_mode='categorical',

        subset='validation'
    )

    return train_generator, val_generator

# ---------------- TRAIN MODEL ---------------- #

def train_model():

    # Input Shape
    input_shape = CONFIG["image_size"] + (1,)

    # Create Model
    model = create_tsynet_model(
        input_shape,
        CONFIG["num_classes"]
    )

    # Prepare Dataset
    train_generator, val_generator = prepare_data()

    # Show Class Labels
    print("\nClass Indices:")
    print(train_generator.class_indices)

    # Callbacks
    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(

            filepath=CONFIG["model_save_path"],

            save_best_only=True,

            monitor='val_accuracy',

            mode='max'
        ),

        tf.keras.callbacks.EarlyStopping(

            monitor='val_loss',

            patience=5,

            restore_best_weights=True
        )
    ]

    # Train Model
    history = model.fit(

        train_generator,

        epochs=CONFIG["epochs"],

        validation_data=val_generator,

        callbacks=callbacks
    )

    # Plot Training History
    plot_training_history(history)

    return model

# ---------------- PLOT TRAINING HISTORY ---------------- #

def plot_training_history(history):

    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)

    plt.plot(
        history.history['accuracy'],
        label='Training Accuracy'
    )

    plt.plot(
        history.history['val_accuracy'],
        label='Validation Accuracy'
    )

    plt.title('Model Accuracy')

    plt.xlabel('Epoch')

    plt.ylabel('Accuracy')

    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)

    plt.plot(
        history.history['loss'],
        label='Training Loss'
    )

    plt.plot(
        history.history['val_loss'],
        label='Validation Loss'
    )

    plt.title('Model Loss')

    plt.xlabel('Epoch')

    plt.ylabel('Loss')

    plt.legend()

    plt.savefig("training_history.png")

    plt.close()

# ---------------- MAIN ---------------- #

if __name__ == "__main__":

    print("\nStarting TSYNET Orthodontic Growth Predictor Training...\n")

    model = train_model()

    print("\nTraining Complete!")

    print(f"\nModel saved to: {CONFIG['model_save_path']}")