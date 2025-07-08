import os
import zipfile
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

# ----------------------------- Data Preparation -----------------------------
def setup_kaggle_api(username, key):
    os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
    with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'w') as f:
        f.write(f'{"username":"{username}","key":"{key}"}')
    os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)

def download_and_extract_dataset(dataset_name, extract_to="dataset"):
    os.system(f'kaggle datasets download -d {dataset_name}')
    zip_path = dataset_name.split("/")[-1] + ".zip"
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

def find_csv(base_path, filename="fer2013.csv"):
    for root, dirs, files in os.walk(base_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

def load_and_prepare_data(csv_path):
    INTERESTED_LABELS = [0, 3, 4, 5, 6]  # Angry, Happy, Sad, Surprise, Neutral
    df = pd.read_csv(csv_path)
    df = df[df['emotion'].isin(INTERESTED_LABELS)]
    label_map = {label: idx for idx, label in enumerate(INTERESTED_LABELS)}
    df['emotion'] = df['emotion'].map(label_map)

    X = np.array([np.fromstring(x, dtype=int, sep=' ') for x in df['pixels']])
    X = X.reshape((-1, 48, 48, 1)).astype('float32') / 255.0
    y = to_categorical(df['emotion'], num_classes=5)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1111, stratify=y_temp, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

# ----------------------------- Build Model -----------------------------
def build_model():
    model = Sequential([
        Conv2D(64, (3, 3), activation='elu', padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(0.001), input_shape=(48, 48, 1)),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='elu', padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Conv2D(128, (3, 3), activation='elu', padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='elu', padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.35),

        Conv2D(256, (3, 3), activation='elu', padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='elu', padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),

        Conv2D(512, (3, 3), activation='elu', padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.5),

        Flatten(),
        Dense(256, activation='elu', kernel_regularizer=l2(0.001), kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.6),

        Dense(5, activation='softmax')
    ])

    optimizer = Nadam(learning_rate=0.0003)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ----------------------------- Callbacks -----------------------------
def create_callbacks(filepath):
    return [
        ModelCheckpoint(filepath, monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1)
    ]

# ----------------------------- Plot Training History -----------------------------
def plot_training_history(history):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()

# ----------------------------- Model Evaluation -----------------------------
def evaluate_model(model, x_val, y_val):
    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")

def plot_confusion(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=labels))

# ----------------------------- Main Execution -----------------------------
if __name__ == "__main__":
    setup_kaggle_api("YOUR_KAGGLE_USERNAME", "YOUR_KAGGLE_API_KEY")
    download_and_extract_dataset("ashishpatel26/facial-expression-recognitionferchallenge")

    csv_path = find_csv("dataset")
    if not csv_path:
        raise FileNotFoundError("fer2013.csv file not found!")

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data(csv_path)
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.2,
        shear_range=0.15,
        horizontal_flip=True
    )
    datagen.fit(X_train)

    model = build_model()
    callbacks = create_callbacks("saved_models/best_emotion_model.keras")

    os.makedirs("saved_models", exist_ok=True)

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=64, shuffle=True),
        validation_data=(X_val, y_val),
        epochs=100,
        callbacks=callbacks,
        verbose=1
    )

    model.save("saved_models/final_emotion_model.keras")
    print("\u2705 Model training complete and saved!")

    plot_training_history(history)
    evaluate_model(model, X_val, y_val)

    y_pred = np.argmax(model.predict(X_val), axis=1)
    y_true = np.argmax(y_val, axis=1)
    labels = ['Angry', 'Happy', 'Sad', 'Surprise', 'Neutral']
    plot_confusion(y_true, y_pred, labels)
