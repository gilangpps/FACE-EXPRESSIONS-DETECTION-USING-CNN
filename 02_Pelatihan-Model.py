import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ===============================
# Load Dataset
# ===============================

data_dir = 'manual_emotion_dataset'
img_size = (48, 48)

emotions = sorted(os.listdir(data_dir))  # misal: ['angry', 'happy', 'sad', 'surprise']
label_map = {emotion: idx for idx, emotion in enumerate(emotions)}

print("🔍 Loading dataset...")

X = []
y = []

for emotion in emotions:
    folder_path = os.path.join(data_dir, emotion)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, img_size)
        img = img.astype('float32') / 255.0
        X.append(img)
        y.append(label_map[emotion])

X = np.array(X)
y = np.array(y)

X = np.expand_dims(X, -1)  # (samples, 48, 48, 1)
y = to_categorical(y, num_classes=len(emotions))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dataset loaded. Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# ===============================
# Model CNN
# ===============================

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size[0], img_size[1], 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(emotions), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ===============================
# Training
# ===============================

history = model.fit(X_train, y_train,
                    epochs=30,
                    batch_size=32,
                    validation_data=(X_test, y_test))

model.save("emotion_detection_model.h5")
print("💾 Model berhasil disimpan: emotion_detection_model.h5")

# ===============================
# Visualisasi Accuracy & Loss
# ===============================

plt.figure(figsize=(14,6))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# ===============================
# Confusion Matrix
# ===============================

y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=emotions)

plt.figure(figsize=(8,8))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Emotion Detection")
plt.show()
