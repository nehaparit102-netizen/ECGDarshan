import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

print("Loading preprocessed data...")
X_train = np.load('data/X_train.npy')
X_test  = np.load('data/X_test.npy')
y_train = np.load('data/y_train.npy')
y_test  = np.load('data/y_test.npy')

# Convert labels to categorical
y_train_cat = to_categorical(y_train, num_classes=5)
y_test_cat  = to_categorical(y_test,  num_classes=5)

print("Building CNN model...")

# Build 1D CNN Model
model = models.Sequential([
    # Block 1
    layers.Conv1D(64, 3, activation='relu', input_shape=(187, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),
    layers.Dropout(0.2),

    # Block 2
    layers.Conv1D(128, 3, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),
    layers.Dropout(0.2),

    # Block 3
    layers.Conv1D(256, 3, activation='relu'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.4),

    # Fully connected layers
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(5, activation='softmax')  # 5 classes
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

print("\nTraining model... this will take 5-10 minutes ⏳")

# Train model
history = model.fit(
    X_train, y_train_cat,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# Evaluate model
print("\nEvaluating model...")
loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")

# Classification report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes,
      target_names=['Normal','Supra','Ventri','Fusion','Unknown']))

# Save model
model.save('data/ecgdarshan_model.h5')
print("\nModel saved! ✅")

# Plot accuracy graph
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('data/training_results.png')
plt.show()
print("Graph saved! ✅")