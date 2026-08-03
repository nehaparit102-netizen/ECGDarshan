
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

print("Loading data...")
X_train = np.load('data/X_train.npy')
X_test  = np.load('data/X_test.npy')
y_train = np.load('data/y_train.npy')
y_test  = np.load('data/y_test.npy')

# Fix class imbalance with SMOTE
print("Applying SMOTE... please wait 3-5 minutes ⏳")
X_flat = X_train.reshape(X_train.shape[0], -1)
smote  = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_flat, y_train)
X_res = X_res.reshape(-1, 187, 1)

print("Class distribution after SMOTE:")
unique, counts = np.unique(y_res, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Class {int(u)}: {c} samples")

# Convert labels
y_train_cat = to_categorical(y_res,  num_classes=5)
y_test_cat  = to_categorical(y_test, num_classes=5)

print("\nBuilding improved CNN model...")

# Improved CNN model
inputs = Input(shape=(187, 1))

# Block 1
x = layers.Conv1D(64, 5, activation='relu', padding='same')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Conv1D(64, 5, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(2)(x)
x = layers.Dropout(0.2)(x)

# Block 2
x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(2)(x)
x = layers.Dropout(0.2)(x)

# Block 3
x = layers.Conv1D(256, 3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.3)(x)

# Fully connected
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(5, activation='softmax')(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.00001,
    verbose=1
)

print("\nTraining... this will take 15-20 minutes ⏳")

history = model.fit(
    X_res, y_train_cat,
    epochs=30,
    batch_size=128,
    validation_split=0.1,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Evaluate
print("\nEvaluating...")
loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")

y_pred         = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred_classes,
    target_names=['Normal','Supra','Ventri',
                  'Fusion','Unknown'],
    zero_division=0
))

# Save improved model
model.save('data/ecgdarshan_model_v2.keras')
print("\nImproved model saved! ✅")

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],
         'b-o', label='Train')
plt.plot(history.history['val_accuracy'],
         'r-o', label='Validation')
plt.title('ECGDarshan V2 - Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],
         'b-o', label='Train')
plt.plot(history.history['val_loss'],
         'r-o', label='Validation')
plt.title('ECGDarshan V2 - Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.suptitle('ECGDarshan Improved Model Results',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('data/training_results_v2.png', dpi=150)
plt.show()
print("Graph saved! ✅")