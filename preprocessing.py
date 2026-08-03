import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

print("Loading data...")
train = pd.read_csv('data/mitbih_train.csv', header=None)
test  = pd.read_csv('data/mitbih_test.csv',  header=None)

# Check exact number of columns
print("Total columns in train:", train.shape[1])

# Separate signals and labels
X_train = train.iloc[:, :-1].values
y_train = train.iloc[:, -1].values

X_test  = test.iloc[:, :-1].values
y_test  = test.iloc[:, -1].values

# Check signal length
signal_length = X_train.shape[1]
print("Signal length:", signal_length)

# Normalize the signals
print("Normalizing signals...")
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Reshape for CNN using actual signal length
X_train = X_train.reshape(-1, signal_length, 1)
X_test  = X_test.reshape(-1, signal_length, 1)

print("X_train shape:", X_train.shape)
print("X_test shape: ", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape: ", y_test.shape)

# Save processed data
np.save('data/X_train.npy', X_train)
np.save('data/X_test.npy',  X_test)
np.save('data/y_train.npy', y_train)
np.save('data/y_test.npy',  y_test)

print("\nPreprocessing done! ✅")
print("Files saved in data folder!")