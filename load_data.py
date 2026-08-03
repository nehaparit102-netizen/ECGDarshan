import pandas as pd
import numpy as np

print("Loading data... please wait ⏳")
print("(Large files may take 1-2 minutes)")

# Load all 4 files
train    = pd.read_csv('data/mitbih_train.csv', header=None)
test     = pd.read_csv('data/mitbih_test.csv',  header=None)
normal   = pd.read_csv('data/Normal.csv',        header=None)
abnormal = pd.read_csv('data/Abnormal .csv',     header=None)

# Check shapes
print("\n--- Data Loaded Successfully! ---")
print("Train data shape:    ", train.shape)
print("Test data shape:     ", test.shape)
print("Normal data shape:   ", normal.shape)
print("Abnormal data shape: ", abnormal.shape)

# Last column is the label
print("\nTrain labels:", train.iloc[:, -1].value_counts())
print("Test labels: ", test.iloc[:, -1].value_counts())

print("\nData ready! ✅")