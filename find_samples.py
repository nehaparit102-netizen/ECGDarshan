import numpy as np

y_test = np.load('data/y_test.npy')

# Find all Normal samples (class 0)
normal_samples = []
for i in range(len(y_test)):
    if y_test[i] == 0.0:
        normal_samples.append(i)

print("Normal ECG sample numbers:")
print(normal_samples[:20])

# Find Abnormal samples
abnormal_samples = []
for i in range(len(y_test)):
    if y_test[i] != 0.0:
        abnormal_samples.append(i)

print("\nAbnormal ECG sample numbers:")
print(abnormal_samples[:20])