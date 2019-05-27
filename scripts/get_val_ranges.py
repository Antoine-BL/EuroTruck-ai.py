import numpy as np

labels = np.load('../data-png/labels/labels.npy')

print(np.amin(labels))
print(np.amax(labels))