import numpy as np
import matplotlib.pyplot as plt

l0 = np.loadtxt("plog0.txt")
l1 = np.loadtxt("plog1.txt")

diffs = []
totDiff = 0
for i in range(len(l0)):
    diff = float(l0[i])-float(l1[i])
    diffs.append(diff)
    totDiff += diff

print("Total time difference is: ", totDiff)
plt.bar(range(len(l0)), diffs)
plt.show()
