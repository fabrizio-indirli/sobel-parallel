import numpy as np
import matplotlib.pyplot as plt

l0 = np.loadtxt("plog0.txt") #serial
l1 = np.loadtxt("plog1.txt") #parallelized

diffs = []
speedups = []
totDiff = 0
for i in range(len(l0)):
    diff = float(l0[i])-float(l1[i])
    diffs.append(diff)
    totDiff += diff
    speedup = float(l0[i])/float(l1[i])
    speedups.append(speedup)

print("Total time difference is: ", totDiff, " s")
print("Average speedup is: ", sum(speedups)/float(len(speedups)))
print("Total speedup is: ", sum(l0)/sum(l1))

plt.bar(range(len(l0)), diffs)
plt.xlabel("Input file id")
plt.ylabel("Execution time difference")
plt.title("Execution time differences between serial and parallelized implementations")
plt.show()

plt.bar(range(len(l0)), speedups)
plt.title("Speedup for each input file")
plt.xlabel("Input file id")
plt.ylabel("Execution speedup")
plt.show()


