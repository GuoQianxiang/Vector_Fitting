import numpy as np

A = np.array([2+3j, 1-2j, 4+1j, 3-4j, 2+1j])
sorted_A = np.sort_complex(A)
sorted_B = sorted(A, key=abs)

print(sorted_A)
print(sorted_B)