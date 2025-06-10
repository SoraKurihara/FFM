import matplotlib.pyplot as plt
import numpy as np

map_array = np.load("Umeda_underground.npy")

print(len(map_array[map_array == 0]))
print(len(map_array[map_array == 1]))
print(len(map_array[map_array == 2]))
print(len(map_array[map_array == 3]))

plt.imshow(map_array)
plt.show()
