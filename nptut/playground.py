import numpy as np

print("Hello")

three_axis_array = np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]])

three_axis_array_simplified = np.arange(8).reshape(2, 2, 2)+1

empty_array = np.zeros((2, 2, 2), dtype=np.int8)

print(empty_array[0][1][0])

#Creates floating point array
float_array = np.linspace(0, 1, 11)

print(float_array)