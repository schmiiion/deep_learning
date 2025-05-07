import numpy as np


#INPUT Tensor
batch_size = 2
input_size = 5

output_size = 4

input_tensor = np.ones((batch_size, input_size))
weights = np.ones((output_size, input_size))


result = input_tensor @ weights
print(result)