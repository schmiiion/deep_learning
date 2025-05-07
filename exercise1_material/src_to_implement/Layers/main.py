from FullyConnected import FullyConnected
import numpy as np

fc = FullyConnected(6, 5)

#columns = input_size
#rows = batch_size
input_tensor = np.array([[1,1,1],[0,0,0]])
print(input_tensor)
print(input_tensor.shape)
fc.forward(input_tensor)