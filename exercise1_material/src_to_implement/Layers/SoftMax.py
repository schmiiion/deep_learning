from .Base import BaseLayer
import numpy as np

class SoftMax(BaseLayer):

    def __init__(self):
        super().__init__()
        self.cache = None

    def forward(self, input_tensor):
        self.cache = input_tensor
        tmp = np.exp(input_tensor)
        output = tmp / np.sum(tmp)
        return output

    def backward(self, error_tensor):
        probabilities = self.cache
        jacobian = probabilities * (1 - probabilities)

        return error_tensor * jacobian