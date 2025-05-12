from .Base import BaseLayer
import numpy as np

class SoftMax(BaseLayer):

    def __init__(self):
        super().__init__()
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        tmp = np.exp(input_tensor)
        output = tmp / np.sum(tmp)
        return output

    def backward(self, error_tensor):
        probabilities = self.input_tensor
        jacobian = probabilities * (1 - probabilities)

        return error_tensor * jacobian