from .Base import BaseLayer
import numpy as np

class SoftMax(BaseLayer):

    def __init__(self):
        super().__init__()
        self.probabilities = None #jacobian depends on probabilities and not on the original inputs -> input_tensor needn't be stored

    def forward(self, input_tensor):
        #Numerical stability: Just putting the input_tensor into np.exp() can result in overflow -> inf or underflow -> 0
        #Thus i am shifting the input by the maximum value of the vector
        #==>Softmax is insensitive to additive shifting of input values
        shifted_input = input_tensor - np.max(input_tensor, axis=1, keepdims=True) #keepdims to enable broadcasting and keep second dim
        tmp = np.exp(shifted_input)
        self.probabilities = tmp / np.sum(tmp, axis=1, keepdims=True)
        return self.probabilities

    def backward(self, error_tensor):
        #implicit computation of the jacobian matrix withput explicitely computing it
        batch_sum = np.sum(error_tensor * self.probabilities, axis=1, keepdims=True)
        grad = self.probabilities * (error_tensor - batch_sum)
        return grad