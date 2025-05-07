from .Base import BaseLayer
import numpy as np

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.weights = np.random.rand(output_size, input_size)
        self._optimizer = None
        self.input_tensor = None
        self.gradient_weights = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    def forward(self, input_tensor):
        """
        :param input_tensor of shape (input_size, batch_size)
        :return: output_tensor of shape (output_size, batch_size)
        """
        self.input_tensor = input_tensor
        return self.weights @ input_tensor

    def backward(self, error_tensor):
        """
        Returns gradient wrt X (input) -> derivative of Loss wrt to x
        Updates weights wrt to gradient of weights
        """
        # weights: (output_size x input_size);
        # error_tensor: (output_size x 1)
        # -> propagate vector back
        error_tensor_previous_layer = self.weights.T @ error_tensor

        # outer product
        grad_wrt_weights = error_tensor @ self.input_tensor.T
        self.gradient_weights = grad_wrt_weights

        if self.trainable and self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, grad_wrt_weights)

        return error_tensor_previous_layer

    @property
    def gradient_weights(self):
        return self.gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value
