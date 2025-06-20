from .Base import BaseLayer
import numpy as np

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.weights = np.random.rand(input_size + 1, output_size)
        self._optimizer = None
        self.input_tensor = None
        self.gradient_weights = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    def forward(self, input_tensor):
        """
        :param input_tensor of shape (batch_size, input_size)
        :return: Tensor with shape (batch_size, output_size)
        """
        self.input_tensor = input_tensor
        #add bias
        batch_size = input_tensor.shape[0]
        bias_column = np.ones((batch_size, 1))
        input_with_bias = np.hstack((input_tensor, bias_column))
        #compute output
        return input_with_bias @ self.weights

    def backward(self, error_tensor):
        """
        Returns gradient wrt X (input) -> derivative of Loss wrt to x
        Updates weights wrt to gradient of weights and stores the gradient
        """
        #calculate gradient wrt input (exclude bias)
        gradient_input = error_tensor @ self.weights.T
        gradient_input = gradient_input[:, :-1]

        #calculate gradient wrt weights (including bias)
        batch_size = self.input_tensor.shape[0]
        bias_column = np.ones((batch_size, 1))
        input_with_bias = np.hstack((self.input_tensor, bias_column))
        #TODO: store input with bias so that the adding is redundant here

        self._gradient_weights = input_with_bias.T @ error_tensor

        #update weights if optimizer is set
        if self.trainable and self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        return gradient_input

