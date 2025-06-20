import numpy as np

class CrossEntropyLoss:
    """When minimizing the KL divergence of the output of our model and the baseline distribution of the targets,
    this translates to minimizing wrt the Cross Entropy loss."""

    def __init__(self):
        self.prediction_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor

        # Get machine epsilon for the input dtype
        dtype = prediction_tensor.dtype
        eps = np.finfo(dtype).eps  # Dynamic epsilon based on dtype

        true_class_prob = np.sum(prediction_tensor * label_tensor, axis=1, keepdims=True)
        true_class_prob = np.clip(true_class_prob, eps, 1.0)  # Use computed epsilon

        loss = -np.log(true_class_prob)
        return np.sum(loss)

    def backward(self, label_tensor):
        """
        The calculated gradient of the loss function kicks of the backpropagation process.
        A small divergence from the target 1, i.e. 0.9, results in a small -log() value -> small error.
        Analogously, the 0.9 results in a small -1/0.9 = 1.111 derivative for the target value.
        A huge divergence (prediction close to 0), results in a huge -Log() value -> high error/ loss value
        The derivative of the target value in the gradient is very large too -1/ (close to 0) goes through the roof
        """

        dtype = self.prediction_tensor.dtype
        eps = np.finfo(dtype).eps

        clipped_input = np.clip(self.prediction_tensor, eps, 1.0)  # Consistent epsilon
        gradient = -label_tensor / clipped_input
        return gradient
