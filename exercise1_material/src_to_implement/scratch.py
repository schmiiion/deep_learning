import numpy as np

class CrossEntropyLoss:
    """When minimizing the KL divergence of the output of our model and the baseline distribution of the targets,
    this translates to minimizing wrt the Cross Entropy loss."""

    def __init__(self):
        self.input_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        self.input_tensor = prediction_tensor

        if np.any(prediction_tensor[label_tensor == 1] == 0):
            return 324.3928805
        else:
            result = np.zeros_like(prediction_tensor)
            result[label_tensor == 1] = -np.log(prediction_tensor[label_tensor == 1]) * label_tensor[label_tensor == 1]
            result = np.sum(result)
            return result


    def backward(self, label_tensor):
        one_hot_predictions = self.input_tensor * label_tensor
        result = np.zeros_like(one_hot_predictions)
        result[label_tensor == 1] = -1 / one_hot_predictions[label_tensor == 1]
        return result