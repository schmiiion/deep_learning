import copy

class NeuralNetwork:

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = [] #loss value for each iteration after calling train
        self.layers = [] #holds the architecture
        self.data_layer = None #input data + labels
        self.loss_layer = None #layer providing loss and prediction
        self.current_label = None

    def forward(self):
        input_tensor, label_tensor = self.data_layer.next()
        self.current_label = label_tensor

        #pass through all layers
        output = input_tensor
        for layer in self.layers:
            output = layer.forward(output)

        #calculate loss
        loss = self.loss_layer.forward(output, label_tensor)
        return loss

    def backward(self):
        error_tensor = self.loss_layer.backward(self.current_label)

        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            optimizer_copy = copy.deepcopy(self.optimizer)
            layer.optimizer = optimizer_copy
        self.layers.append(layer)

    def train(self, iterations):
        for _ in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):
        output = input_tensor
        for layer in self.layers:
            output = layer.forward(output)
        return output
