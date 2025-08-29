import numpy as np


class NeuralNetWork:
    weights = None
    biases = None

    outputs = None
    activations = None

    lr = None

    def __init__(self, shape, learn_rate=0.1):
        assert len(shape) > 1
        self.weights = []
        self.biases = []
        self.lr = learn_rate
        for i in range(len(shape) - 1):
            self.weights.append(np.random.random((shape[i + 1], shape[i])))
            self.biases.append(np.random.random((shape[i + 1], 1)))

    def forward(self, x):
        assert x.shape[0] == self.weights[0].shape[1]
        self.outputs = []
        self.activations = []
        self.activations.append(x)
        self.outputs.append(x)
        for i in range(len(self.weights)):
            output = np.dot(self.weights[i], x) + self.biases[i]
            x = output
            self.outputs.append(output)
            self.activations.append(NeuralNetWork.sigmoid(output))

        print(self.activations[-1])
        return self.activations[-1]

    def backward(self, y):
        assert y.shape[0] == self.activations[-1].shape[0]
        self.errors = []
        #         loss = 0.5*np.sum((self.activations[-1]-y)**2)
        g_loss = self.activations[-1] - y
        error = g_loss * NeuralNetWork.sigmoid_derivative(self.outputs[-1])
        for i in range(len(self.weights) - 1, -1, -1):
            g_w = self.lr * np.dot(error, self.activations[i].T)
            g_b = self.lr * error
            #             print(error)
            #             print(NeuralNetWork.sigmoid_derivative(self.outputs[i]))
            error = np.dot(self.weights[i].T, error) * NeuralNetWork.sigmoid_derivative(
                self.outputs[i]
            )
            self.weights[i] -= g_w
            self.biases[i] -= g_b

    def set_weights(self, weights):
        assert len(weights) == len(self.weights)
        for i in range(len(weights)):
            self.weights[i] = weights[i]
        return self

    def set_biases(self, biases):
        assert len(biases) == len(self.biases)
        for i in range(len(biases)):
            self.biases[i] = biases[i]
        return self

    @classmethod
    def sigmoid(cls, x):
        return 1 / (1 + np.exp(-x))

    @classmethod
    def sigmoid_derivative(cls, x):
        return cls.sigmoid(x) * cls.sigmoid(1 - x)
