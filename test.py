import numpy as np


class NN:

    def __init__(self, layers_arr):
        self.weights = []
        self.biases = []
        i = 0

        for value in layers_arr:

            if(i <= 1):
                self.weights.append(np.random.rand(layers_arr[i + 1], value))
                self.biases.append(np.random.rand(layers_arr[i + 1], 1))

            i = i + 1

        print("Weights size: " + str(len(self.weights)))
        print("Bias size: " + str(self.biases[0].shape))

    def feed_forward(self, inputs):
        inputted = False
        i = 0
        z = [] # This is the raw (w.a + b)
        a = [] # this represents the activations of the previous layer after squishification

        for weight in self.weights:

            if(not inputted):
                z.append(np.matmul(weight, inputs) + self.biases[i])
                print("Before activation: ")
                print(z[-1])
                a.append(1.0 / (1.0 + np.exp(-z[-1])))
                inputted = True
                print("Fed Forward: ")
                print(a[-1])
            else:
                z.append(np.matmul(weight, a[-1]) + self.biases[i])
                a.append(1.0 / (1.0 + np.exp(-z[-1])))
                print("Fed Forward: ")
                print(a[-1])

            i = i + 1

example_net = NN([784, 16, 10])
example_net.feed_forward(np.random.rand(784, 1))
