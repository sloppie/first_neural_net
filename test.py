import numpy as np


class NN:

    def __init__(self, layers_arr):
        self.weights = []
        self.biases = []
        self.errors = np.zeros([layers_arr[-1], 1])
        i = 0

        for value in layers_arr:

            if(i <= 1):
                self.weights.append(np.random.rand(layers_arr[i + 1], value))
                self.biases.append(np.random.rand(layers_arr[i + 1], 1))

            i = i + 1

        print("Weights size: " + str(len(self.weights)))
        print("Bias size: " + str(self.biases[0].shape))

    def feed_forward(self, inputs, expected_output):
        inputted = False
        i = 0
        z = [] # This is the raw (w.a + b)
        a = [] # this represents the activations of the previous layer after squishification

        for weight in self.weights:

            if(not inputted):
                z.append(np.matmul(weight, inputs) + self.biases[i])
                a.append(1.0 / (1.0 + np.exp(-z[-1])))
                inputted = True
            else:
                z.append(np.matmul(weight, a[-1]) + self.biases[i])
                a.append(1.0 / (1.0 + np.exp(-z[-1])))

            i = i + 1

        self.calculate_error(a[-1], expected_output)
        # return [z, a]

    def calculate_error(self, a, expected_output):
        self.errors += expected_output - a[-1]

# Initialise the Neural network
example_net = NN([784, 16, 10])
expected_output = np.zeros((10, 1))
expected_output[3, 0] = 1.0
example_net.feed_forward(np.random.rand(784, 1), expected_output)
print("error: \n" + str(example_net.errors))
