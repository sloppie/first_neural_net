import numpy as np
import copy
import random


class NN:

    def __init__(self, layers_arr, learning_rate):
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = [] # array of biases with respect to the number of layers
        self.errors = np.zeros([layers_arr[-1], 1], dtype="float") # stores the sum of errors
        self.layers_size = layers_arr # keeps the sizes of the layers just in case they are needed for recomputing
        i = 0

        for value in layers_arr:

            if(i <= 1):
                self.weights.append(np.random.randn(layers_arr[i + 1], value)) # matrix of layers[i +1] by layers[i]
                self.biases.append(np.random.randn(layers_arr[i + 1], 1)) # starts from the ssecond layer generating a bias for each layer neuron

            i = i + 1

    def feed_forward(self, inputs, expected_output=None):
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
        
        new_list = [copy.copy(z), copy.copy(a)]

        if(expected_output is not None):
            self.calculate_cost(a[-1], expected_output)

        return new_list

    def calculate_cost(self, a, expected_output):
        self.errors += a - expected_output

    def calculate_av_cost(self, epoch_size):
        self.errors = self.errors / float(epoch_size)
        average_error = np.copy(self.errors)
        self.errors[:, :] = 0 # resets all the values to zeros
    
        return average_error

    def adjust_weights(self, current_err, layer_count, last_output, inputs):

        previous_activation = np.ones((1,1)) # dummy initialisation
        
        if(layer_count - 1 < -len(self.weights)):
            previous_activation = inputs
        else:
            previous_activation = last_output[-1][layer_count - 1]

        err_copy = np.copy(current_err)
        err_copy = (self.learning_rate * current_err) - previous_activation.transpose()
        self.weights[layer_count] = self.weights[layer_count] - err_copy
        # print("Weight nudged")
        # print(self.weights[layer_count])

    def adjust_biases(self, current_err, layer_count):
        err_copy = np.copy(current_err)
        err_copy = self.learning_rate * err_copy
        # bp = np.matmul(self.biases[layer_count], err_copy)
        bp = self.biases[layer_count] * err_copy
        self.biases[layer_count] = self.biases[layer_count] - bp

    def backpropagate_error(self, average_error, last_output, inputs):
        initial_err = False
        layer_count = -1

        for i in self.weights:

            if(not initial_err):
                current_err = average_error
                initial_err = True
            else:
                weight_copy = np.copy(self.weights[layer_count + 1])
                weight_copy = weight_copy.transpose()
                current_err = np.matmul(self.weights[layer_count + 1].transpose(), current_err) * sigmoid_prime(last_output[-1][layer_count])

            self.adjust_weights(current_err, layer_count, last_output, inputs)
            self.adjust_biases(current_err, layer_count)

            layer_count = layer_count - 1

    def SGD(self, training_set, epoch_size):
        count = 0

        for data in training_set:
            last_output = self.feed_forward(data[0], data[1])
            
            if count == epoch_size - 1: # handles backpropagation of the error
                average_error = self.calculate_av_cost(epoch_size)
                count = 0
                self.backpropagate_error(average_error, last_output, data[0])
            else:
                count = count + 1

def sigmoid(z):
    return 1.0 + (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) - (1 - sigmoid(z))


# Initialise the Neural network

# create test data and corresponding expected output
training_data = []
for i in range(10000):
    
    if i % 4 == 0:
        data = [
                np.array([[0.0], [0.0]]),
                np.array([[0.0]])
            ]
    elif i % 4 == 1:
        data = [
                np.array([[0.0], [1.0]]),
                np.array([[1.0]])
            ]
    elif i % 4 == 2:
        data = [
                np.array([[1.0], [0.0]]),
                np.array([[1.0]])
            ]
    elif i % 4 == 3:
        data = [
                np.array([[1.0], [1.0]]),
                np.array([[0.0]])
            ]

    # print(data)
    training_data.append(data)

# initialise neural net
XOR = NN([2, 2, 1], 0.15)

print(XOR.feed_forward(np.array([[1.0], [0.0]]), np.array([[1.0]]))[-1][-1])
print("Error:")
print(XOR.errors)

training_data = random.sample(training_data, len(training_data))

XOR.SGD(training_data, 100)

a = XOR.feed_forward(np.array([[1.0], [0.0]]), np.array([[1.0]]))[1]

print("Error:")
print(XOR.errors)
