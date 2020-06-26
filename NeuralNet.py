from math import exp
from random import seed
from random import random

class NeuralNet(object):
    def __init__(self, layers):
        seed(1)
        self._learning_rate = 0.01
        self._layers = list()
        for inputs in range(1, len(layers)):
            self._layers.append(Layer(layers[inputs - 1], layers[inputs]))

    def __repr__(self):
        return '<NeuralNet object __repr__>'

    def __str__(self):
        return "\n".join(str(layer) for layer in self._layers)

    def train(self, data):
        for row in data:
            self._forward_propagate(row[:-1])
            self._back_propagate(row[-1])
            self._update_weights(row, self.learning_rate)

    def test(self, data):
        correct = 0;
        total = 0;
        for row in data:
            self._forward_propagate(row[:-1])
            rounded = self._round_output(self.outputs)
            expected = self._expected_output(row[-1])
            print("predicted: ", rounded, "expected: ", expected)
            total += 1
            if rounded == expected:
                correct += 1
        print(correct, "/", total, " correct, accuracy: ", correct / total)
        return correct / total

    def _forward_propagate(self, row):
        inputs = row
        for layer in self._layers:
            for perceptron in layer:
                perceptron.outpt = self._transfer(perceptron.activate(inputs))
            inputs = layer.outputs

    def _back_propagate(self, expected):
        for perceptron, exptd in zip(self._layers[-1], self._expected_output(expected)):
            perceptron.error = (exptd - perceptron.outpt) \
                    * self._transfer_derivative(perceptron.outpt)
        for current_layer in reversed(range(len(self._layers) - 1)):
            for perc_current in self._layers[current_layer]:
                error = 0.0
                for perc_next in self._layers[current_layer + 1]:
                    error += perc_next.weights[perc_current.identity] * perc_next.error
                perc_current.error = error \
                        * self._transfer_derivative(perc_current.outpt)

    def _update_weights(self, row, learning_rate):
        for curr_layer in range(len(self._layers)):
            inputs = row[:-1]
            if curr_layer != 0:
                inputs = self._layers[curr_layer - 1].outputs
            for perceptron in self._layers[curr_layer]:
                for inpt in range(len(inputs)):
                    perceptron.weights[inpt] += learning_rate \
                            * perceptron.error * inputs[inpt]
                perceptron.bias += learning_rate * perceptron.error

    def _transfer(self, activation):
        return 1.0 / (1.0 + exp(-activation))

    def _transfer_derivative(self, outpt):
        return outpt * (1 - outpt)

    def _expected_output(self, outputs):
        return [1 if outpt == outputs else 0 for outpt in range(len(self._layers[-1]))]

    def _correct_output(self):
        outputs = self.outputs
        self.outputs = [1 if outpt == max(outputs) else 0 for outpt in outputs]

    def _round_output(self, outputs):
        return [1 if outpt == max(outputs) else 0 for outpt in outputs]

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, new_rate):
        self._learning_rate = new_rate

    @property
    def outputs(self):
        return self._layers[-1].outputs

    @outputs.setter
    def outputs(self, new_outputs):
        self._layers[-1].outputs = new_outputs

    @property
    def activate(self):
        return (self._transfer, self._transfer_derivative)

    @activate.setter
    def activate(self, new_activate):
        self._transfer = new_activate[0]
        self._transfer_derivative = new_activate[1]

class Layer(object):
    def __init__(self, inputs, perceptrons):
        self._perceptrons = [Perceptron(perceptron, inputs) for perceptron in range(perceptrons)]

    def __iter__(self):
        return self._perceptrons.__iter__()

    def __len__(self):
        return len(self._perceptrons)

    def __repr__(self):
        return '<Layer object __repr__>'

    def __str__(self):
        return self._identity + "\n   " + "\n   ".join(str(perceptron) \
                for perceptron in self._perceptrons)

    @property
    def perceptrons(self):
        return self.perceptrons

    @property
    def identity(self):
        return self._identity

    @property
    def errors(self):
        return [perceptron.error for perceptron in self._perceptrons]

    @property
    def outputs(self):
        return [perceptron.outpt for perceptron in self._perceptrons]

    @outputs.setter
    def outputs(self, new_outputs):
        for perceptron, new_output in zip(self._perceptrons, new_outputs):
            perceptron.outpt = new_output



class Perceptron(object):
    def __init__(self, identity, inputs):
        self._identity = identity
        self._weights = [random() for inpt in range(inputs + 1)]
        self._error = 0.0
        self._output = 0.0

    def __repr__(self):
        return '<Perceptron object __repr__>'

    def __str__(self):
        return "identity: " + str(self._identity) + "\n   " \
                + "weights: " + str(self._weights) + "\n   " \
                + "output: " + str(self._output) + "\n   " \
                + "error: " + str(self._error)

    def activate(self, inputs):
        activation = self.bias
        for inpt in range(len(inputs)):
            activation += self.weights[inpt] * inputs[inpt]
        return activation

    @property
    def identity(self):
        return self._identity

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, new_weights):
        self._weights = new_weights

    @property
    def bias(self):
        return self._weights[-1]


    @bias.setter
    def bias(self, new_bias):
        self._weights[-1] = new_bias

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, new_error):
        self._error = new_error

    @property
    def outpt(self):
        return self._output

    @outpt.setter
    def outpt(self, new_output):
        self._output = new_output
