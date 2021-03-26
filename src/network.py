"""
network.py
~~~~~~~~~~

A module to implement a recurrent neural network with four layers that simulates the PDA for unary addition defined in
``machine.py``.
"""

import numpy
from stack import Stack


class Network:

    def __init__(self):
        # weight matrices:
        self.weights = [
            numpy.array([[1, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0],
                         [0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 1],
                         [0, 0, 0, 4, 0],
                         [0, 0, 0, 0, 4],
                         [0, 0, 0, 4, 0],
                         [0, 0, 0, 0, 4]]),
            numpy.array([[1, 0, 0, 0, 1, 0, 1, 0],
                         [1, 0, 0, 0, -1, 0, 1, 0],
                         [1, 0, 0, 0, 0, 0, -1, 0],
                         [0, 1, 0, 0, 0, 0, 0, 1],
                         [0, 1, 0, 0, 0, 0, 0, -1],
                         [1, 0, 4, 0, -1, 0, 1, 0],
                         [1, 0, 4, 0, -3, 0, 1, 0],
                         [1, 0, 1, 0, 0, 0, -1, 0],
                         [0, 1, 1/4, 0, 0, 0, 0, 1],
                         [0, 1, 1, 0, 0, 0, 0, -1],
                         [1, 0, 0, 1/4, 1, 0, 1, 0],
                         [1, 0, 0, 1, -1, 0, 1, 0],
                         [1, 0, 0, 1, 0, 0, -1, 0],
                         [0, 1, 0, 4, 0, -2, 0, 1],
                         [0, 1, 0, 1, 0, 0, 0, -1]]),
            numpy.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
        ]
        # bias vectors:
        self.biases = [
            numpy.array([0, 0, 0, 0, -2, -2, 0, 0]),
            numpy.array([-2, -1, 0, -1, 0, -4, -3, -1, -5/4, -1, -9/4, -2, -1, -3, -1]),
            numpy.zeros(5)
        ]

    def execute(self, stack1):
        """Simulates a PDA for unary addition, where ``stack1`` is the content of its first stack at the beginning of
        execution (i.e. a list containing the two unary numbers to add separated by a zero)."""

        def feedforward(a):
            """Computes the network output if vector ``a`` is input, then calls itself with that output as its own
            input until the third neuron of the output layer, which represents the halting state of the PDA,
            contains the value ``1`` (i.e. the PDA is in the halting state)."""
            if a[2] == 1:
                return a[3], a[4]
                # the fourth and fifth neuron of the output layer contain the encodings of the PDA's first and second
                # stack, respectively.
            for w, b in zip(self.weights, self.biases):
                a = sigmoid(numpy.dot(w, a) + b)
            return feedforward(a)

        a = numpy.array([1, 0, 0, Stack(stack1).encoding, Stack([]).encoding])
        # The first neuron activation set to 1 means that the PDA is in its initial state.
        # stack1_encoding, stack2_encoding = feedforward(a)
        # print('Stack 1 encoding = {}'.format(stack1_encoding))
        # print('Stack 2 encoding = {}'.format(stack2_encoding))
        return feedforward(a)


def sigmoid(x):
    """Applies the sigmoid function

                     0 if x < 0
        sigmoid(x) = x if 0 <= x <= 1
                     1 if x > 1

    elementwise."""
    return numpy.piecewise(x, [x < 0, numpy.logical_and(0 <= x, x <= 1), x > 1], [0, lambda x: x, 1])
