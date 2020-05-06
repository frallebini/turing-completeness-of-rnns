"""
test.py
~~~~~~~

Just a few lines of code to test whether the neural network defined in ``network.py`` actually simulates the PDA in
``machine.py``; that is, given the same input to the PDA and the network, one gets the same output.
"""

from machine import DoubleStackPDA
from network import Network

stack = [1, 0, 1]

print('\n======== Double-stack PDA ========\n')
m = DoubleStackPDA(stack)
m.execute()

print('\n======== Neural network ========\n')
n = Network()
n.execute(stack)
