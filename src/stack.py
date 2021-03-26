"""
stack.py
~~~~~~~~

A module to implement a binary stack whose content is encoded as a particular rational number
that allows the usual stack operations to be easily redefined to operate on such number.
"""


class Stack:

    def __init__(self, stack):
        """``stack`` must be a list of binary values representing, from left to right, the stack content from top to
        bottom."""
        self.encoding = 0
        for i in range(len(stack)):
            self.encoding += (2 * stack[i] + 1) / 4 ** (i + 1)

    # stack operations:
    def top(self):
        return self.encoding > 0.5

    def nonempty(self):
        return self.encoding != 0

    def push(self, symbol):
        if symbol != 0 and symbol != 1:
            raise ValueError('Stack symbols must be binary')
        self.encoding = self.encoding / 4 + (2 * symbol + 1) / 4

    def pop(self):
        if not self.nonempty():
            raise RuntimeError('Cannot pop an empty stack')
        self.encoding = 4 * self.encoding - 2 * self.top() - 1
