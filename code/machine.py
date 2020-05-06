"""
machine.py
~~~~~~~~~~

A module to implement a pushdown automaton (PDA) with two binary stacks (equivalent, in terms of computational power, to
a binary single-tape Turing machine) that computes unary addition.
"""

from stack import Stack


class DoubleStackPDA:

    def __init__(self, stack1):
        """``stack1`` must be a list containing the two unary numbers to add separated by a zero."""
        self.stack1 = Stack(stack1)
        self.stack2 = Stack([])  # stack2 starts empty
        self.states = ['s0', 's1', 'halt']
        self.state = self.states[0]

    def update(self):
        """Implements the PDA transitions."""
        if self.state is self.states[0] and self.stack1.nonempty() and self.stack1.top():
            self.stack1.pop()
            self.stack2.push(1)
            print('Popped stack1, pushed 1 in stack 2, and stayed in s0')
        elif self.state is self.states[0] and self.stack1.nonempty() and not self.stack1.top():
            self.stack1.pop()
            print('Popped stack1 and stayed in s0')
        elif self.state is self.states[0] and not self.stack1.nonempty():
            self.state = self.states[1]
            print('Went to s1')
        elif self.state is self.states[1] and self.stack2.nonempty():
            self.stack1.push(1)
            self.stack2.pop()
            print('Pushed 1 in stack 1, popped stack2, and stayed in s1')
        elif self.state is self.states[1] and not self.stack2.nonempty():
            self.state = self.states[2]
            print('Went to halt\n')
        else:
            raise RuntimeError('Undefined transition')

    def execute(self):
        """Updates the state and stack contents of the PDA until it reaches the halting state. When the halting state
        has been reached, the first stack will contain the unary sum of the two initial numbers (whereas the second
        stack will be empty)."""
        while self.state is not self.states[2]:
            self.update()
        print('Stack 1 encoding = {}'.format(self.stack1.encoding))
        print('Stack 2 encoding = {}'.format(self.stack2.encoding))
