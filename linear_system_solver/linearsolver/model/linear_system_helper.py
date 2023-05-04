import numpy as np


# Helper class for linear system solver
class LinearSystemHelper:
    def __init__(self, A, b, conjugate_gradient = False):
        self.A = A
        self.b = b
        self.x = init_x(A)

        if(conjugate_gradient):
            self.p = b - A.dot(init_x(A))
            self.r = b - A.dot(init_x(A))


# Initialize the solution vector
def init_x(A):
    return np.zeros(A.shape[1])