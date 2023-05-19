import numpy as np

class LinearSystemResult:
    def __init__(self, solution, time, iterations, x_true):
        self.solution = solution
        self.time = time
        self.iterations = iterations
        self.relative_error = rel_error(solution, x_true)


# Compute relative error
def rel_error(x, x_true):
    return np.linalg.norm(x - x_true) / np.linalg.norm(x_true)