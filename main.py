import scipy as sp
import numpy as np
import time as t

# Load the data
spa1 = sp.mmread('data/spa1.mtx')
spa2 = sp.mmread('data/spa2.mtx')
vem1 = sp.mmread('data/vem1.mtx')
vem2 = sp.mmread('data/vem2.mtx')

max_iter = 1000

# TODO 1: Input management
# TODO 2: Check if the matrix is diagonally dominant etc
# TODO 3: Calculate time for each method 

def solve(A, b, tol, update):

    start = t.time()
    x = init_x(A)
    k = 0
    prev_x = x
    while not converged(x, prev_x, k, A, b, tol):
        prev_x = x
        x = update(A, b, x)
        k += 1
        if(k > max_iter):
            print("Max iterations reached")
            break
    end = t.time()

    time_elapsed = end - start
    res = dict();
    res['solution'] = x
    res['time'] = time_elapsed
    res['iterations'] = k

    return res


# Initialize the solution vector
def init_x(A):
    return np.zeros(A.shape[1])


# Check if the algorithm has converged
def converged(x, prev_x, k, A, b, tol):
    if(k == 0):
        cond1 = False
    else:
        cond1 = np.linalg.norm(x - prev_x) / np.linalg.norm(init_x(A)) < tol 
    cond2 = np.linalg.norm(A.dot(x) - b) / np.linalg.norm(b) < tol
    return cond1 or cond2


# Compute the Jacobi update
def jacobi(A, b, x):
    P = np.diag(A.diagonal())
    inv_P = np.diag(1 / A.diagonal())
    return x - inv_P.dot(A.dot(x) - b)


# Solve the linear system Ax = b using forward substitution
def forward_substitution(A, b):
    n = A.shape[0]
    x = np.zeros(n)
    for i in range(n):
        x[i] = (b[i] - A[i, :i].dot(x[:i])) / A[i, i]
    return x


# Compute the Gauss-Seidel update
def gauss_seidel(A, b, x):
    P = np.tril(A)
    y = forward_substitution(P, b)
    return x - y


# Test methods with x = [1, 1, ... 1]
def test(A, tol):
    x = np.ones(A.shape[1])
    b = A.dot(x)
    print("Jacobi")
    jacobi_res = solve(A, b, tol, jacobi)
    print("Gauss-Seidel")
    gauss_seidel_res = solve(A, b, tol, gauss_seidel)


# Compute relative error
def rel_error(x, x_true):
    return np.linalg.norm(x - x_true) / np.linalg.norm(x_true)


# Print the stats of the solution
def print_stats(res):
    print("Solution: ", res['solution'])
    print("Time: ", res['time'])
    print("Iterations: ", res['iterations'])

