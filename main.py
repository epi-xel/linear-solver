import scipy as sp
from scipy.io import mmread
import scipy.sparse.linalg
import numpy as np
import time as t

max_iter = 20000

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# TODO 1: Input management
# TODO 2: Check if the matrix is diagonally dominant etc

# Initialize the solution vector
def init_x(A):
    return np.zeros(A.shape[1])


# Check if the algorithm has converged
def converged(x, prev_x, k, A, b, tol):
    return np.linalg.norm(A.dot(x) - b) / np.linalg.norm(b) < tol


# Compute the Jacobi update
def jacobi(A, b, x):
    P = np.diag(A.diagonal())
    inv_P = np.diag(1 / A.diagonal())
    return x - inv_P.dot(A.dot(x) - b)


# Solve the linear system Ax = b using forward substitution
def forward_substitution(A, b):
    n = A.shape[0]
    x = np.zeros(n)
    A = A.tocsc()

    for i in range(n):
        x[i] = (b[i] - A[i, :i].dot(x[:i])) / float(A[i, i])
 
    return x


# Compute the Gauss-Seidel update
def gauss_seidel(A, b, x):
    P = sp.sparse.tril(A)
    r = b - A.dot(x)
    #y = sp.sparse.linalg.spsolve_triangular(P.tocsr(), r)
    y = forward_substitution(P, r)
    return x + y


# Compute relative error
def rel_error(x, x_true):
    return np.linalg.norm(x - x_true) / np.linalg.norm(x_true)


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
            print(bcolors.FAIL
                + "Max iterations reached"
                + bcolors.ENDC )
            break

    end = t.time()

    time_elapsed = end - start
    res = dict();
    res['solution'] = x
    res['time'] = time_elapsed
    res['iterations'] = k

    return res


# Print the stats of the solution
def print_stats(res, x_true, method, last = False):

    print(bcolors.BOLD
          + "Stats for "
          + bcolors.OKCYAN
          + method 
          + bcolors.ENDC
          + bcolors.BOLD
          + " method" 
          + bcolors.ENDC)
    print(bcolors.OKGREEN 
          + "> Relative error:  " 
          + bcolors.ENDC 
          + str(rel_error(res['solution'], x_true)))
    print(bcolors.OKGREEN 
          + "> Elapsed time:    " 
          + bcolors.ENDC 
          + str(res['time']) + " sec")
    print(bcolors.OKGREEN
          + "> Iterations:      " 
          + bcolors.ENDC 
          + str(res['iterations']))
    if not last:
        print("------------------------------------------")
    

# Solve the system with each method
def solve_with_each_method(A, b, x, tol):

    jacobi_res = solve(A, b, tol, jacobi)
    print_stats(jacobi_res, x, "Jacobi")
    gauss_seidel_res = solve(A, b, tol, gauss_seidel)
    print_stats(gauss_seidel_res, x, "Gauss-Seidel", True)


def test():
    
    # Load the data
    data = ['spa1', 'spa2', 'vem1', 'vem2']

    # Test methods with x = [1, 1, ... 1]
    for m in data:
        A = mmread('data/' + m + '.mtx')
        x = np.ones(A.shape[1])
        b = A.dot(x)
        print(bcolors.HEADER + "Matrix " + m + bcolors.ENDC)
        solve_with_each_method(A, b, x, 1e-6)
        print("==========================================")

test()

