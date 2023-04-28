import scipy as sp
import numpy as np
import time as t
import copy
#import scipy.sparse.linalg as ssl

class LinearSystemHelper:
    def __init__(self, A, b, conjugate_gradient = False):
        self.A = A
        self.b = b
        self.x = init_x(A)

        if(conjugate_gradient):
            self.p = b - A.dot(init_x(A))
            self.r = b - A.dot(init_x(A))

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

# TODO 2: Check if the matrix is diagonally dominant etc
# TODO 3: Check if tutti 0 b, togli colori

# Initialize the solution vector
def init_x(A):
    return np.zeros(A.shape[1])


# Check if the algorithm has converged
def converged(ls, tol):
    return np.linalg.norm(ls.A.dot(ls.x) - ls.b) / np.linalg.norm(ls.b) < tol


# Compute the Jacobi update
def jacobi(ls):
    P = np.diag(ls.A.diagonal())
    inv_P = np.diag(1 / ls.A.diagonal())
    ls.x = ls.x - inv_P.dot(ls.A.dot(ls.x) - ls.b)
    return


# Solve the linear system Ax = b using forward substitution
def forward_substitution(A, b):
    n = A.shape[0]
    x = np.zeros(n)
    A = A.tocsc()
    for i in range(n):
        x[i] = (b[i] - A[i, :i].dot(x[:i])) / float(A[i, i])
 
    return x


# Compute the Gauss-Seidel update
def gauss_seidel(ls):
    P = sp.sparse.tril(ls.A)
    r = ls.b - ls.A.dot(ls.x)
    #y = sp.sparse.linalg.spsolve_triangular(P.tocsr(), r)
    y = forward_substitution(P, r)
    ls.x = ls.x + y
    return


# Compute the gradient descend update
def gradient_descent(ls):
    r = ls.b - ls.A.dot(ls.x)
    alpha = np.transpose(r).dot(r) / np.transpose(r).dot(ls.A.dot(r))
    ls.x = ls.x + alpha * r
    return

# Compute the conjugate gradient update
def conjugate_gradient(ls):
    alpha = np.transpose(ls.p).dot(ls.r) / np.transpose(ls.p).dot(ls.A.dot(ls.r))
    ls.x = ls.x + alpha * ls.p
    ls.r = ls.b - ls.A.dot(ls.x)
    beta = np.transpose(ls.p).dot(ls.A.dot(ls.r)) / np.transpose(ls.p).dot(ls.A.dot(ls.p))
    ls.p = ls.r - beta * ls.p
    return


# Compute relative error
def rel_error(x, x_true):
    return np.linalg.norm(x - x_true) / np.linalg.norm(x_true)


def solve(ls, tol, update, max_iter):

    start = t.time()
    k = 0

    while not converged(ls, tol):
    
        update(ls)
        k += 1

        if(k > max_iter):
            print(bcolors.FAIL
                + "Max iterations reached"
                + bcolors.ENDC )
            break

    end = t.time()

    time_elapsed = end - start

    res = dict();
    res['solution'] = ls.x
    res['time'] = time_elapsed
    res['iterations'] = k

    return res


# Print the stats of the solution
def print_stats(res, x_true, method, last = False):

    print(bcolors.BOLD
          + "Stats for "
          + bcolors.OKBLUE
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
def solve_with_each_method(A, b, x, tol, max_iter):

    ls1 = LinearSystemHelper(A, b)
    jacobi_res = solve(ls1, tol, jacobi, max_iter)
    print_stats(jacobi_res, x, "Jacobi")

    ls2 = LinearSystemHelper(A, b)
    gauss_seidel_res = solve(ls2, tol, gauss_seidel, max_iter)
    print_stats(gauss_seidel_res, x, "Gauss-Seidel")

    ls3 = LinearSystemHelper(A, b)
    gradient_descent_res = solve(ls3, tol, gradient_descent, max_iter)
    print_stats(gradient_descent_res, x, "Gradient Descent")

    ls4 = LinearSystemHelper(A, b, conjugate_gradient = True)
    conjugate_gradient_res = solve(ls4, tol, conjugate_gradient, max_iter)
    print_stats(conjugate_gradient_res, x, "Conjugate Gradient", True)

