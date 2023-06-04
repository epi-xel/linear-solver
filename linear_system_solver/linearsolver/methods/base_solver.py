import numpy as np
import linearsolver.methods.update as upd
from time import time
from linearsolver.utils.print_utils import bcolors
from linearsolver.helpers.ls_result import LinearSystemResult
from linearsolver.utils.constants import MAX_ITER
from linearsolver.helpers.ls_helper import LinearSystemHelper
from sksparse.cholmod import cholesky


# Check if the algorithm has converged
def converged(ls, tol):
    return np.linalg.norm(ls.A.dot(ls.x) - ls.b) / np.linalg.norm(ls.b) < tol


def init_solver(A, b, x, method, check):

    ls = None
    update = None

    if(method == "conjugate-gradient"):
        ls = LinearSystemHelper(A, b, x, conjugate_gradient = True)
    else:
        ls = LinearSystemHelper(A, b, x)

    if(check):
        if(checks(ls.A, ls.b) == -1):
            return
        
    if(method == "jacobi"):
        update = upd.jacobi
    elif(method == "gauss-seidel"):
        update = upd.gauss_seidel
    elif(method == "gradient"):
        update = upd.gradient_descent
    elif(method == "conjugate-gradient"):
        update = upd.conjugate_gradient
    else:
        print(bcolors.FAIL + "Error: Method " + method + " not found" + bcolors.ENDC)
        return
        
    return ls, update


def solve(A, b, x, tol, method, max_iter = MAX_ITER, check = True):

    ls, update = init_solver(A, b, x, method, check)

    start = time()
    k = 0

    while not converged(ls, tol):
    
        update(ls)
        k += 1

        if(k > max_iter):
            print(bcolors.FAIL
                + "Max iterations reached"
                + bcolors.ENDC )
            break

    end = time()

    time_elapsed = end - start

    res = LinearSystemResult(ls.x, time_elapsed, k, ls.x_true)

    return res


def checks(A, b):

    if(A.shape[0] != A.shape[1]):
        print(bcolors.FAIL + "Error: Matrix is not square" + bcolors.ENDC)
        return -1
    
    if(b.nonzero()[0].size == 0):
        print(bcolors.WARNING + "b is all zeros, answer is an array of zeros" + bcolors.ENDC)
        return -1
    
    try:
        cholesky(A.tocsc())
    except:
        print(bcolors.FAIL + "Error: Matrix is not positive definite or symmetric" + bcolors.ENDC)
        return -1

    return 1