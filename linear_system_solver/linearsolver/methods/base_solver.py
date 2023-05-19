import numpy as np
from time import time
from linearsolver.utils.print_utils import bcolors
from linearsolver.helpers.ls_result import LinearSystemResult
from sksparse.cholmod import cholesky


# Check if the algorithm has converged
def converged(ls, tol):
    return np.linalg.norm(ls.A.dot(ls.x) - ls.b) / np.linalg.norm(ls.b) < tol


def solve(ls, tol, update, max_iter, check = True):

    if(check):
        if(checks(ls.A, ls.b) == -1):
            return

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