import numpy as np
import time as t
import linearsolver.model.ls_result as lsr
import linearsolver.utils.print_utils as pu
from sksparse.cholmod import cholesky
import sksparse.cholmod as cholmod


# Check if the algorithm has converged
def converged(ls, tol):
    return np.linalg.norm(ls.A.dot(ls.x) - ls.b) / np.linalg.norm(ls.b) < tol


def solve(ls, tol, update, max_iter):

    if(checks(ls.A, ls.b) == -1):
        return

    start = t.time()
    k = 0

    while not converged(ls, tol):
    
        update(ls)
        k += 1

        if(k > max_iter):
            print(pu.bcolors.FAIL
                + "Max iterations reached"
                + pu.bcolors.ENDC )
            break

    end = t.time()

    time_elapsed = end - start

    res = lsr.LSResult(ls.x, time_elapsed, k)

    return res


def checks(A, b):
    if(A.shape[0] != A.shape[1]):
        print(pu.bcolors.FAIL + "Matrix is not square" + pu.bcolors.ENDC)
        return -1
    
    if(b.nonzero()[0].size == 0):
        print(pu.bcolors.FAIL + "b is all zeros" + pu.bcolors.ENDC)
        return -1
    
    try:
        cholesky(A.tocsc())
    except:
        print(pu.bcolors.FAIL + "Matrix is not positive definite or symmetric" + pu.bcolors.ENDC)
        return -1

    return 1