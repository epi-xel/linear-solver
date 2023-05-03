import numpy as np
import time as t
import api.model.ls_result as lsr
import api.utils.print_utils as pu


# TODO 2: Check if the matrix is simmetric definite positive
# TODO 3: Check if tutti 0 b

# Check if the algorithm has converged
def converged(ls, tol):
    return np.linalg.norm(ls.A.dot(ls.x) - ls.b) / np.linalg.norm(ls.b) < tol


def solve(ls, tol, update, max_iter):

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