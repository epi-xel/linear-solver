from scipy.io import mmread
import numpy as np
import utils

MAX_ITER = 20000
TOL = 1e-6

def test():
    
    # Load the data
    data = ['spa1', 'spa2', 'vem1', 'vem2']

    # Test methods with x = [1, 1, ... 1]
    for m in data:

        A = mmread('data/' + m + '.mtx')
        x = np.ones(A.shape[1])
        b = A.dot(x)

        print("\n==========================================")
        print(utils.bcolors.OKCYAN + "MATRIX " + m + utils.bcolors.ENDC)
        print("Tolerance: " + str(TOL))
        print("==========================================")

        utils.solve_with_each_method(A, b, x, TOL, MAX_ITER)

test()