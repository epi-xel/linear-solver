from scipy.io import mmread
import numpy as np
import utils

max_iter = 20000

def test():
    
    # Load the data
    data = ['spa1', 'spa2', 'vem1', 'vem2']

    # Test methods with x = [1, 1, ... 1]
    for m in data:
        A = mmread('data/' + m + '.mtx')
        x = np.ones(A.shape[1])
        b = A.dot(x)
        print("\n==========================================")
        print(utils.bcolors.HEADER + "Matrix " + m + utils.bcolors.ENDC)
        print("==========================================")
        utils.solve_with_each_method(A, b, x, 1e-6, max_iter)

test()