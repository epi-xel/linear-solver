from scipy.io import mmread
import scipy.sparse.linalg as sl
import scipy as sp
import numpy as np
import scipy.sparse as sparse
import api.logic.helpers as helper
import api.utils.print_utils as pu
import api.utils.parser as pars
import api.model.constants as const
import glob, os


def is_matrix_definite_positive(A):
    vals, vecs = sl.eigs(A)
    return sp.all(vals)


def is_matrix_symmetric(A):

    A1 = A.tocsr()
    A2 = sparse.csr_matrix.transpose(A.tocsr())
    return (np.array_equal(A1.indptr, A2.indptr) and 
            np.array_equal(A1.indices, A2.indices) and 
            np.array_equal(A1.data, A2.data))


# Read a matrix from a .mtx file and solve the linear system Ax = b
def read_and_solve(path, tols):

    # Read matrix from file
    try:
        A = mmread(path)
    except Exception as e:
        print('Unable to read matrix: check path or file format')
        return

    # Test methods with x = [1, 1, ... 1]
    x = np.ones(A.shape[1])
    b = A.dot(x)

    # Print matrix info
    print("\n==========================================")
    print(pu.bcolors.OKCYAN + "MATRIX " + os.path.basename(path) + pu.bcolors.ENDC)
    print("==========================================")

    if (not is_matrix_definite_positive(A) or not is_matrix_symmetric(A)):
        print(pu.bcolors.FAIL + "Matrix is not definite positive or symmetric" + pu.bcolors.ENDC)
        return

    for tol in tols:
        print("\nTolerance: " + str(tol))
        print("******************************************")
        helper.solve_with_each_method(A, b, x, tol, const.MAX_ITER)


# Run all matrices .mtx in the specified folder
def test(path, tols):

    if(path[-1] != '/'):
        path += '/'
    data = sorted(glob.glob(path + '*.mtx'))
    
    for m in data:
        read_and_solve(m, tols)


def input_matrix(matrix_path, tols):
    read_and_solve(matrix_path, tols)


# Parse command line arguments
def init_solver():

    tols = const.TOLS

    parser = pars.init_parser()

    if(parser.parse_args().tolerance):
        tols = parser.parse_args().tolerance

    if(parser.parse_args().all):
        test(parser.parse_args().all, tols)
    elif(parser.parse_args().matrix):
        input_matrix(parser.parse_args().matrix, tols)
    else:
        parser.print_help()
