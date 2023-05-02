from scipy.io import mmread
import numpy as np
import api.logic.helpers as helper
import api.utils.print_utils as pu
import api.utils.parser as pars
import api.model.constants as const
import glob, os


# Read a matrix from a .mtx file and solve the linear system Ax = b
def read_and_solve(path, tols):

    # Read matrix from file
    try:
        A = mmread(path)
    except Exception as e:
        print('Unable to read matrix: check path or file format')
        exit(0)

    # Test methods with x = [1, 1, ... 1]
    x = np.ones(A.shape[1])
    b = A.dot(x)

    # Print matrix info
    print("\n==========================================")
    print(pu.bcolors.OKCYAN + "MATRIX " + os.path.basename(path) + pu.bcolors.ENDC)
    print("==========================================")

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

    parser = pars.init_parser()

    if(parser.parse_args().tolerance):
        tols = parser.parse_args().tolerance

    if(parser.parse_args().all):
        test(parser.parse_args().all, const.TOLS)
    elif(parser.parse_args().matrix):
        input_matrix(parser.parse_args().matrix, const.TOLS)
    else:
        parser.print_help()




