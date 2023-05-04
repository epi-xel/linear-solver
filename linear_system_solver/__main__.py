from scipy.io import mmread
import numpy as np
import linearsolver.methods.helpers as helper
import linearsolver.utils.constants as const
import argparse
import glob, os


# Parse command line arguments
def init_parser():
    
    parser = argparse.ArgumentParser(
                    prog='Linear System Solver',
                    description='Solve a linear system Ax = b and compute relative error ' +
                                'where A is a sparse matrix and x is a vector of ones')

    parser.add_argument('-a', '--all', metavar='path/to/folder', type=str, help='Run all matrices .mtx in the specified folder')
    parser.add_argument('-m', '--matrix', metavar='path/to/file.mtx', type=str, help='Matrix to solve as path to .mtx file')
    parser.add_argument('-t','--tolerance', metavar='tolerance', nargs='+', 
                        help='Tolerances to use in the methods. If none, tolerances used: 1e-4, 1e-6, 1e-8, 1e-10', type=float)

    return parser


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

    helper.complete_solve(os.path.basename(path), A, b, x, tols)



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

    parser = init_parser()

    if(parser.parse_args().tolerance):
        tols = parser.parse_args().tolerance

    if(parser.parse_args().all):
        test(parser.parse_args().all, tols)
    elif(parser.parse_args().matrix):
        input_matrix(parser.parse_args().matrix, tols)
    else:
        parser.print_help()


if __name__ == '__main__':
    init_solver()