from scipy.io import mmread
import numpy as np
import utils
import argparse
import glob, os

MAX_ITER = 20000
TOL = 1e-6

def read_and_solve(path):

    try:
        A = mmread(path)
    except Exception as e:
        print('Unable to read matrix: check path or file format')
        exit(0)

    # Test methods with x = [1, 1, ... 1]
    x = np.ones(A.shape[1])
    b = A.dot(x)

    print("\n==========================================")
    print(utils.bcolors.OKCYAN + "MATRIX " + os.path.basename(path) + utils.bcolors.ENDC)
    print("Tolerance: " + str(TOL))
    print("==========================================")

    utils.solve_with_each_method(A, b, x, TOL, MAX_ITER)


def test():

    data = sorted(glob.glob('test/*.mtx'))
    for m in data:
        read_and_solve(m)


def input_matrix(matrix_path):
    read_and_solve(matrix_path)


def init_parser():
    parser = argparse.ArgumentParser(
                    prog='Linear System Solver',
                    description='Solve a linear system Ax = b and compute relative error ' +
                                'where A is a sparse matrix and x is a vector of ones')

    parser.add_argument('-t', '--test', action='store_true', help='Run test matrices .mtx from test/ folder')
    parser.add_argument('-m', '--matrix', type=str, help='Matrix to solve as path to .mtx file')

    if(parser.parse_args().test):
        test()
    elif(parser.parse_args().matrix):
        input_matrix(parser.parse_args().matrix)

init_parser()



