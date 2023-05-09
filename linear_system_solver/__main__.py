from scipy.io import mmread
from pathlib import Path
import numpy as np
import linearsolver.methods.big_ops as helper
import linearsolver.utils.constants as const
import linearsolver.utils.analize as analize
import pandas as pd
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
    parser.add_argument('-e', '--export', action='store_true', help='Export results to csv file and plot graphs')

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

    return helper.complete_solve(os.path.basename(path), A, b, x, tols)


# Run all matrices .mtx in the specified folder
def run_all(path, tols, export_results):

    if(path[-1] != '/'):
        path += '/'
    data = sorted(glob.glob(path + '*.mtx'))

    df = analize.init_ls_df()

    for m in data:
        df1 = read_and_solve(m, tols)
        if df1 is not None:
            df = pd.concat([df, df1], ignore_index=True)

    #export_results(df)
    if(export_results):
        analize.export_results(df)
        

def input_matrix(matrix_path, tols, export_results):
    df = read_and_solve(matrix_path, tols)
    if(export_results):
        export_results(df)


def export_results(df):
    analize.compare_results(df)
    analize.export_results(df)


# Parse command line arguments
def init_solver():

    tols = const.TOLS

    parser = init_parser()

    export_results = False

    if(parser.parse_args().tolerance):
        tols = parser.parse_args().tolerance

    if(parser.parse_args().export):
        export_results = True

    if(parser.parse_args().all):
        run_all(parser.parse_args().all, tols, export_results)
    elif(parser.parse_args().matrix):
        input_matrix(parser.parse_args().matrix, tols, export_results)
    else:
        parser.print_help()


if __name__ == '__main__':
    init_solver()