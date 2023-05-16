from scipy.io import mmread
from pathlib import Path
from linearsolver.helpers.df_helper import ResultsStats
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
    parser.add_argument('-e', '--export', metavar='path/to/folder', type=str, help='Export results to csv file and plot graphs in specified folder')

    return parser


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

    return helper.complete_solve(os.path.basename(path), A, b, x, tols)


# Run all matrices .mtx in the specified folder
def run_all(path, tols, folder):

    if(path[-1] != '/'):
        path += '/'
    data = sorted(glob.glob(path + '*.mtx'))

    stats = ResultsStats()

    for m in data:
        stats0 = read_and_solve(m, tols)
        if stats0 is not None:
            stats.merge_stats(stats0)

    #export_results(df)
    if(folder is not None):
        analize.export_results(stats, folder)
        

def input_matrix(matrix_path, tols, folder):
    stats = read_and_solve(matrix_path, tols)
    if(folder is not None):
        analize.export_results(stats, folder)


# Parse command line arguments
def init_solver():

    tols = const.TOLS

    parser = init_parser()

    folder = None

    if(parser.parse_args().tolerance):
        tols = parser.parse_args().tolerance

    if(parser.parse_args().export):
        folder = parser.parse_args().export

    if(parser.parse_args().all):
        run_all(parser.parse_args().all, tols, folder)
    elif(parser.parse_args().matrix):
        input_matrix(parser.parse_args().matrix, tols, folder)
    else:
        parser.print_help()


if __name__ == '__main__':
    init_solver()