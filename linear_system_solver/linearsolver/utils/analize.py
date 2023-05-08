import scipy.sparse as sp
import pandas as pd


def analize_matrix(A):
    shape = A.tocsr().get_shape()
    nonzero = A.tocsr().count_nonzero()
    size = shape[0] * shape[1] 
    density = nonzero / size
    return size, density


def build_result_df(A, res, name, tol, method):
    size, density = analize_matrix(A)
    data = {'Matrix': [name],
            'Size': [size],
            'Density': [density],
            'Tolerance': [tol],
            'Method': [method],
            'Relative error': [res.relative_error],
            'Time': [res.time],
            'Iterations': [res.iterations]
            }
    
    df = pd.DataFrame(data)
    return df


def compare_result(results):

    return