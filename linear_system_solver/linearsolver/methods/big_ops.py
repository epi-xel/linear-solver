from linearsolver.helpers.ls_helper import LinearSystemHelper
from linearsolver.methods.base_solver import solve
from linearsolver.methods.base_solver import checks
from linearsolver.utils.print_utils import print_stats
from linearsolver.utils.print_utils import bcolors
from linearsolver.utils.analize import build_result_df
import linearsolver.methods.update as update
import linearsolver.utils.constants as const 
import pandas as pd


# Solve the system with each method and print the results. A, b and x are coo_matrix
def solve_with_each_method(A, b, x, tol, max_iter, matrix_name):

    ls1 = LinearSystemHelper(A, b, x)
    jacobi_res = solve(ls1, tol, update.jacobi, max_iter, check = False)
    print_stats(jacobi_res, x, "Jacobi")
    df = build_result_df(A, jacobi_res, matrix_name, tol, "Jacobi")

    ls2 = LinearSystemHelper(A, b, x)
    gauss_seidel_res = solve(ls2, tol, update.gauss_seidel, max_iter, check = False)
    print_stats(gauss_seidel_res, x, "Gauss-Seidel")
    df = pd.concat([df, build_result_df(A, gauss_seidel_res, matrix_name, tol, "Gauss-Seidel")], ignore_index=True)

    ls3 = LinearSystemHelper(A, b, x)
    gradient_descent_res = solve(ls3, tol, update.gradient_descent, max_iter, check = False)
    print_stats(gradient_descent_res, x, "Gradient Descent")
    df = pd.concat([df, build_result_df(A, gradient_descent_res, matrix_name, tol, "Gradient")], ignore_index=True)

    ls4 = LinearSystemHelper(A, b, x, conjugate_gradient = True)
    conjugate_gradient_res = solve(ls4, tol, update.conjugate_gradient, max_iter, check = False)
    print_stats(conjugate_gradient_res, x, "Conjugate Gradient", True)
    df = pd.concat([df, build_result_df(A, conjugate_gradient_res, matrix_name, tol, "Conjugate Gradient")], ignore_index=True)

    return df


# Solve the system with each method and each tolerance, print the results and return complete df
def complete_solve(matrix_name, A, b, x, tols):

    # Print matrix info
    print("\n" + "=" * const.PRINTED_LINES_LENGTH)
    print(bcolors.OKCYAN + "MATRIX " + matrix_name + bcolors.ENDC)
    print("=" * const.PRINTED_LINES_LENGTH)

    if(checks(A, b) == -1):
        return
    
    df = pd.DataFrame(columns=["Matrix", "Size", "Density", "Tolerance", "Method", "Relative error", "Time", "Iterations"])

    for tol in tols:
        print("\nTolerance: " + str(tol))
        print("*" * const.PRINTED_LINES_LENGTH)
        df = pd.concat([df, solve_with_each_method(A, b, x, tol, const.MAX_ITER, matrix_name)])

    return df