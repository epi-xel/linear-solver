from linearsolver.methods.base_solver import *
from linearsolver.utils.print_utils import print_stats
from linearsolver.utils.print_utils import bcolors
from linearsolver.helpers.df_helper import ResultsStats
import linearsolver.utils.constants as const 


# Solve the system with each method and print the results. A, b and x are coo_matrix
def solve_with_each_method(A, b, x, tol, max_iter, matrix_name):

    stats = ResultsStats()

    jacobi_res = solve(A, b, x, tol, Method.JACOBI, max_iter = max_iter, check = False)
    print_stats(jacobi_res, x, "Jacobi")
    stats.add_stats(A, jacobi_res, matrix_name, tol, "Jacobi")

    gauss_seidel_res = solve(A, b, x, tol, Method.GAUSS_SEIDEL, max_iter = max_iter, check = False)
    print_stats(gauss_seidel_res, x, "Gauss-Seidel")
    stats.add_stats(A, gauss_seidel_res, matrix_name, tol, "Gauss-Seidel")

    gradient_descent_res = solve(A, b, x, tol, Method.GRADIENT, max_iter = max_iter, check = False)
    print_stats(gradient_descent_res, x, "Gradient Descent")
    stats.add_stats(A, gradient_descent_res, matrix_name, tol, "Gradient Descent")

    conjugate_gradient_res = solve(A, b, x, tol, Method.CONJUGATE_GRADIENT, max_iter = max_iter, check = False)
    print_stats(conjugate_gradient_res, x, "Conjugate Gradient", True)
    stats.add_stats(A, conjugate_gradient_res, matrix_name, tol, "Conjugate Gradient")

    return stats


# Solve the system with each method and each tolerance, print the results and return complete df
def complete_solve(matrix_name, A, b, x, tols):

    # Print matrix info
    print("\n" + "=" * const.PRINTED_LINES_LENGTH)
    print(bcolors.OKCYAN + "MATRIX " + matrix_name + bcolors.ENDC)
    print("=" * const.PRINTED_LINES_LENGTH)

    if(checks(A, b) == -1):
        return
    
    stats = ResultsStats()

    for tol in tols:
        print("\nTolerance: " + str(tol))
        print("*" * const.PRINTED_LINES_LENGTH)
        stats.merge_stats(solve_with_each_method(A, b, x, tol, const.MAX_ITER, matrix_name))

    return stats