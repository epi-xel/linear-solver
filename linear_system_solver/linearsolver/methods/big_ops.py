from linearsolver.helpers.ls_helper import LinearSystemHelper
from linearsolver.methods.base_solver import solve
from linearsolver.methods.base_solver import checks
from linearsolver.utils.print_utils import print_stats
from linearsolver.utils.print_utils import bcolors
import linearsolver.methods.update as update
import linearsolver.utils.constants as const 


# Solve the system with each method and print the results. A, b and x are coo_matrix
def solve_with_each_method(A, b, x, tol, max_iter):

    ls1 = LinearSystemHelper(A, b)
    jacobi_res = solve(ls1, tol, update.jacobi, max_iter, check = False)
    print_stats(jacobi_res, x, "Jacobi")

    ls2 = LinearSystemHelper(A, b)
    gauss_seidel_res = solve(ls2, tol, update.gauss_seidel, max_iter, check = False)
    print_stats(gauss_seidel_res, x, "Gauss-Seidel")

    ls3 = LinearSystemHelper(A, b)
    gradient_descent_res = solve(ls3, tol, update.gradient_descent, max_iter, check = False)
    print_stats(gradient_descent_res, x, "Gradient Descent")

    ls4 = LinearSystemHelper(A, b, conjugate_gradient = True)
    conjugate_gradient_res = solve(ls4, tol, update.conjugate_gradient, max_iter, check = False)
    print_stats(conjugate_gradient_res, x, "Conjugate Gradient", True)


# Solve the system with each method and each tolerance and print the results
def complete_solve(matrix_name, A, b, x, tols):

    # Print matrix info
    print("\n" + "=" * const.PRINTED_LINES_LENGTH)
    print(bcolors.OKCYAN + "MATRIX " + matrix_name + bcolors.ENDC)
    print("=" * const.PRINTED_LINES_LENGTH)

    if(checks(A, b) == -1):
        return

    for tol in tols:
        print("\nTolerance: " + str(tol))
        print("*" * const.PRINTED_LINES_LENGTH)
        solve_with_each_method(A, b, x, tol, const.MAX_ITER)