import linearsolver.helpers.linear_system_helper as lsh
import linearsolver.utils.print_utils as pu
import linearsolver.methods.update as update
import linearsolver.methods.base_solver as bs
import linearsolver.utils.constants as const 


# Solve the system with each method and print the results
def solve_with_each_method(A, b, x, tol, max_iter):

    ls1 = lsh.LinearSystemHelper(A, b)
    jacobi_res = bs.solve(ls1, tol, update.jacobi, max_iter)
    pu.print_stats(jacobi_res, x, "Jacobi")

    ls2 = lsh.LinearSystemHelper(A, b)
    gauss_seidel_res = bs.solve(ls2, tol, update.gauss_seidel, max_iter)
    pu.print_stats(gauss_seidel_res, x, "Gauss-Seidel")

    ls3 = lsh.LinearSystemHelper(A, b)
    gradient_descent_res = bs.solve(ls3, tol, update.gradient_descent, max_iter)
    pu.print_stats(gradient_descent_res, x, "Gradient Descent")

    ls4 = lsh.LinearSystemHelper(A, b, conjugate_gradient = True)
    conjugate_gradient_res = bs.solve(ls4, tol, update.conjugate_gradient, max_iter)
    pu.print_stats(conjugate_gradient_res, x, "Conjugate Gradient", True)


# Solve the system with each method and each tolerance and print the results
def complete_solve(matrix_name, A, b, x, tols):

    # Print matrix info
    print("\n" + "=" * const.PRINTED_LINES_LENGTH)
    print(pu.bcolors.OKCYAN + "MATRIX " + matrix_name + pu.bcolors.ENDC)
    print("=" * const.PRINTED_LINES_LENGTH)

    if(bs.checks(A, b) == -1):
        return

    for tol in tols:
        print("\nTolerance: " + str(tol))
        print("*" * const.PRINTED_LINES_LENGTH)
        solve_with_each_method(A, b, x, tol, const.MAX_ITER)