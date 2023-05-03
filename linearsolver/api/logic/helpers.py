import api.model.linear_system_helper as lsh
import api.utils.print_utils as pu
import api.logic.methods as methods
import api.logic.base_solver as bs


# Solve the system with each method
def solve_with_each_method(A, b, x, tol, max_iter):

    ls1 = lsh.LinearSystemHelper(A, b)
    jacobi_res = bs.solve(ls1, tol, methods.jacobi, max_iter)
    pu.print_stats(jacobi_res, x, "Jacobi")

    ls2 = lsh.LinearSystemHelper(A, b)
    gauss_seidel_res = bs.solve(ls2, tol, methods.gauss_seidel, max_iter)
    pu.print_stats(gauss_seidel_res, x, "Gauss-Seidel")

    ls3 = lsh.LinearSystemHelper(A, b)
    gradient_descent_res = bs.solve(ls3, tol, methods.gradient_descent, max_iter)
    pu.print_stats(gradient_descent_res, x, "Gradient Descent")

    ls4 = lsh.LinearSystemHelper(A, b, conjugate_gradient = True)
    conjugate_gradient_res = bs.solve(ls4, tol, methods.conjugate_gradient, max_iter)
    pu.print_stats(conjugate_gradient_res, x, "Conjugate Gradient", True)

