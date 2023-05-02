import scipy as sp
import numpy as np
import time as t
import api.model.linear_system_helper as lsh
import api.utils.print_utils as pu
import api.model.ls_result as lsr


# TODO 2: Check if the matrix is diagonally dominant etc
# TODO 3: Check if tutti 0 b


# Check if the algorithm has converged
def converged(ls, tol):
    return np.linalg.norm(ls.A.dot(ls.x) - ls.b) / np.linalg.norm(ls.b) < tol


# Compute the Jacobi update
def jacobi(ls):
    P = np.diag(ls.A.diagonal())
    inv_P = np.diag(1 / ls.A.diagonal())
    ls.x = ls.x - inv_P.dot(ls.A.dot(ls.x) - ls.b)
    return


# Solve the linear system Ax = b using forward substitution
def forward_substitution(A, b):
    n = A.shape[0]
    x = np.zeros(n)
    A = A.tocsc()
    for i in range(n):
        x[i] = (b[i] - A[i, :i].dot(x[:i])) / float(A[i, i])
 
    return x


# Compute the Gauss-Seidel update
def gauss_seidel(ls):
    P = sp.sparse.tril(ls.A)
    r = ls.b - ls.A.dot(ls.x)
    #y = sp.sparse.linalg.spsolve_triangular(P.tocsr(), r)
    y = forward_substitution(P, r)
    ls.x = ls.x + y
    return


# Compute the gradient descend update
def gradient_descent(ls):
    r = ls.b - ls.A.dot(ls.x)
    alpha = np.transpose(r).dot(r) / np.transpose(r).dot(ls.A.dot(r))
    ls.x = ls.x + alpha * r
    return


# Compute the conjugate gradient update
def conjugate_gradient(ls):
    alpha = np.transpose(ls.p).dot(ls.r) / np.transpose(ls.p).dot(ls.A.dot(ls.r))
    ls.x = ls.x + alpha * ls.p
    ls.r = ls.b - ls.A.dot(ls.x)
    beta = np.transpose(ls.p).dot(ls.A.dot(ls.r)) / np.transpose(ls.p).dot(ls.A.dot(ls.p))
    ls.p = ls.r - beta * ls.p
    return


def solve(ls, tol, update, max_iter):

    start = t.time()
    k = 0

    while not converged(ls, tol):
    
        update(ls)
        k += 1

        if(k > max_iter):
            print(pu.bcolors.FAIL
                + "Max iterations reached"
                + pu.bcolors.ENDC )
            break

    end = t.time()

    time_elapsed = end - start

    res = lsr.LSResult(ls.x, time_elapsed, k)

    return res


# Solve the system with each method
def solve_with_each_method(A, b, x, tol, max_iter):

    ls1 = lsh.LinearSystemHelper(A, b)
    jacobi_res = solve(ls1, tol, jacobi, max_iter)
    pu.print_stats(jacobi_res, x, "Jacobi")

    ls2 = lsh.LinearSystemHelper(A, b)
    gauss_seidel_res = solve(ls2, tol, gauss_seidel, max_iter)
    pu.print_stats(gauss_seidel_res, x, "Gauss-Seidel")

    ls3 = lsh.LinearSystemHelper(A, b)
    gradient_descent_res = solve(ls3, tol, gradient_descent, max_iter)
    pu.print_stats(gradient_descent_res, x, "Gradient Descent")

    ls4 = lsh.LinearSystemHelper(A, b, conjugate_gradient = True)
    conjugate_gradient_res = solve(ls4, tol, conjugate_gradient, max_iter)
    pu.print_stats(conjugate_gradient_res, x, "Conjugate Gradient", True)

