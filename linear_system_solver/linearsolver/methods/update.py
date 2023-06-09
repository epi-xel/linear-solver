import scipy as sp
import numpy as np


# Compute the Jacobi update
def jacobi(ls):
    inv_P = np.diag(1 / ls.A.diagonal())
    ls.x = ls.x - inv_P.dot(ls.A.dot(ls.x) - ls.b)


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
    # y = sp.sparse.linalg.spsolve_triangular(P.tocsr(), r)
    y = forward_substitution(P, r)
    ls.x = ls.x + y


# Compute the gradient descend update
def gradient(ls):
    r = ls.b - ls.A.dot(ls.x)
    alpha = np.transpose(r).dot(r) / np.transpose(r).dot(ls.A.dot(r))
    ls.x = ls.x + alpha * r


# Compute the conjugate gradient update
def conjugate_gradient(ls):
    alpha = np.transpose(ls.p).dot(ls.r) / np.transpose(ls.p).dot(ls.A.dot(ls.r))
    ls.x = ls.x + alpha * ls.p
    ls.r = ls.b - ls.A.dot(ls.x)
    beta = np.transpose(ls.p).dot(ls.A.dot(ls.r)) / np.transpose(ls.p).dot(ls.A.dot(ls.p))
    ls.p = ls.r - beta * ls.p      