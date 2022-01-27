import numpy as np


def bfgs_update(B_k, s_k, y_k):
    dk = s_k
    wk = y_k
    tol1 = 1e-14

    Bd = B_k.dot(dk)
    wTd = np.dot(wk, dk)

    sign = 1. if wTd >= 0. else -1.
    if abs(wTd) > tol1:
        B_new = B_k + np.outer(wk, wk) / wTd
    else:
        B_new = B_k + np.outer(wk, wk) / sign / tol1

    dTBd = np.dot(dk, Bd)
    sign = 1. if dTBd >= 0. else -1.
    if abs(dTBd) > tol1:
        B_new -= np.outer(Bd, Bd) / dTBd
    else:
        B_new -= np.outer(Bd, Bd) / sign / tol1

    return B_new