from os import listdir
from os.path import isfile, join
import xlrd
import pandas as pd
import numpy as np
import pickle


def choose_n(D):
    """
    Determine the appropriate number for lambda. Return the number suggested by
    the ratio-based method, and the percentile vector.
    """
    num = D.shape[0]
    D = D.ravel()
    ratio = D[1:] / D[0:-1]
    n_ratio = np.argmin(ratio[0:int(num / 2)]) + 1

    c_D = D.cumsum()
    cum_ratio = c_D / D.sum()

    return n_ratio, cum_ratio


def varimax(Phi, gamma=1.0, q=20, tol=1e-6):
    from scipy import eye, asarray
    from scipy.linalg import svd
    from numpy import diag
    p, k = Phi.shape
    R = eye(k)
    d = 0
    for i in range(q):
        d_old = d
        Lambda = Phi @ R
        u, s, vh = svd(Phi.T @ (asarray(Lambda) ** 3 - (gamma / p) * Lambda @ diag(diag(Lambda.T @ Lambda))))
        R = u @ vh
        d = s.sum()
        if d_old != 0 and d / d_old < 1 + tol: break
    return Phi @ R, R


def factorize1(m_series, h_list=None):
    """
    Matrix time series factorization with two the same loading matrix

    Parameters
    ----------
    m_series: ndarray
        The T*n*n matrix time series.
    h_list: list
        A list of time lag, default [1]

    Returns
    -------
    The n*n orthogonal loading matrix `Q`, and `D` with n eigenvalues.
    """
    if h_list is None:
        h_list = [1]
    m_series = m_series.copy()
    m_series[np.isnan(m_series)] = 0
    T, n, _ = m_series.shape
    M = np.zeros([n, n])
    m = m_series.mean(axis=0)

    for h in h_list:
        M_h = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                c_t = m_series[0:T - h, :, i]
                c_h = m_series[h:T, :, j]
                r_t = m_series[0:T - h, i, :]
                r_h = m_series[h:T, j, :]

                o_c = (c_t - m[:, i].reshape(1, -1)).T @ (c_h - m[:, j].reshape(1, -1))
                o_r = (r_t - m[i, :].reshape(1, -1)).T @ (r_h - m[j, :].reshape(1, -1))

                M_h += o_c @ o_c.T + o_r @ o_r.T
        M += M_h / (T - h)

    D, Q = np.linalg.eig(M)
    idx_D = np.argsort(D)[::-1]
    D = D[idx_D]
    Q = Q[:, idx_D]
    return Q, D


def factorize2(m_series, h_list=None):
    """
    Matrix time series factorization with two different loading matrices

    Parameters
    ----------
    m_series: ndarray
        The T*n*n matrix time series.
    h_list: list
        A list of time lag, default [1]

    Returns
    -------
    The n*n orthogonal loading matrix `Q1` and 'Q2', and `D1` and `D2` with n eigenvalues.
    """
    if h_list is None:
        h_list = [1]
    m_series = m_series.copy()
    m_series[np.isnan(m_series)] = 0
    T, n, _ = m_series.shape
    M_c = np.zeros([n, n])
    M_r = np.zeros([n,n])
    m = m_series.mean(axis=0)

    for h in h_list:
        M_c_h = np.zeros([n, n])
        M_r_h = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                c_t = m_series[0:T - h, :, i]
                c_h = m_series[h:T, :, j]
                r_t = m_series[0:T - h, i, :]
                r_h = m_series[h:T, j, :]

                o_c = (c_t - m[:, i].reshape(1, -1)).T @ (c_h - m[:, j].reshape(1, -1))
                o_r = (r_t - m[i, :].reshape(1, -1)).T @ (r_h - m[j, :].reshape(1, -1))

                M_c_h += o_c @ o_c.T
                M_r_h += o_r @ o_r.T
        M_c += M_c_h / (T-h)
        M_r += M_r_h / (T-h)

    D1, Q1 = np.linalg.eig(M_c)
    idx = np.argsort(D1)[::-1]
    D1 = D1[idx]
    Q1 = Q1[:, idx]

    D2, Q2 = np.linalg.eig(M_r)
    idx = np.argsort(D2)[::-1]
    D2 = D2[idx]
    Q2 = Q2[:, idx]
    return Q1, D1, Q2, D2