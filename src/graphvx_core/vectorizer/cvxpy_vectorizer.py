import cvxpy as cp
import numpy as np

# unvectorize works on numpy arrays

scaling = False

def unvectorize_scalar(x):
    return x[0]


def cp_vectorize_scalar(x):
    return cp.hstack([x]), unvectorize_scalar


def cp_vectorize_vector(x):
    return x, lambda x: x


def unvectorize_matrix(x, n, m):
    return np.reshape(x, (n, m))


def cp_vectorize_matrix(x):
    n, m = x.shape
    unvec = lambda x: unvectorize_matrix(x, n, m)
    return cp.vec(x, order="C"), unvec


def unvectorize_sdp_matrix(x, n):
    mat = np.zeros((n, n))
    index = 0

    for i in range(n):
        for j in range(i, n):
            if i == j:
                factor = 1. / np.sqrt(2) if scaling else 1.
                mat[i, j] = x[index] * factor
            else:
                mat[i, j] = x[index]
                mat[j, i] = x[index]
            index += 1

    return mat


def cp_vectorize_sdp_matrix(x):
    n, m = x.shape
    if n != m:
        raise ValueError("Input matrix must be square.")

    vec = []

    for i in range(n):
        for j in range(i, n):
            if i == j:
                factor = np.sqrt(2) if scaling else 1.
                vec.append(factor * x[i, j])
            else:
                vec.append(x[i, j])

    def unvec_fn(v):
        return unvectorize_sdp_matrix(v, n)

    return cp.hstack(vec), unvec_fn


class cpVectorizer():

    def __init__(self):
        pass

    @staticmethod
    def auto(x):
        s = x.shape
        if len(s) == 0:
            return cp_vectorize_scalar(x)
        elif len(s) == 1:
            return cp_vectorize_vector(x)
        elif cpVectorizer.is_psd_matrix(x):
            return cp_vectorize_sdp_matrix(x)
        elif len(s) == 2:
            return cp_vectorize_matrix(x)

    @staticmethod
    def get_var_size(v):
        # return v.size
        if len(v.shape) == 2 and cpVectorizer.is_psd_matrix(v):
            n = v.shape[0]
            return int((n) * (n + 1) / 2)
        else:
            return v.size[0] if hasattr(v.size, '__len__') else v.size

    @staticmethod
    def is_psd_matrix(var):

        # Check if the variable is square
        if var.shape[0] != var.shape[1]:
            return False

        # Check if the variable has the PSD attribute
        return var.attributes.get('PSD', False)
