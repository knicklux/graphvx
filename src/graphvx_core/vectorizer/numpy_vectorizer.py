import numpy as np

# unvectorize works on numpy arrays

def unvectorize_scalar(x):
    return x[0]


def np_vectorize_scalar(x):
    return np.hstack([x]), unvectorize_scalar


def np_vectorize_vector(x):
    return x, lambda x: x


def unvectorize_matrix(x, n, m):
    return np.reshape(x, (n, m))


def np_vectorize_matrix(x):
    n, m = x.shape
    unvec = lambda x: unvectorize_matrix(x, n, m)
    return x.flatten(order="C"), unvec


class npVectorizer():

    def __init__(self):
        self.vector = []
        self.offsets = []
        self.unvector = []

    def add_variable(self, x, vectorization=None):
        if vectorization is None:
            vectorization = npVectorizer.auto
        x_vec, unvec = vectorization(x)
        self.vector.append(x_vec)
        self.offsets.append(x_vec.shape[0])
        self.unvector.append(unvec)

    def vectorize(self):
        return cp.hstack(self.vector)

    def unvectorize(self, result):
        results = []
        current_offset = 0
        for i in range(len(self.vector)):
            val = result[current_offset: current_offset + self.offsets[i]]
            val = self.unvector[i](val)
            results.append(val)
            current_offset += self.offsets[i]
        return results

    @staticmethod
    def auto(x):
        s = x.shape
        if len(s) == 0:
            return np_vectorize_scalar(x)
        if len(s) == 1:
            return np_vectorize_vector(x)
        if len(s) == 2:
            return np_vectorize_matrix(x)
