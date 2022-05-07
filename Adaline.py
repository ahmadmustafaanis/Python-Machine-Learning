import numpy as np


class Adaline:
    """
        Implements Adaptive Linear Neuron Classifier
    Parameters:
        eta: FLOAT
            Learning Rate
        n_iter: int
            Epochs
        random_state: int
            Randome number generator for weight initlaization

    Attributes:
        w_ : 1d array
            1d Array containing weights
        cost_ : list
            List containg cost for each epoch

    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def net_input(self, X):
        return np.dot(X, self.w_[1, :]) + self.w_[0, :]

    def activation(self, X):
        return X

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=1.0, size=1 + X.shape[1])

        self.cost_ = []

        for _ in range(self.n_iter):
            net_input = self.net_input(X)  # hp = wx + b
            output = self.activation(net_input)

            errors = y - output
            self.w_[1, :] += X.T.dot(errors)
            self.w_[0, :] += errors.sum()
            cost = (errors**2).sum() / 2
            self.cost_.append(cost)

        return self

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
