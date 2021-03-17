import numpy as np
from random import shuffle

class MyLogisticRegression():
    """
    Description:
        My personnal logistic regression to classify things.
    """
    def __init__(self, theta, alpha=0.001, n_cycle=1000, lambda_ = 0.0, penalty='l2'):
        self.alpha = alpha
        self.n_cycle = n_cycle
        self.theta = np.array(theta).reshape(-1, 1)
        self.penalty = penalty
        self.lambda_ = lambda_

    def theta0(self, theta):
        theta[0] = 0
        return theta

    def sigmoid_(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return None
        x = x.astype(np.float)
        if x.ndim == 0:
            x = np.array(x, ndmin=1)
        return (1 / (1 + (np.exp(x * -1))))
    
    def predict_(self, x):
        if (x.ndim == 1):
            x = x.reshape(-1, 1)
        x_plus = np.column_stack((np.full((x.shape[0], self.theta.shape[0] - x.shape[1]) , 1), x))
        return self.sigmoid_(x_plus.dot(self.theta).reshape(-1, 1))

    def cost_(self, x: np.ndarray, y: np.ndarray, eps=1e-15):
        y_hat = self.predict_(x)
        ones = np.ones(y.shape)
        arr_size = y.shape[0]
        if (self.penalty == 'l2'):
            l2 = float(self.theta.transpose().dot(self.theta) - self.theta[0]**2)
        elif self.penalty == 'none':
            l2_cor = 0
        log_loss_array = y.transpose().dot(np.log(y_hat + eps)) + (ones - y).transpose().dot(np.log(ones - y_hat + eps))
        return np.sum(log_loss_array) / ((-1) * arr_size) + self.lambda_ * l2 / (2 * arr_size) 

    def fit_(self, x: np.ndarray, y: np.ndarray):
        # if y.ndim > 1:
        #     y = np.array([elem for lst in y for elem in lst])
        x_plus = np.column_stack((np.full((x.shape[0], self.theta.shape[0] - x.shape[1]) , 1), x))  # add intercept
        for i in range(self.n_cycle):
            y_hat = self.sigmoid_(x_plus.dot(self.theta).reshape(-1, 1)) #self.predict_(x)
            if self.penalty == 'l2':
                l2_cor = self.lambda_ * self.theta0(self.theta)
            elif self.penalty == 'none':
                l2_cor = 0
            gradient = (x_plus.transpose().dot(np.subtract(y_hat, y)) + l2_cor) / y.shape[0]   # to give us the direction to a better theta-i
            self.theta = np.subtract(self.theta, np.multiply(self.alpha, gradient)) # improve theta with alpha-step
        return self.theta

    def add_polynomial_features(self, x: np.ndarray, power: int) -> np.ndarray:
        if x.size == 0:
            return None
        copy_x = x
        for nb in range(2, power + 1):
            x = np.column_stack((x, copy_x ** nb))
        return x