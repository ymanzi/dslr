import numpy as np
from random import shuffle

def data_spliter(x: np.ndarray, y: np.ndarray, proportion: float):
    """
            Shuffles and splits the dataset (given by x and y) into a training and a test set, while respecting the given proportion of examples to be kept in the traning set.
        Args:
            x: has to be an numpy.ndarray, a matrix of dimension m * n.
            y: has to be an numpy.ndarray, a vector of dimension m * 1.
        proportion: has to be a float, the proportion of the dataset that will be assigned to the training set.
        Returns:
            (x_train, y_train, x_test, y_test) as a tuple of numpy.ndarray
            None if x or y is an empty numpy.ndarray.
            None if x and y do not share compatible dimensions.
        Raises:
            This function should not raise any Exception.
    """
    if x.size == 0 or y.size == 0 or x.shape[0] != y.shape[0]:
        return None
    random_zip = list(zip(x.tolist(), y))
    shuffle(random_zip)
    new_x = []
    new_y = []
    for e1, e2 in random_zip:
        new_x.append(e1)
        new_y.append(e2)
    new_x = np.array(new_x)
    new_y = np.array(new_y)
    proportion_position = int(x.shape[0] * proportion)
    ret_array = []
    ret_array.append(new_x[:proportion_position])
    ret_array.append(new_y[:proportion_position])
    ret_array.append(new_x[proportion_position:])
    ret_array.append(new_y[proportion_position:])
    return np.array(ret_array, dtype=np.ndarray)

def zscore_normalization(x):
    def mean(x):
        return float(sum(x) / len(x))

    def std(x):
        mean = float(sum(x) / len(x))
        f = lambda x: (x - mean)**2
        tmp_lst = list(map(f, x))
        return float(sum(tmp_lst) / len(x)) ** (0.5)

    mean = mean(x)
    standard_deviation = std(x)
    f = lambda x: (x - mean) / standard_deviation 
    return np.array(list(map(f, x)))

def minmax_normalization(x):
    array_min = min(x)
    array_max = max(x)
    diff_max_min = array_max - array_min
    f = lambda x: (x - array_min) / diff_max_min
    return np.array(list(map(f, x)))

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