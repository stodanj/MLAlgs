import numpy as np

class LinearRegression():
    '''Linear regression solver'''
    def __init__(self, X: np.array, Y: np.array, learning_rate: np.float64 = 0.01, epochs: np.int16 = 1000, loss_threshold: np.float64 = 1e-9):
        self.X = X
        self.Y = Y
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_threshold = loss_threshold
        self.loss_history = []

        # Validate input
        assert X.shape[0] == Y.shape[0], "X and y must have same number of samples"

        # Set initial conditions
        num_features = self.X.shape[1]
        output_dim = self.Y.shape[1]
        self.w = np.random.rand(num_features, output_dim)
        self.b = 0.0

    def MSE(self):
        Y_pred = self.predict()
        return np.mean(np.square(Y_pred - self.Y))
    
    def RMSE(self):
        return np.sqrt(self.MSE())
    
    def predict(self):
        return self.X @ self.w + self.b
    
    def _bias_gradient(self):
        Y_pred = self.predict()
        err = Y_pred - self.Y
        # As y could be multivariate, specify that we are summing across samples only
        return (2 / len(err)) * np.sum(err, axis=0) 
    
    def _weights_gradient(self):
        Y_pred = self.predict()
        err = Y_pred - self.Y
        return (2 / len(err)) * self.X.T @ err 
    
    def _gradient_update(self):
        self.w -= self.learning_rate * self._weights_gradient()
        self.b -= self.learning_rate * self._bias_gradient()

    def fit(self):
        '''Fit using gradient descent'''
        for _ in range(self.epochs):
            self._gradient_update()
            loss = self.RMSE()
            self.loss_history.append(loss)
            if loss < self.loss_threshold:
                return self
        return self

    def score(self):
        '''Return R^2 score'''
        Y_pred = self.predict()
        Y_mean = np.mean(self.Y, axis=0)
        return 1 - np.sum((self.Y - Y_pred)**2) / np.sum((self.Y - Y_mean)**2)    
