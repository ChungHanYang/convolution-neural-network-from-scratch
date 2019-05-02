import numpy as np


class CrossEntropyLayer(object):
    def __init__(self):
        """
        Constructor
        """
        self.x = None
        self.t = None

    def forward(self, x, t):
        """
        Implements forward pass of cross entropy

        l(x,t) = -1/N * sum(log(x) * t)

        where
        x = input (number of samples x feature dimension)
        t = target with one hot encoding (number of samples x feature dimension)
        N = number of samples (constant)

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x feature dimension
        t : np.array
            The target data (one-hot) of size number of training samples x feature dimension

        Returns
        -------
        np.array
            The output of the loss

        Stores
        -------
        self.x : np.array
             The input data (need to store for backwards pass)
        self.t : np.array
             The target data (need to store for backwards pass)
        """
        self.x = np.copy(x)
        self.t = np.copy(t)
        n = x.shape[0]

        ce = -np.sum(np.log(x)*t)/n

        return ce
        #raise NotImplementedError

    def backward(self, y_grad=None):
        """
        Compute "backward" computation of softmax loss layer

        Returns
        -------
        np.array
            The gradient at the input

        """
        result = np.zeros((self.x.shape[0], self.x.shape[1]))
        nb = self.x.shape[0]
        for i in xrange(self.x.shape[0]):
            for j in xrange(self.x.shape[1]):
                result[i][j] = (-self.t[i][j])/(self.x[i][j]*nb)

        return result


        #raise NotImplementedError
