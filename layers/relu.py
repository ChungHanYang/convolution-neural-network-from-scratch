import numpy as np


class ReluLayer(object):
    def __init__(self):
        """
        Rectified Linear Unit
        """
        self.y = None

    def forward(self, x):
        """
        Implement forward pass of Relu

        y = x if x > 0
        y = 0 otherwise

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number of features

        Returns
        -------
        np.array
            The output of the layer

        Stores
        -------
        self.y : np.array
             The output data (need to store for backwards pass)
        """
        if len(x.shape) == 2:
            y = np.zeros((x.shape[0], x.shape[1]))
            for i in xrange(x.shape[0]):
                for j in xrange(x.shape[1]):
                    if x[i][j] > 0:
                        y[i][j] = x[i][j]
                    else:
                        y[i][j] = 0
            self.y = np.copy(y)
            return y
        else:
            y = np.zeros((x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
            for k in xrange(x.shape[0]):
                for z in xrange(x.shape[1]):
                    for i in xrange(x.shape[2]):
                        for j in xrange(x.shape[3]):
                            if x[k, z, i, j] > 0:
                                y[k, z, i, j] = x[k, z, i, j]
                            else:
                                y[k, z, i, j] = 0
            self.y = np.copy(y)
            return y
        #raise NotImplementedError

    def backward(self, y_grad):
        """
        Implement backward pass of Relu

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input
        """
        if len(y_grad.shape) == 2:
            result = np.zeros((y_grad.shape[0], y_grad.shape[1]))
            for i in xrange(y_grad.shape[0]):
                for j in xrange(y_grad.shape[1]):
                    if self.y[i][j] > 0:
                        result[i][j] = 1
                        result[i][j] *= y_grad[i][j]

                    else:
                        result[i][j] = 0

            return result
        else:
            result = np.zeros((y_grad.shape[0], y_grad.shape[1], y_grad.shape[2], y_grad.shape[3]))
            for i in xrange(y_grad.shape[0]):
                for j in xrange(y_grad.shape[1]):
                    for q in xrange(y_grad.shape[2]):
                        for k in xrange(y_grad.shape[3]):
                            if self.y[i, j, q, k] > 0:
                                result[i, j, q, k] = 1
                                result[i, j, q, k] *= y_grad[i, j, q, k]

                            else:
                                result[i, j, q, k] = 0

            return result
        #raise NotImplementedError

    def update_param(self, lr):
        pass  # no parameters to update
