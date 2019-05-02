import numpy as np


class SoftMaxLayer(object):
    def __init__(self):
        """
        Constructor
        """
        self.y = None

    def forward(self, x):
        """
        Implement forward pass of softmax

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
             The output of the layer (needed for backpropagation)
        """
        self.x = np.copy(x)
        max_element = np.amax(x, axis=1)
        total = np.zeros((x.shape[0], 1)).astype(float)
        y = np.zeros((x.shape[0], x.shape[1])).astype(float)

        for i in xrange(len(x)):
            x[i] -= max_element[i]
            total[i] = np.exp(x[i]).sum()
        for j in xrange(len(total)):
            y[j] = np.exp(x[j])/total[j]
        self.y = np.copy(y)
        return y

        #raise NotImplementedError

    def backward(self, y_grad):
        """
        Compute "backward" computation of softmax

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input

        """
        #result = np.zeros((y_grad.shape[0], y_grad.shape[1]))
        result = np.zeros((len(self.y), len(self.y[1])))
        for i in xrange(len(self.y)):
            s = self.y[i].reshape(-1, 1)
            k = np.dot(y_grad[i], np.diagflat(s) - np.dot(s, s.T))
            for j in xrange(len(k)):
                result[i][j] = k[j]

        return result







        return result


        #raise NotImplementedError

    def update_param(self, lr):
        pass  # no learning for softmax layer
# x = np.array([[0.5, 0.6, 0.4, 0.8],
#               [0.3, 0.2, 0.1, 0.43],
#               [0.7, 0.5, 0.6, 0.55]])
# layer = SoftMaxLayer()
# print layer.forward(x)
# #print layer.backward(x)