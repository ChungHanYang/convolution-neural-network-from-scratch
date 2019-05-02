import numpy as np
import scipy.signal


class ConvLayer(object):
    def __init__(self, n_i, n_o, h):
        """
        Convolutional layer

        Parameters
        ----------
        n_i : integer
            The number of input channels
        n_o : integer
            The number of output channels
        h : integer
            The size of the filter
        """
        # glorot initialization
        #raise NotImplementedError

        self.n_i = n_i
        self.n_o = n_o

        self.W_grad = None
        self.b_grad = None
        self.h = h
        fin = n_i * h * h
        fout = n_o * h * h
        self.W = np.random.randn(n_o, n_i, h, h) * np.sqrt(np.sqrt(2 / float(fin + fout)))
        self.b = np.zeros((1, n_o)).astype(float)

    def forward(self, x):
        """
        Compute "forward" computation of convolutional layer

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of input channels x number of rows x number of columns

        Returns
        -------
        np.array
            The output of the convolution

        Stores
        -------
        self.x : np.array
             The input data (need to store for backwards pass)
        """

        self.x = np.copy(x)
        fcon = np.zeros((x.shape[0], self.n_o, x.shape[2], x.shape[3]))

        for i in xrange(x.shape[0]):
            for k in xrange(self.n_o):
                for j in xrange(self.n_i):
                    temp_x = np.pad(x[i, j, :, :], ((self.h / 2, self.h / 2), (self.h / 2, self.h / 2)), 'constant', constant_values=0)

                    fcon[i, k] += (scipy.signal.correlate(temp_x, self.W[k, j, :, :], mode='valid'))
                fcon[i, k] += self.b[0, k]

        return fcon

        #raise NotImplementedError

    def backward(self, y_grad):
        """
        Compute "backward" computation of convolutional layer

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input

        Stores
        -------
        self.b_grad : np.array
             The gradient with respect to b (same dimensions as self.b)
        self.w_grad : np.array
             The gradient with respect to W (same dimensions as self.W
        """
        self.b_grad = np.zeros((1, self.n_o))
        self.W_grad = np.zeros((self.n_o, self.n_i, self.h, self.h))
        x_grad = np.zeros((y_grad.shape[0], self.n_i, y_grad.shape[2], y_grad.shape[3]))
        for z in xrange(y_grad.shape[1]):
            for i in xrange(y_grad.shape[0]):
                for j in xrange(y_grad.shape[2]):
                    for k in xrange(y_grad.shape[3]):
                        self.b_grad[0, z] += y_grad[i, z, j, k]

        for k in xrange(self.n_i):
            for i in xrange(y_grad.shape[0]):
                for j in xrange(y_grad.shape[1]):
                    x_grad[i, k, :, :] += scipy.signal.convolve(y_grad[i, j, :, :], self.W[j, k, :, :], mode='same')

        for i in xrange(y_grad.shape[0]):
            for j in xrange(self.n_o):
                for k in xrange(self.n_i):
                    temp_x = np.pad(self.x[i, k, :, :], ((self.h / 2, self.h / 2), (self.h / 2, self.h / 2)), 'constant', constant_values=0)
                    self.W_grad[j, k, :, :] += scipy.signal.correlate(temp_x, y_grad[i, j, :, :], mode='valid')

        return x_grad
        #raise NotImplementedError

    def update_param(self, lr):
        """
        Update the parameters with learning rate lr

        Parameters
        ----------
        lr : floating point
            The learning rate

        Stores
        -------
        self.W : np.array
             The updated value for self.W
        self.b : np.array
             The updated value for self.b
        """
        self.W = self.W - lr * self.W_grad
        self.b = self.b - lr * self.b_grad
        #raise NotImplementedError
