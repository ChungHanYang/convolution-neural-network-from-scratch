import numpy as np


class FullLayer(object):
    def __init__(self, n_i, n_o):
        """
        Fully connected layer

        Parameters
        ----------
        n_i : integer
            The number of inputs
        n_o : integer
            The number of outputs
        """
        self.x = None
        self.W_grad = None
        self.b_grad = None
        self.n_i = n_i
        self.n_o = n_o
        # need to initialize self.W and self.b
        self.W = np.transpose(np.random.randn(n_i, n_o)*np.sqrt(np.sqrt(2/float(n_i + n_o))))
        self.b = np.zeros((1, n_o)).astype(float)
        #raise NotImplementedError

    def forward(self, x):
        """
        Compute "forward" computation of fully connected layer

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
        self.x : np.array
             The input data (need to store for backwards pass)
        """
        self.x = np.copy(x)
        fx = np.dot(x, self.W.T) + self.b

        return fx

        #raise NotImplementedError

    def backward(self, y_grad):
        """
        Compute "backward" computation of fully connected layer

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output nb*n0

        Returns
        -------
        np.array
            The gradient at the input

        Stores
        -------
        self.b_grad : np.array
             The gradient with respect to b (same dimensions as self.b)
        self.W_grad : np.array
             The gradient with respect to W (same dimensions as self.W
        """

        one = np.ones((1, y_grad.shape[0]))
        self.b_grad = np.dot(one, y_grad)#np.sum(y_grad, axis=0, keepdims=True)
        self.W_grad = np.dot(y_grad.T, self.x)
        x_grad = np.dot(y_grad, self.W)


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
        self.W = self.W - lr*self.W_grad
        self.b = self.b - lr*self.b_grad


        #raise NotImplementedError
