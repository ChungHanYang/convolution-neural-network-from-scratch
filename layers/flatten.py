import numpy as np

class FlattenLayer(object):
    def __init__(self):
        """
        Flatten layer
        """
        self.orig_shape = None # to store the shape for backpropagation

    def forward(self, x):
        """
        Compute "forward" computation of flatten layer

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of input channels x number of rows x number of columns

        Returns
        -------
        np.array
            The output of the flatten operation
            size = training samples x (number of input channels * number of rows * number of columns)
            (should make a copy of the data with np.copy)

        Stores
        -------
        self.orig_shape : list
             The original shape of the data
        """
        self.orig_shape = x.shape
        temp = np.copy(x)
        output = np.reshape(temp, (x.shape[0], x.shape[1]*x.shape[2]*x.shape[3]))
        return output
        #raise NotImplementedError

    def backward(self, y_grad):
        """
        Compute "backward" computation of flatten layer

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input
        """
        temp = np.copy(y_grad)
        output = np.reshape(temp, (self.orig_shape))
        return output
        #raise NotImplementedError

    def update_param(self, lr):
        pass
