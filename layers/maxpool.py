import numpy as np


class MaxPoolLayer(object):
    def __init__(self, size=2):
        """
        MaxPool layer
        Ok to assume non-overlapping regions
        """
        self.locs = None  # to store max locations
        self.size = size  # size of the pooling

    def forward(self, x):
        """
        Compute "forward" computation of max pooling layer

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of input channels x number of rows x number of columns

        Returns
        -------
        np.array
            The output of the maxpooling

        Stores
        -------
        self.locs : np.array
             The locations of the maxes (needed for back propagation)
        """
        result = np.zeros((x.shape[0], x.shape[1], x.shape[2] // self.size, x.shape[3] // self.size))
        location = np.zeros((x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
        r_e, c_e = self.size * (x.shape[2] // self.size), self.size * (x.shape[3] // self.size)
        r_s, c_s = 0, 0
        maxi = 0
        for i in xrange(x.shape[0]):
            for j in xrange(x.shape[1]):
                m, n = x[i][j].shape[:2]
                ny = m // self.size
                nx = n // self.size
                mat_pad = x[i, j, :ny * self.size, :nx * self.size, ...]
                new_shape = (ny, self.size, nx, self.size) + x[i][j].shape[2:]
                result[i][j] = np.nanmax(mat_pad.reshape(new_shape), axis=(1, 3))

                for k in xrange(r_s, r_e - self.size + 1, self.size):
                    for l in xrange(c_s, c_e - self.size + 1, self.size):
                        temp = x[i, j, k:k + self.size, l:l + self.size]
                        index1, index2 = np.unravel_index(np.argmax(temp, axis=None), temp.shape)
                        # print index1, index2
                        # t_index1 = index1 + k
                        # t_index2 = index2 + l
                        # location[i, j, t_index1, t_index2] = 1
                        for in1 in xrange(self.size):
                            for in2 in xrange(self.size):
                                if temp[in1, in2] == temp[index1, index2]:
                                    location[i, j, in1 + k, in2 + l] = 1

        self.locs = np.copy(location)

        return result

        #raise NotImplementedError

    def backward(self, y_grad):
        """
        Compute "backward" computation of maxpool layer

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input
        """
        output = np.zeros((self.locs.shape[0], self.locs.shape[1], self.locs.shape[2], self.locs.shape[3])).astype('float64')
        for i in xrange(y_grad.shape[0]):
            for j in xrange(y_grad.shape[1]):
                for q in xrange(0, y_grad.shape[2]):
                    for k in xrange(0, y_grad.shape[3]):
                        for in1 in xrange(q * self.size, q * self.size + self.size):
                            for in2 in xrange(k * self.size, k * self.size + self.size):
                                if self.locs[i, j, in1, in2] == 1:
                                    output[i, j, in1, in2] = y_grad[i, j, q, k]

        return output

        #raise NotImplementedError

    def update_param(self, lr):
        pass
