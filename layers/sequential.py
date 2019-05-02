from __future__ import print_function
import numpy as np
from full import FullLayer
from softmax import SoftMaxLayer
from cross_entropy import CrossEntropyLayer
from relu import ReluLayer
from conv import ConvLayer
from maxpool import MaxPoolLayer
from flatten import FlattenLayer


class Sequential(object):
    def __init__(self, layers, loss):
        """
        Sequential model

        Implements a sequence of layers

        Parameters
        ----------
        layers : list of layer objects
        loss : loss object
        """
        self.layers = layers
        self.loss = loss

    def forward(self, x, target=None):
        """
        Forward pass through all layers

        if target is not none, then also do loss layer

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number of features
        target : np.array
            The target data of size number of training samples x number of features (one-hot)

        Returns
        -------
        np.array
            The output of the model
        """
        inp = np.copy(x)
        for obj in self.layers:
            inp = obj.forward(inp)

        try:
            if len(target) > 0:
                result = self.loss.forward(inp, target)
                return result

        except:
            return inp
        #raise NotImplementedError

    def backward(self):
        """
        Compute "backward" computation of fully connected layer

        Returns
        -------
        np.array
            The gradient at the input

        """

        grads = self.loss.backward()

        for i in reversed(self.layers):
            grads = i.backward(grads)
        return grads
        #raise NotImplementedError

    def update_param(self, lr):
        """
        Update the parameters with learning rate lr

        Parameters
        ----------
        lr : floating point
            The learning rate
        """
        for obj in self.layers:
            obj.update_param(lr)
        #raise NotImplementedError

    def fit(self, x, y, epochs, lr, batch_size):
        """
        Fit parameters of all layers using batches

        Parameters
        ----------
        x : numpy matrix
            Training data (number of samples x number of features)
        y : numpy matrix
            Training labels (number of samples x number of features) (one-hot)
        epochs: integer
            Number of epochs to run (1 epoch = 1 pass through entire data)
        lr: float
            Learning rate
        batch_size: integer
            Number of data samples per batch of gradient descent
        """
        n_batch = x.shape[0] / batch_size

        loss = np.zeros(epochs)
        b_loss = 0.0
        for i in xrange(epochs):
            print("Start Epoch:", i + 1)
            for j in xrange(1, n_batch + 2):
                if j <= n_batch:
                    b_loss += self.forward(x[(j - 1) * batch_size:j * batch_size], y[(j - 1) * batch_size:j * batch_size])
                    self.backward()
                    self.update_param(lr)
                    print ("Epoch", i + 1, "complete iteration:", j)
                else:
                    b_loss += self.forward(x[(j - 1) * batch_size:], y[(j - 1) * batch_size:])
                    self.backward()
                    self.update_param(lr)
                    print ("Epoch", i + 1, "complete iteration:", j)

            print("Complete Epoch:", i + 1)
            loss[i] = b_loss / batch_size
            b_loss = 0.0
        return loss

        #raise NotImplementedError

    def predict(self, x):
        """
        Return class prediction with input x

        Parameters
        ----------
        x : numpy matrix
            Testing data data (number of samples x number of features)

        Returns
        -------
        np.array
            The output of the model (integer class predictions)
        """
        prob = np.copy(x)
        y = np.zeros((x.shape[0], 1))
        prob = self.forward(prob)
        for i in xrange(prob.shape[0]):
            y[i] = np.argmax(prob[i])

        return y
        #raise NotImplementedError
