"""
=> Your Name: Chung Han Yang

In this script, you need to plot the average training loss vs epoch using a learning rate of 0.1 and a batch size of 128 for 15 epochs.

=> Final accuracy on the test set : 0.78

"""
import numpy as np
from layers.full import FullLayer
from layers.softmax import SoftMaxLayer
from layers.cross_entropy import CrossEntropyLayer
from layers.sequential import Sequential
from layers.relu import ReluLayer
from layers.dataset import cifar100
from layers.conv import ConvLayer
from layers.maxpool import MaxPoolLayer
from layers.flatten import FlattenLayer
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = cifar100(seed=1213346041)
conv1 = ConvLayer(3, 16, 3)
relu1 = ReluLayer()
pool1 = MaxPoolLayer(size=2)
conv2 = ConvLayer(16, 32, 3)
relu2 = ReluLayer()
pool2 = MaxPoolLayer(size=2)
flat1 = FlattenLayer()
full1 = FullLayer(2048, 3)
softmax = SoftMaxLayer()
loss = CrossEntropyLayer()

model = Sequential(
    (conv1,
     relu1,
     pool1,
     conv2,
     relu2,
     pool2,
     flat1,
     full1,
     softmax),
    loss
)
loss = model.fit(x_train, y_train, epochs=15, lr=0.1, batch_size=128)
y_pred = model.predict(x_test)
cnt = 0
for i in xrange(len(y_test)):
    for j in xrange(len(y_test[0])):
        if y_test[i, j] == 1 and j == y_pred[i]:
            cnt += 1
acc = float(cnt) / len(y_test)

print acc

plt.plot(np.arange(1, 16, 1), loss, label='Loss for lr = 0.1')
plt.legend()
plt.show()
