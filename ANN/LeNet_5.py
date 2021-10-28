import numpy as np
from ANN.Layer import *

class LeNet5:
    def __init__(self):
        self.conv1 = Conv2d(1, 6, 5, stride=1, padding=2)
        self.pooling1 = MaxPooling(kernel_size=2, stride=2)
        self.BN1 = BN(6, momentum=0.9, is_train=True)
        self.relu1 = Relu()

        self.conv2 = Conv2d(6, 16, 5, stride=1, padding=0)
        self.pooling2 = MaxPooling(kernel_size=2, stride=2)
        self.BN2 = BN(16, momentum=0.9, is_train=True)
        self.relu2 = Relu()

        self.conv3 = Conv2d(16, 120, 5, stride=1, padding=0)

        self.fc4 = Fc(120*1*1, 84)
        self.relu4 = Relu()
        self.fc5 = Fc(84, 10)

        self.softmax = Softmax_CrossEntropy()

    def forward(self, input, labels, is_train=True):
        x = self.conv1.forward(input)
        x = self.pooling1.forward(x)
        x = self.BN1.forward(x, is_train)
        x = self.relu1.forward(x)

        x = self.conv2.forward(x)
        x = self.pooling2.forward(x)
        x = self.BN2.forward(x, is_train)
        x = self.relu2.forward(x)

        x = self.conv3.forward(x)

        x = self.fc4.forward(x)
        x = self.relu4.forward(x)
        x = self.fc5.forward(x)

        x = self.softmax.forward(x)
        loss = self.softmax.calculate_loss(x, labels)

        return loss, x

    def backward(self, learning_rate):
        eta = self.softmax.gradient()

        eta = self.fc5.backward(eta, learning_rate)
        eta = self.relu4.backward(eta)
        eta = self.fc4.backward(eta, learning_rate)

        eta = self.conv3.backward(eta, learning_rate)

        eta = self.relu2.backward(eta)
        eta = self.BN2.backward(eta, learning_rate)
        eta = self.pooling2.backward(eta)
        eta = self.conv2.backward(eta, learning_rate)

        eta = self.relu1.backward(eta)
        eta = self.BN1.backward(eta, learning_rate)
        eta = self.pooling1.backward(eta)
        eta = self.conv1.backward(eta, learning_rate)
