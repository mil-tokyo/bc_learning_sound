"""
 Implementation of EnvNet [Tokozume and Harada, 2017]
 opt.fs = 16000
 opt.inputLength = 24014

"""

import chainer
import chainer.functions as F
import chainer.links as L
from convbnrelu import ConvBNReLU


class EnvNet(chainer.Chain):
    def __init__(self, n_classes):
        super(EnvNet, self).__init__(
            conv1=ConvBNReLU(1, 40, (1, 8)),
            conv2=ConvBNReLU(40, 40, (1, 8)),
            conv3=ConvBNReLU(1, 50, (8, 13)),
            conv4=ConvBNReLU(50, 50, (1, 5)),
            fc5=L.Linear(50 * 11 * 14, 4096),
            fc6=L.Linear(4096, 4096),
            fc7=L.Linear(4096, n_classes)
        )
        self.train = True

    def __call__(self, x):
        h = self.conv1(x, self.train)
        h = self.conv2(h, self.train)
        h = F.max_pooling_2d(h, (1, 160))
        h = F.swapaxes(h, 1, 2)

        h = self.conv3(h, self.train)
        h = F.max_pooling_2d(h, 3)
        h = self.conv4(h, self.train)
        h = F.max_pooling_2d(h, (1, 3))

        h = F.dropout(F.relu(self.fc5(h)), train=self.train)
        h = F.dropout(F.relu(self.fc6(h)), train=self.train)

        return self.fc7(h)
