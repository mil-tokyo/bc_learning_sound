import chainer
import chainer.functions as F
import chainer.links as L


class ConvBNReLU(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 initialW=chainer.initializers.HeNormal(), nobias=True):
        super(ConvBNReLU, self).__init__(
            conv=L.Convolution2D(in_channels, out_channels, ksize, stride, pad,
                                 initialW=initialW, nobias=nobias),
            bn=L.BatchNormalization(out_channels)
        )

    def __call__(self, x, train):
        h = self.conv(x)
        h = self.bn(h, test=not train)

        return F.relu(h)
