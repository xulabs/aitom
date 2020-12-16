import chainer
from chainer import links as L
from chainer import functions as F
from chainer.functions.evaluation import accuracy
from chainer import reporter


# modify from https://github.com/pfnet-research/chainer-ADDA.git


class Encoder(chainer.Chain):
    def __init__(self, h=256, dropout=0.5):
        self.dropout = dropout
        super(Encoder, self).__init__(
            conv1=L.ConvolutionND(3, 1, 8, ksize=5),
            conv2=L.ConvolutionND(3, 8, 16, ksize=5),
            conv3a=L.ConvolutionND(3, 16, 32, ksize=4),
            conv3b=L.ConvolutionND(3, 32, 64, ksize=4),
            conv4a=L.ConvolutionND(3, 64, 128, ksize=3),
            conv4b=L.ConvolutionND(3, 128, 128, ksize=3),
            conv5a=L.ConvolutionND(3, 128, 128, ksize=1),
            conv5b=L.ConvolutionND(3, 128, 256, ksize=1),
            fc=L.Linear(None, 500))

    def __call__(self, x):
        h = F.max_pooling_nd(F.relu(self.conv1(x)), 1, 1)
        h = F.max_pooling_nd(F.relu(self.conv2(h)), 2, 2)
        h = F.relu(self.conv3a(h))
        h = F.max_pooling_nd(F.relu(self.conv3b(h)), 2, 2)
        h = F.relu(self.conv4a(h))
        h = F.max_pooling_nd(F.relu(self.conv4b(h)), 2, 2)
        h = F.relu(self.conv5a(h))
        h = F.max_pooling_nd(F.relu(self.conv5a(h)), 2, 2)
        h = F.dropout(F.relu(self.fc(h)), ratio=self.dropout)
        return h


class Discriminator(chainer.Chain):
    def __init__(self, h=500):
        super(Discriminator, self).__init__(
            l1=L.Linear(None, h),
            l2=L.Linear(h, h),
            l3=L.Linear(h, 2))

    def __call__(self, x):
        l1 = F.leaky_relu(self.l1(x))
        l2 = F.leaky_relu(self.l2(l1))
        out = self.l3(l2)
        return out


class Classifier(chainer.Chain):
    def __init__(self, num_classes, dropout=0.5):
        self.dropout = dropout
        super(Classifier, self).__init__(
            l1=L.Linear(None, num_classes))

    def __call__(self, x):
        prediction = self.l1(x)
        return prediction


class Loss(chainer.Chain):
    def __init__(self, num_classes):
        super(Loss, self).__init__(
            encoder=Encoder(),
            classifier=Classifier(num_classes))

    def __call__(self, x, t):
        encode = self.encoder(x)
        classify = self.classifier(encode)

        self.accuracy = accuracy.accuracy(classify, t)
        self.loss = F.softmax_cross_entropy(classify, t)

        reporter.report({"accuracy": self.accuracy, "loss": self.loss}, self)
        return self.loss
