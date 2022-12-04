import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
from util import get_module_grad, set_module_grad


class Model(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.actfunc = nn.ReLU(True)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.actfunc(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

    def predict(self, inputs):
        output = self.forward(inputs)
        return torch.argmax(output, -1)

    def predict_acc_loss(self, inputs, labels, loss_func=torch.nn.NLLLoss()):
        output = self.forward(inputs)
        pred = torch.argmax(output, -1)
        acc = torch.sum(pred.eq(labels)) / output.shape[0]
        loss = loss_func(output, labels)
        return pred, acc, loss

    def draw(self, xRange=[-8, 5], yRange=[-2, 6], step=0.1):
        xx, yy = np.meshgrid(
            np.arange(xRange[0], xRange[1], step), np.arange(yRange[0], yRange[1], step)
        )
        z = self.predict(torch.tensor(np.c_[xx.ravel(), yy.ravel()]).float())
        z = z.reshape(xx.shape)
        plt.contourf(xx, yy, z, cmap=plt.cm.viridis)

    def get_grad_vector(self):
        return get_module_grad(self)

    def set_grad_vector(self, grad):
        set_module_grad(self, grad)


net = Model(hidden_size=256)


if __name__ == "__main__":
    from torchinfo import summary

    # summary(net, (2, 1, 2))
    # x = torch.randn(2, 1, 2)
    # y = net.forward(x)
    # print(x)
    # print(x.shape)
    # print(y)
    # print(y.shape)

    # net.get_grad_vector()
