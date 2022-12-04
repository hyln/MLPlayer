import torch
from torchvision import models
from functorch import make_functional_with_buffers


linear_dim = 4096
class_num = 10

net = models.resnet18()
net.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
net.fc = torch.nn.Sequential(
    torch.nn.Linear(net.fc.in_features, linear_dim),
    torch.nn.ReLU(True),
    torch.nn.Dropout(),
    torch.nn.Linear(linear_dim, linear_dim),
    torch.nn.ReLU(True),
    torch.nn.Dropout(),
    torch.nn.Linear(linear_dim, class_num),
)
net_f, params, buffers = make_functional_with_buffers(net)


def forward(images, labels, model=net, loss_func=torch.nn.CrossEntropyLoss()):
    output = model(images)
    _, pred = torch.max(output, 1)
    acc = torch.sum(torch.Tensor(pred == labels)) / output.shape[0]
    loss = loss_func(output, labels)
    return pred, acc, loss


def net_func(params, buffers, input):
    output = torch.sum(net_f(params, buffers, input), dim=-1)
    return output.reshape((output.shape[0], 1))


if __name__ == "__main__":
    from torchinfo import summary

    summary(net, (1, 1, 28, 28))

    # x = torch.randn(2, 1, 28, 28)
    # print(x)
    # y = net_func(params, buffers, x)
    # print(y)
    # print(y.shape)
    # n = 0
    # for params in params:
    #     # print(params)
    #     n += 1
    # print(n)

    # from dataset import train_dataloader, test_dataloader, test_dataset
    # n = 0
    # for batch, (images, labels) in enumerate(train_dataloader):
    #     pred, acc, loss = forward(images, labels)
    #     print("pred: {}".format(pred))
    #     print("acc: {}".format(acc))
    #     print("loss: {}".format(loss))
    #     n = n + 1
    # print(n)
    # n = 0
    # for batch, (images, labels) in enumerate(test_dataloader):
    #     pred, acc, loss = forward(images, labels)
    #     print("pred: {}".format(pred))
    #     print("acc: {}".format(acc))
    #     print("loss: {}".format(loss))
    #     n = n + 1
    # print(n)
