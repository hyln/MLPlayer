import time
import torch
from dataset import test_dataloader
from net import net, forward


def test(
    dataloader=test_dataloader,
    model=net,
    loss_func=torch.nn.CrossEntropyLoss(),
):
    avg_acc, avg_loss, batch_num = 0.0, 0.0, 0
    model.eval()
    with torch.no_grad():
        for _, (images, labels) in enumerate(dataloader):
            _, acc, loss = forward(images, labels, model=model, loss_func=loss_func)
            avg_acc += acc.item()
            avg_loss += loss.item()
            batch_num += 1
    return avg_acc / batch_num, avg_loss / batch_num


if __name__ == "__main__":
    avg_acc, avg_loss = test()
    print("avg_acc: {}, avg_loss: {}".format(avg_acc, avg_loss))
