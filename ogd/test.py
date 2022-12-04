import time
import torch
from dataset import task1_dataloader
from net import net


def test(
    dataloader=task1_dataloader,
    model=net,
    loss_func=torch.nn.NLLLoss(),
):
    avg_acc, avg_loss, batch_num = 0.0, 0.0, 0
    model.eval()
    with torch.no_grad():
        for _, (inputs, labels) in enumerate(dataloader):
            _, acc, loss = model.predict_acc_loss(inputs, labels, loss_func=loss_func)
            avg_acc += acc.item()
            avg_loss += loss.item()
            batch_num += 1
    return avg_acc / batch_num, avg_loss / batch_num


if __name__ == "__main__":
    avg_acc, avg_loss = test()
    print("avg_acc: {}, avg_loss: {}".format(avg_acc, avg_loss))
