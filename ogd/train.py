import time
import matplotlib.pyplot as plt
import torch
from dataset import task1_dataset, task2_dataset, task1_dataloader, task2_dataloader
from net import net, Model
from log import writer, logger
from ogd import OGD_Util
from test import test


epoch_num = 1000


def train(
    name="task1",
    dataloader=task1_dataloader,
    model=net,
    loss_func=torch.nn.NLLLoss(),
    optimizer_func=torch.optim.SGD,
    learning_rate=1e-3,
    epoch=epoch_num,
    ogd_util=None,
):
    global_step = 0.0
    optimizer = optimizer_func(model.parameters(), lr=learning_rate)
    for i in range(epoch):
        for batch, (inputs, labels) in enumerate(dataloader):
            _, acc, loss = model.predict_acc_loss(inputs, labels, loss_func=loss_func)
            optimizer.zero_grad()
            loss.backward()
            if ogd_util is not None:
                grad = model.get_grad_vector()
                grad = ogd_util.orthogonalize(grad)
                model.set_grad_vector(grad)
            optimizer.step()

            global_step += 1
            logger.debug(
                "[Epoch {}][Batch {}]train_acc: {}, train_loss: {}".format(
                    i, batch, acc.item(), loss.item()
                )
            )

            # add metric
            tag = ""
            if ogd_util is not None:
                tag = "ogd_"
            tag_scalar_dict = {
                name + "_" + tag + "train_acc": acc.item(),
                name + "_" + tag + "train_loss": loss.item(),
            }
            if name == "task1":
                test_acc, test_loss = test(dataloader=task2_dataloader)
                tag_scalar_dict[name + "_" + tag + "task2_test_acc"] = test_acc
                tag_scalar_dict[name + "_" + tag + "task2_test_loss"] = test_loss
            elif name == "task2":
                test_acc, test_loss = test(dataloader=task1_dataloader)
                tag_scalar_dict[name + "_" + tag + "task1_test_acc"] = test_acc
                tag_scalar_dict[name + "_" + tag + "task1_test_loss"] = test_loss
            writer.add_scalars(
                main_tag="training&test",
                tag_scalar_dict=tag_scalar_dict,
                global_step=global_step,
            )

        if (i + 1) % 5 == 0:
            model.draw()
            task1_dataset.draw()
            task2_dataset.draw()
            plt.title(f"{name} epoch {i + 1}")
            plt.pause(0.05)
            plt.show()
            plt.clf()

    if ogd_util is not None:
        ogd_util.save(dataloader=dataloader, model=model)


if __name__ == "__main__":
    ogd_util = OGD_Util()

    xlim = [-8, 5]
    ylim = [-2, 6]
    plt.xlim(*xlim)
    plt.ylim(*ylim)

    plt.ion()
    train(name="task1", dataloader=task1_dataloader)
    plt.ioff()

    plt.ion()
    train(name="task2", dataloader=task2_dataloader)
    plt.ioff()

    net = Model(hidden_size=128)
    plt.ion()
    train(
        name="task1",
        dataloader=task1_dataloader,
        ogd_util=ogd_util,
    )
    plt.ioff()

    plt.ion()
    train(
        name="task2",
        dataloader=task2_dataloader,
        ogd_util=ogd_util,
    )
    plt.ioff()
