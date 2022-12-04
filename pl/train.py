import time
import torch
from dataset import train_dataloader, ntk_dataloader
from net import net, forward
from test import test
from log import logger, writer
import norm
import ntk

epoch_num = 20
ntk_freq = 100


def train(
    dataloader=train_dataloader,
    model=net,
    loss_func=torch.nn.CrossEntropyLoss(),
    optimizer_func=torch.optim.SGD,
    learning_rate=1e-3,
    epoch=epoch_num,
):
    global_step = 0.0
    optimizer = optimizer_func(model.parameters(), lr=learning_rate)
    for i in range(epoch):
        epoch_start = time.perf_counter()
        avg_acc, avg_loss, batch_num = 0.0, 0.0, 0
        for batch, (images, labels) in enumerate(dataloader):
            batch_start = time.perf_counter()
            # forward & backward
            _, acc, loss = forward(images, labels, model=model, loss_func=loss_func)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # add metric for batch
            avg_acc += acc.item()
            avg_loss += loss.item()
            writer.add_scalars(
                main_tag="batch training",
                tag_scalar_dict={
                    "train_acc": acc.item(),
                    "train_loss": loss.item(),
                },
                global_step=global_step,
            )
            batch_num += 1
            global_step += 1
            logger.debug(
                "[Epoch {}][Batch {}][Cost: {}s] train_acc: {}, train_loss: {}".format(
                    i, batch, time.perf_counter() - batch_start, acc.item(), loss.item()
                )
            )

            # add global gradient norm, weight norm, miu, lf, cn
            global_miu = norm.log_global_grad_norm_miu_by_def(
                net.parameters(), loss, global_step
            )
            global_lf = norm.log_global_weight_norm_lf_by_def(
                net.parameters(), loss, global_step
            )
            writer.add_scalars(
                main_tag="global_cn_by_def",
                tag_scalar_dict={
                    "global_cn": global_lf / global_miu,
                },
                global_step=global_step,
            )

            # add local gradient norm, weight norm, miu, lf, cn
            norm.log_local_cn_by_def(net.named_parameters(), loss, global_step)

            # add ntk visualization
            if global_step % ntk_freq == 0:
                for _, (input, _) in enumerate(ntk_dataloader):
                    start = time.perf_counter()
                    ntk.log_ntk(input, global_step)
                    cost = time.perf_counter() - start
                    logger.debug(
                        "[Epoch {}][Batch {}][NTK Cost: {}s]".format(i, batch, cost)
                    )
                    writer.add_scalar(
                        tag="ntk time cost",
                        scalar_value=cost,
                        global_step=global_step,
                    )
                    break

            # add metric for gradient norm
            # norm.log_global_grad_norm_miu_by_def(net.parameters(), loss, global_step)
            # norm.log_local_grad_norm_miu_by_def(
            #     net.named_parameters(), loss, global_step
            # )

            # add metric for weight norm
            # norm.log_global_weight_norm_lf_by_def(net.parameters(), loss, global_step)
            # norm.log_local_weight_norm_lf_by_def(
            #     net.named_parameters(), loss, global_step
            # )

        # add epoch metric
        avg_acc = avg_acc / batch_num
        avg_loss = avg_loss / batch_num
        test_avg_acc, test_avg_loss = test()
        writer.add_scalars(
            main_tag="training&test",
            tag_scalar_dict={
                "train_acc": avg_acc,
                "train_loss": avg_loss,
                "test_acc": test_avg_acc,
                "test_loss": test_avg_loss,
            },
            global_step=i,
        )
        logger.info(
            "[Epoch {}][[Cost: {}s]] train_acc: {}, train_loss: {}, test_acc: {}, test_loss: {}".format(
                i,
                time.perf_counter() - epoch_start,
                avg_acc,
                avg_loss,
                test_avg_acc,
                test_avg_loss,
            )
        )


if __name__ == "__main__":
    train()
