import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import CustomDataset
from log import logger, writer
from util import get_module_grad, set_module_grad, angle


class LinearModel(nn.Module):
    def __init__(self, in_features=4, out_features=1):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(x)

    def get_grad_vector(self):
        return get_module_grad(self)

    def set_grad_vector(self, grad):
        set_module_grad(self, grad)


linear_model = LinearModel()

task_data = [[1.0, 3.0, -5.0, 2.0], [5.0, 2.0, 2.0, -1.0]]
task_label = [[1.0], [-1.0]]
task1_data = [[1.0, 3.0, -5.0, 2.0]]
task1_label = [[1.0]]
task2_data = [[5.0, 2.0, 2.0, -1.0]]
task2_label = [[-1.0]]

task_dataset = CustomDataset(task_data, task_label)
task1_dataset = CustomDataset(task1_data, task1_label)
task2_dataset = CustomDataset(task2_data, task2_label)

task_dataloader = DataLoader(dataset=task_dataset, batch_size=1, shuffle=False)
task1_dataloader = DataLoader(dataset=task1_dataset, batch_size=1, shuffle=False)
task2_dataloader = DataLoader(dataset=task2_dataset, batch_size=1, shuffle=False)

epoch_num = 100


def train(
    name="task1",
    dataloader=task_dataloader,
    model=linear_model,
    loss_func=torch.nn.MSELoss(),
    optimizer_func=torch.optim.SGD,
    learning_rate=1e-3,
    epoch=epoch_num,
    ogd_util=None,
):
    global_step = 0.0
    optimizer = optimizer_func(model.parameters(), lr=learning_rate)
    for i in range(epoch):
        for batch, (inputs, labels) in enumerate(dataloader):
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = loss_func(outputs, labels)
            loss.backward()
            train_grad = model.get_grad_vector()
            if ogd_util is not None:
                grad = model.get_grad_vector()
                grad = ogd_util.orthogonalize(grad)
                model.set_grad_vector(grad)
            optimizer.step()
            global_step += 1
            logger.debug(
                "[Epoch {}][Batch {}]train_loss: {}".format(i, batch, loss.item())
            )

            tag = ""
            if ogd_util is not None:
                tag = "ogd_"
            tag_scalar_dict = {
                name + "_" + tag + "train_loss": loss.item(),
            }
            grad_tag_scalar_dict = {}
            if name == "task1":
                model.eval()
                for batch, (inputs, labels) in enumerate(task2_dataloader):
                    outputs = model(inputs)
                    optimizer.zero_grad()
                    test_loss = loss_func(outputs, labels)
                    test_loss.backward()
                    test_grad = model.get_grad_vector()
                    tag_scalar_dict[
                        name + "_" + tag + "task2_test_loss"
                    ] = test_loss.item()
                    grad_tag_scalar_dict["gradient vector angle on task1"] = angle(
                        train_grad, test_grad
                    )
            elif name == "task2":
                model.eval()
                for batch, (inputs, labels) in enumerate(task1_dataloader):
                    outputs = model(inputs)
                    optimizer.zero_grad()
                    test_loss = loss_func(outputs, labels)
                    test_loss.backward()
                    test_grad = model.get_grad_vector()
                    tag_scalar_dict[
                        name + "_" + tag + "task1_test_loss"
                    ] = test_loss.item()
                    grad_tag_scalar_dict["gradient vector angle on task2"] = angle(
                        train_grad, test_grad
                    )
            writer.add_scalars(
                main_tag="linear training&test",
                tag_scalar_dict=tag_scalar_dict,
                global_step=global_step,
            )
            writer.add_scalars(
                main_tag="gradient vector angle",
                tag_scalar_dict=grad_tag_scalar_dict,
                global_step=global_step,
            )

    if ogd_util is not None:
        ogd_util.save_for_regression(dataloader=dataloader, model=model)
        print("ogd_util", ogd_util.basis_vectors)


if __name__ == "__main__":
    from torchinfo import summary
    from util import angle

    # summary(linear_model, (1, 4))
    # train()

    train(name="task1", dataloader=task1_dataloader, model=linear_model)
    train(name="task2", dataloader=task2_dataloader, model=linear_model)

    # from ogd import OGD_Util
    # linear_model = LinearModel()
    # ogd_util = OGD_Util()
    # task1_grad = train(
    #     name="task1", dataloader=task1_dataloader, model=linear_model, ogd_util=ogd_util
    # )
    # task2_grad = train(
    #     name="task2", dataloader=task2_dataloader, model=linear_model, ogd_util=ogd_util
    # )
    # print(task1_grad, task2_grad)
    # print(angle(task1_grad, task2_grad))
