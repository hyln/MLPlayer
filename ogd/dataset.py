import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt

batch_size = 64
num_classes = 2


class CustomDataset(Dataset):
    def __init__(self, data, label, name="unknown_task"):
        self.name = name
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return (torch.tensor(self.data[index]), torch.tensor(self.label[index]))

    def __len__(self):
        return len(self.label)

    def draw(self):
        label0data = self.data[self.label == 0]
        label1data = self.data[self.label == 1]
        plt.scatter(label0data[:, 0], label0data[:, 1], label=f"{self.name}_D1")
        plt.scatter(label1data[:, 0], label1data[:, 1], label=f"{self.name}_D2")
        plt.legend()


task1_data = np.load("data/task1_data.npy")
task1_label = np.load("data/task1_label.npy")
task2_data = np.load("data/task2_data.npy")
task2_label = np.load("data/task2_label.npy")

task1_dataset = CustomDataset(task1_data, task1_label, "task1")
task2_dataset = CustomDataset(task2_data, task2_label, "task2")

task1_dataloader = DataLoader(
    dataset=task1_dataset, batch_size=batch_size, shuffle=True
)
task2_dataloader = DataLoader(
    dataset=task2_dataset, batch_size=batch_size, shuffle=True
)


if __name__ == "__main__":
    # print(task1_data.shape)
    # print(task1_label.shape)
    # print(task2_data.shape)
    # print(task2_label.shape)
    #
    # for x, y in task1_dataloader:
    #     print(x, y)
    #     print(x.shape)
    #     print(y.shape)
    #     break
    xlim = [-8, 5]
    ylim = [-2, 6]
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    task1_dataset.draw()
    task2_dataset.draw()
    plt.show()
    # for x, y in task2_dataloader:
    #     print(x, y)
    #     print(x.shape)
    #     print(y.shape)
