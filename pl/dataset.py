import torch
from torchvision import datasets, transforms

batch_size = 128
num_worker = 16
ntk_batch = 32

train_dataset = datasets.MNIST(
    root="./data", train=True, transform=transforms.ToTensor(), download=True
)
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker
)
test_dataset = datasets.MNIST(
    root="./data", train=False, transform=transforms.ToTensor(), download=True
)
test_dataloader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker
)
ntk_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=ntk_batch, shuffle=True
)
