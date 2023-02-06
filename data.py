import torch
import torchvision.transforms as transforms
from torchvision import datasets

def generate_loader(opt):
    img_size = opt.input_size

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    dataset = datasets.MNIST(root="./datasets", train=True, download=True, transform=transform)

    kwargs = {
        "batch_size": opt.batch_size,
        "num_workers": opt.num_workers,
        "shuffle": True,
        "drop_last": True,
    }

    return torch.utils.data.DataLoader(dataset, **kwargs)