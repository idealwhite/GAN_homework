from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid

from torch.utils.data import TensorDataset
import numpy as np
import torch

face_folder = ImageFolder('./AnimeDataset', transform=transforms.Compose([transforms.Resize([64,64]),
                                                                          transforms.ToTensor(),
                                                                          transforms.Normalize([0]*3, [1]*3)]))

def get_noise_batch(batch_size, dim_noise, device):
    noise_batch = torch.normal(0, 1, [batch_size, dim_noise]).to(device)
    return noise_batch

def get_fake_batch(generator, batch_size, dim_noise, device):
    noise_batch = get_noise_batch(batch_size, dim_noise, device)
    batch_fake = generator(noise_batch).to(device)
    return batch_fake


if __name__ == "__main__":
    print(face_folder[0])
    print(get_noise_batch(3, 100, "cpu").shape)
