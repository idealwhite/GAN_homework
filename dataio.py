from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid

import numpy as np
import torch

face_folder = ImageFolder('./AnimeDataset', transform=transforms.Compose([transforms.Resize([65,65]),
                                                                          transforms.ToTensor(),
                                                                          transforms.Normalize(0, 1)]))

def get_noise_batch(batch_size, dim_noise, device):
    noise_batch = torch.normal(0, 1, [batch_size, dim_noise]).to(device)
    return noise_batch

def get_fake_batch(generator, batch_size, dim_noise, device):
    noise_batch = get_noise_batch(batch_size, dim_noise, device)
    batch_fake = generator(noise_batch).to(device)
    return batch_fake

def get_image_batch(dataset, batch_size, device):
    sampler = np.random.randint(0, len(dataset), batch_size)
    image_batch = torch.stack([dataset[s][0] for s in sampler], dim=0).to(device)
    return image_batch


if __name__ == "__main__":

    print(get_noise_batch(3, 100, "cpu").shape)

    print(get_image_batch(face_folder, 3, "cpu").shape)