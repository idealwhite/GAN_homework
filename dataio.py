from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

import numpy as np
import torch

face_folder = ImageFolder('./AnimeDataset', transform=transforms.Compose([transforms.ToTensor()]))

def get_fake_batch(generator, batch_size, dim_noise):
    noise_batch = get_noise_batch(batch_size, dim_noise)
    batch_fake = generator(noise_batch)
    return batch_fake

def get_image_batch(dataset, batch_size):
    sampler = np.random.randint(0, len(dataset), batch_size)
    return dataset[sampler]

def get_noise_batch(batch_size, dim_noise):
    noise_batch = torch.normal(0, 1, [batch_size, dim_noise])
    return noise_batch

if __name__ == "__main__":
    print(face_folder.classes[0])
    print(face_folder[0])