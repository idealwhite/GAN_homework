from dataio import *
from discriminator import Discriminator
from generator import Generator, generator_loss

import torch
from torch.optim import Adam, RMSprop


def update_discriminator(train_data, generator, discriminator, optimizer, batch_size, dim_noise, device):
    discriminator.train()

    batch_image = get_image_batch(train_data, batch_size, device)
    batch_fake = get_fake_batch(generator, batch_size, dim_noise, device)

    discriminator.zero_grad()
    loss = discriminator(batch_image, batch_fake)

    loss.backward()
    torch.nn.utils.clip_grad_value_(discriminator.parameters(), 0.01)

    optimizer.step()


def update_generator(generator, discriminator, optimizer, batch_size, dim_noise, device):
    discriminator.eval()
    generator.train()

    discriminator.zero_grad()
    generator.zero_grad()

    batch_noise = get_noise_batch(batch_size, dim_noise, device)

    batch_fake_image = generator(batch_noise)
    predicted_fake_label = discriminator(batch_fake_image)

    loss = generator_loss(predicted_fake_label)
    loss.backward()
    optimizer.step()

if __name__ == '__main__':
    max_epoch = 10
    n_update_d = 10
    n_update_g = 10
    batch_size = 3
    dim_noise = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = Discriminator()
    G = Generator()

    D.to(device)
    G.to(device)

    optimizer_D = Adam(D.parameters(), lr=1e-4)
    optimizer_G = RMSprop(G.parameters(), lr=1e-4)
    train_data = face_folder

    for epoch in range(max_epoch):
        for n in range(n_update_d):
            update_discriminator(train_data, G, D, optimizer_D, batch_size, dim_noise, device)

        for n in range(n_update_g):
            update_generator(G, D, optimizer_G, batch_size, dim_noise, device)
