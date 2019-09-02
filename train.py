from dataio import *
from discriminator import Discriminator
from generator import Generator, generator_loss

import torch
from torch.optim import Adam, RMSprop


max_epoch = 100
n_update_d = 1000
n_update_g = 1000
batch_size = 3
dim_noise = 128

D = Discriminator()
G = Generator()

optimizer_D = Adam(D.parameters(), lr=1e-4)


def update_discriminator(train_data, generator, discriminator, optimizer):
    discriminator.train()

    batch_image = get_image_batch(train_data, batch_size)
    batch_fake = get_fake_batch(generator, batch_size)

    discriminator.zero_grad()
    loss = discriminator(batch_image, batch_fake)

    loss.backward()
    optimizer.step()


def update_generator(train_data, generator, discriminator, optimizer):
    discriminator.eval()
    generator.train()

    discriminator.zero_grad()
    generator.zero_grad()

    batch_noise = get_noise_batch(batch_size, dim_noise=dim_noise)

    batch_fake_image = generator(batch_noise)
    predicted_fake_label = discriminator(batch_fake_image)

    loss = generator_loss(predicted_fake_label)
    loss.backward()
    optimizer.step()

for epoch in range(max_epoch):
    for n in range(n_update_d):
        update_discriminator()

    for n in range(n_update_g):
        update_discriminator()
