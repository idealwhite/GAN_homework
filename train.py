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
    return loss.item()


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

    return loss.item()

def eval_G(generator, batch_size, dim_noise, device, grid=False):
    generator.eval()
    with torch.no_grad():
        noise = get_noise_batch(batch_size, dim_noise, device)
        output_images = generator(noise)

    if grid == True:
        output_images = make_grid(output_images)
    return output_images

if __name__ == '__main__':
    from tensorboardX import SummaryWriter
    import argparse

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--batch_size", default=5, type=int, required=False,
                        help="The input data dir.")
    parser.add_argument("--max_epoch", default=100, type=int, required=False,
                        help="Training epoch of D-G recurrence.")
    parser.add_argument("--n_update_d", default=500, type=int, required=False,
                        help="num of batch when update D in an epoch.")
    parser.add_argument("--n_update_g", default=100, type=int, required=False,
                        help="num of batch when update G in an epoch.")
    parser.add_argument("--dim_noise", default=100, type=int, required=False)
    args = parser.parse_args()

    max_epoch = args.max_epoch
    n_update_d = args.n_update_d
    n_update_g = args.n_update_g
    batch_size = args.batch_size
    dim_noise = args.dim_noise

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = Discriminator()
    G = Generator()

    D.to(device)
    G.to(device)

    # train
    optimizer_D = RMSprop(D.parameters(), lr=1e-4)
    optimizer_G = RMSprop(G.parameters(), lr=1e-4)
    train_data = face_folder
    writer = SummaryWriter(logdir='./log/')

    for epoch in range(max_epoch):
        loss_epoch_d, loss_epoch_g = 0,0
        for n in range(n_update_d):
            loss_d = update_discriminator(train_data, G, D, optimizer_D, batch_size, dim_noise, device)
            loss_epoch_d += loss_d / n_update_d
        for n in range(n_update_g):
            loss_g = update_generator(G, D, optimizer_G, batch_size, dim_noise, device)
            loss_epoch_g += loss_g / n_update_g

        print('Loss- D: %.5f, G: %.5f' % (loss_epoch_d, loss_epoch_g))
        writer.add_scalar('loss_D', loss_d, global_step=epoch)
        writer.add_scalar('loss_G', loss_g, global_step=epoch)

        # evaluation
        generate_img = eval_G(G, batch_size=9, dim_noise=100, device=device, grid=True)
        writer.add_image('fake_image', generate_img, global_step=epoch)

    # test
    noise = get_noise_batch(9, 100, device)
    output_images = G(noise)
    save_image(output_images, './output/imgs.png')