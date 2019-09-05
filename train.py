from dataio import *
from discriminator import Discriminator
from generator import Generator, generator_loss
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from torch.optim import Adam, RMSprop

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def update_discriminator(batch_image, generator, discriminator, optimizer, batch_size, dim_noise, device):
    discriminator.train()

    batch_fake = get_fake_batch(generator, batch_size, dim_noise, device)

    discriminator.zero_grad()
    loss_img = discriminator(batch_image, fake_image=False)
    loss_img.backward()
    loss_fake = discriminator(batch_fake, fake_image=True)
    loss_fake.backward()

    optimizer.step()

    for p in discriminator.parameters():
        p.data.clamp_(-0.01, 0.01)

    return loss_img.item() + loss_fake.item()


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
        output_images = make_grid(output_images*0.5+0.5, nrow=4)
    return output_images

if __name__ == '__main__':
    from tqdm import tqdm, trange
    import argparse

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--batch_size", default=2, type=int, required=False,
                        help="The input data dir.")
    parser.add_argument("--max_epoch", default=10, type=int, required=False,
                        help="Training epoch of D-G recurrence.")
    parser.add_argument("--n_update_d", default=1, type=int, required=False,
                        help="num of batch when update D in an epoch.")
    parser.add_argument("--n_update_g", default=1, type=int, required=False,
                        help="num of batch when update G in an epoch.")
    parser.add_argument("--dim_noise", default=100, type=int, required=False)
    parser.add_argument("--n_eval_epoch", default=100, type=int, required=False,
                        help="epochs per eval")
    parser.add_argument("--model_name", default='gan', type=str, required=False,
                        help="name of this model")
    args = parser.parse_args()

    max_epoch = args.max_epoch
    n_update_d = args.n_update_d
    n_update_g = args.n_update_g
    batch_size = args.batch_size
    dim_noise = args.dim_noise
    n_eval_epoch = args.n_eval_epoch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = Discriminator()
    G = Generator()

    D.apply(weights_init)
    G.apply(weights_init)

    D.to(device)
    G.to(device)

    # train
    optimizer_D = RMSprop(D.parameters(), lr=1e-4)
    optimizer_G = RMSprop(G.parameters(), lr=1e-4)

    face_dataset = TensorDataset(torch.stack([f[0] for f in face_folder], dim=0).to(device))
    dataloader = DataLoader(face_dataset, batch_size=batch_size, shuffle=True)

    from tensorboardX import SummaryWriter
    writer = SummaryWriter(logdir='./log/'+args.model_name)

    for epoch in trange(max_epoch):
        loss_epoch_d, loss_epoch_g = 0,0
        for i, batch_image in enumerate(dataloader):
            if i % n_update_d == 0:
                loss_d = update_discriminator(batch_image[0], G, D, optimizer_D, batch_size, dim_noise, device)
                loss_epoch_d += loss_d / n_update_d

            if i % n_update_g == 0:
                loss_g = update_generator(G, D, optimizer_G, batch_size, dim_noise, device)
                loss_epoch_g += loss_g / n_update_g

        # evaluation
        if epoch % n_eval_epoch == 0:
            # print('Epoch: %d => Loss D: %.5f, G: %.5f' % (epoch, loss_epoch_d, loss_epoch_g))
            writer.add_scalar('loss_D', loss_d, global_step=epoch)
            writer.add_scalar('loss_G', loss_g, global_step=epoch)
            generate_img = eval_G(G, batch_size=16, dim_noise=100, device=device, grid=True)
            writer.add_image('fake_image', generate_img, global_step=epoch)

    # test
    noise = get_noise_batch(25, 100, device)
    output_images = G(noise)
    save_image(output_images, nrow=5, filename='./output/imgs.png')
    torch.save(G, './output/'+args.model_name+'.MODEL')
