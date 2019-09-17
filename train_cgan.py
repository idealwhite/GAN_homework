from dataio import *
from models import ConditionalDiscriminator, ConditionalGenerator
from torch.utils.data import DataLoader

import torch
from torch.optim import RMSprop

def update_discriminator(batch_image, batch_condition, batch_error_image, batch_err_condition, generator, discriminator, optimizer, clip, dim_noise, device):
    discriminator.train()
    batch_size = len(batch_image)

    batch_fake = get_fake_batch_conditional(generator, batch_condition, batch_size, dim_noise, device)

    discriminator.zero_grad()
    # True image, true condition
    loss_img = discriminator(batch_image, batch_condition, fake_image=False)
    loss_img.backward()

    # Error image, true condition
    loss_error_img = discriminator(batch_error_image, batch_condition, fake_image=True) * 0.3
    loss_error_img.backward()

    # True image, Error condition
    loss_error_img = discriminator(batch_image, batch_err_condition, fake_image=True) * 0.3
    loss_error_img.backward()

    # Error image, error condition
    loss_fake = discriminator(batch_fake, batch_condition, fake_image=True) * 0.4
    loss_fake.backward()
    grad = discriminator.conv[0].weight_orig.grad.mean().item()

    optimizer.step()

    if clip:
        for p in discriminator.parameters():
            p.data.clamp_(-0.01, 0.01)

    return loss_img.item() + loss_fake.item(), grad


def update_generator(generator, discriminator, batch_condition, optimizer, batch_size, dim_noise, device):
    discriminator.eval()
    generator.train()

    discriminator.zero_grad()
    generator.zero_grad()

    batch_noise = get_noise_batch(batch_size, dim_noise, device)

    batch_fake_image = generator(batch_noise, batch_condition)
    loss_generate = discriminator(batch_fake_image, batch_condition, fake_image=False)

    loss_generate.backward()
    grad = generator.generate[0].weight.grad.mean().item()

    discriminator.zero_grad()

    optimizer.step()

    return loss_generate.item(), grad

def eval_G(generator, dim_noise, device, grid=False):
    generator.eval()
    with torch.no_grad():
        noise = get_noise_batch(25, dim_noise, device)
        tags = ['blue hair blue eyes'] * 5 \
               + ['blue hair green eyes'] * 5 \
               + ['blue hair red eyes'] * 5 \
               + ['green hair blue eyes'] * 5 \
                + ['green hair red eyes'] * 5
        batch_condition = tags2vec_batch(tags, device)
        output_images = generator(noise, batch_condition)

    if grid == True:
        output_images = make_grid(output_images*0.5+0.5, nrow=5)
    return output_images

if __name__ == '__main__':
    from tqdm import trange
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
    parser.add_argument("--loss", default='BCE', type=str, required=False,
                        help="loss function. Now support BCE/WGAN/hinge")
    parser.add_argument("--model_name", default='gan', type=str, required=False,
                        help="name of this model")
    parser.add_argument("--tb", default=False, type=bool, required=False,
                        help="use tensorboard")
    parser.add_argument("--clip", default=False, type=bool, required=False,
                        help="apply weight clip in Discriminator")
    parser.add_argument("--lr", default=1e-4, type=float, required=False)
    args = parser.parse_args()

    max_epoch = args.max_epoch
    n_update_d = args.n_update_d
    n_update_g = args.n_update_g
    batch_size = args.batch_size
    dim_noise = args.dim_noise
    n_eval_epoch = args.n_eval_epoch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = ConditionalDiscriminator(args.loss)
    G = ConditionalGenerator()

    D.to(device)
    G.to(device)

    # train
    optimizer_D = RMSprop(D.parameters(), lr=args.lr)
    optimizer_G = RMSprop(G.parameters(), lr=args.lr)

    face_dataset = get_condition_image_dataset(device)
    dataloader = DataLoader(face_dataset, batch_size=batch_size, shuffle=True)

    if args.tb:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(logdir='./log/'+args.model_name)

    for epoch in trange(max_epoch):
        loss_epoch_d, loss_epoch_g = 0,0
        for i, batch in enumerate(dataloader):
            batch_image, batch_condition, batch_error_image, batch_err_condition = batch

            if i % n_update_d == 0:
                loss_d, grad_d = update_discriminator(batch_image, batch_condition, batch_error_image, batch_err_condition,\
                                                      G, D, optimizer_D, args.clip, dim_noise, device)
                loss_epoch_d += loss_d / n_update_d

            if i % n_update_g == 0:
                loss_g, grad_g = update_generator(G, D, batch_condition, optimizer_G, batch_size, dim_noise, device)
                loss_epoch_g += loss_g / n_update_g

        # evaluation
        if epoch % n_eval_epoch and args.tb:
            # print('Epoch: %d => Loss D: %.5f, G: %.5f' % (epoch, loss_epoch_d, loss_epoch_g))
            writer.add_scalar('loss_D', loss_d, global_step=epoch)
            writer.add_scalar('loss_G', loss_g, global_step=epoch)
            writer.add_scalar('grad_D_low', grad_d, global_step=epoch)
            writer.add_scalar('grad_G_low', grad_g, global_step=epoch)

            generate_img = eval_G(G, dim_noise=100, device=device, grid=True)
            writer.add_image('fake_image', generate_img, global_step=epoch)

    # test
    noise = get_noise_batch(25, 100, device)
    output_images = G(noise)
    save_image(output_images, nrow=5, filename='./output/imgs.png')
    torch.save(G, './output/'+args.model_name+'.MODEL')
