import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out


class Discriminator(nn.Module):
    def __init__(self, loss):
        super(Discriminator, self).__init__()
        self.loss = loss
        self.conv = nn.Sequential(spectral_norm(nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)),
                                  nn.LeakyReLU(),

                                  spectral_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)),
                                  # nn.InstanceNorm2d(64),
                                  nn.LeakyReLU(),

                                  spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
                                  # nn.InstanceNorm2d(128),
                                  nn.LeakyReLU(),

                                  Self_Attn(128),
                                  spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
                                  # nn.InstanceNorm2d(256),
                                  nn.LeakyReLU(),

                                  Self_Attn(256),
                                  spectral_norm(nn.Conv2d(256, 1, kernel_size=4, stride=1)),
                                  )
        self.apply(weights_init)

    def forward_logit(self, image):
        features = self.conv(image)
        logit = features.view(len(image), -1)
        return logit.squeeze()

    def forward(self, image, fake_image=False):
        logits = self.forward_logit(image)

        if fake_image is False:
            if self.loss.lower() == 'bce':
                loss = nn.BCELoss()(nn.Sigmoid()(logits), torch.ones(len(image)).to(logits.device))
            elif self.loss.lower() == 'wgan':
                loss = torch.mean(logits)
            elif self.loss.lower() == 'hinge':
                loss = torch.mean(nn.ReLU()(1.0 - logits))
            elif self.loss.lower() == 'soft_hinge':
                loss = torch.mean(nn.Softplus()(1.0 - logits))
        else:
            if self.loss.lower() == 'bce':
                loss = nn.BCELoss()(nn.Sigmoid()(logits), torch.zeros(len(image)).to(logits.device))
            elif self.loss.lower() == 'wgan':
                loss = -torch.mean(logits)
            elif self.loss.lower() == 'hinge':
                loss = torch.mean(nn.ReLU()(1.0 + logits))
            elif self.loss.lower() == 'soft_hinge':
                loss = torch.mean(nn.Softplus()(1.0 + logits))
        return loss

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.generate = nn.Sequential(spectral_norm(nn.ConvTranspose2d(100, 512, kernel_size=4, stride=1, padding=0)),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(), # 4x4

                                      spectral_norm(nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(), # 8x8

                                      spectral_norm(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(), # 16x16

                                      Self_Attn(128),
                                      spectral_norm(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(),  # 32x32

                                      Self_Attn(64),
                                      spectral_norm(nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)),

                                      spectral_norm(nn.ConvTranspose2d(3, 3, kernel_size=4, stride=1, padding=1)),
                                      nn.Tanh() # 64x64
                                      )
        self.apply(weights_init)

    def forward(self, noise):
        noise = noise.view(-1, 100, 1, 1)
        fake_image = self.generate(noise)
        return fake_image

# ------------- Condition GAN ----------------
class ConditionalDiscriminator(nn.Module):
    def __init__(self, loss):
        super(ConditionalDiscriminator, self).__init__()
        self.loss = loss
        self.condition_encode = nn.Sequential(nn.Linear(22, 64),
                                              nn.ReLU(),
                                              )
        self.conv = nn.Sequential(spectral_norm(nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)),
                                  nn.LeakyReLU(),

                                  spectral_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)),
                                  # nn.InstanceNorm2d(64),
                                  nn.LeakyReLU(),

                                  Self_Attn(64),
                                  spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
                                  # nn.InstanceNorm2d(128),
                                  nn.LeakyReLU(),

                                  spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
                                  # nn.InstanceNorm2d(256),
                                  nn.LeakyReLU(),
                                  )
        self.discriminator = nn.Sequential(spectral_norm(nn.Conv2d(256+64, 1, kernel_size=4, stride=1)),
                                           )
        self.apply(weights_init)

    def forward_logit(self, condition, image):
        batch_size = len(condition)
        features = self.conv(image)

        condition = self.condition_encode(condition)
        condition = condition.repeat(1, 4*4).view(batch_size, -1, 4, 4)
        condition_features = torch.cat([features, condition], dim=1)


        logit = self.discriminator(condition_features)
        return logit.squeeze()

    def forward(self, image, condition, fake_image=False):

        logits = self.forward_logit(condition, image)

        if fake_image is False:
            if self.loss.lower() == 'bce':
                loss = nn.BCELoss()(nn.Sigmoid()(logits), torch.ones(len(image)).to(logits.device))
            elif self.loss.lower() == 'wgan':
                loss = torch.mean(logits)
            elif self.loss.lower() == 'hinge':
                loss = torch.mean(nn.ReLU()(1.0 - logits))
            elif self.loss.lower() == 'soft_hinge':
                loss = torch.mean(nn.Softplus()(1.0 - logits))
        else:
            if self.loss.lower() == 'bce':
                loss = nn.BCELoss()(nn.Sigmoid()(logits), torch.zeros(len(image)).to(logits.device))
            elif self.loss.lower() == 'wgan':
                loss = -torch.mean(logits)
            elif self.loss.lower() == 'hinge':
                loss = torch.mean(nn.ReLU()(1.0 + logits))
            elif self.loss.lower() == 'soft_hinge':
                loss = torch.mean(nn.Softplus()(1.0 + logits))
        return loss

class ConditionalMultiDiscriminator(nn.Module):
    def __init__(self, loss):
        super(ConditionalMultiDiscriminator, self).__init__()
        self.loss = loss
        self.condition_encode = nn.Sequential(nn.Linear(22, 64),
                                              nn.ReLU(),
                                              )
        self.conv = nn.Sequential(spectral_norm(nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)),
                                  nn.LeakyReLU(),

                                  spectral_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)),
                                  # nn.InstanceNorm2d(64),
                                  nn.LeakyReLU(),

                                  spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
                                  # nn.InstanceNorm2d(128),
                                  nn.LeakyReLU(),

                                  spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
                                  # nn.InstanceNorm2d(256),
                                  nn.LeakyReLU(),
                                  )
        self.discriminator_1 = spectral_norm(nn.Conv2d(256+64, 1, kernel_size=4, stride=1))
        self.discriminator_2 = spectral_norm(nn.Conv2d(256 + 64, 1, kernel_size=4, stride=1))
        self.discriminator_3 = spectral_norm(nn.Conv2d(256 + 64, 1, kernel_size=4, stride=1))
        self.apply(weights_init)

    def forward_logit(self, condition, image, n_discriminator):
        batch_size = len(condition)
        features = self.conv(image)

        condition = self.condition_encode(condition)
        condition = condition.repeat(1, 4*4).view(batch_size, -1, 4, 4)
        condition_features = torch.cat([features, condition], dim=1)

        if n_discriminator == 1:
            logit = self.discriminator_1(condition_features)
        elif n_discriminator == 2:
            logit = self.discriminator_2(condition_features)
        elif n_discriminator == 3:
            logit = self.discriminator_3(condition_features)
        return logit.squeeze()

    def forward(self, image, condition, fake_image=False, n_discriminator=1):

        logits = self.forward_logit(condition, image, n_discriminator)

        if fake_image is False:
            if self.loss.lower() == 'bce':
                loss = nn.BCELoss()(nn.Sigmoid()(logits), torch.ones(len(image)).to(logits.device))
            elif self.loss.lower() == 'wgan':
                loss = torch.mean(logits)
            elif self.loss.lower() == 'hinge':
                loss = torch.mean(nn.ReLU()(1.0 - logits))
            elif self.loss.lower() == 'soft_hinge':
                loss = torch.mean(nn.Softplus()(1.0 - logits))
        else:
            if self.loss.lower() == 'bce':
                loss = nn.BCELoss()(nn.Sigmoid()(logits), torch.zeros(len(image)).to(logits.device))
            elif self.loss.lower() == 'wgan':
                loss = -torch.mean(logits)
            elif self.loss.lower() == 'hinge':
                loss = torch.mean(nn.ReLU()(1.0 + logits))
            elif self.loss.lower() == 'soft_hinge':
                loss = torch.mean(nn.Softplus()(1.0 + logits))
        return loss


class ConditionalGenerator(nn.Module):
    def __init__(self):
        super(ConditionalGenerator, self).__init__()
        self.condition_transfer = nn.Sequential(
            nn.Linear(22, 64),
            nn.ReLU(),
        )

        self.generate = nn.Sequential(
            nn.ConvTranspose2d(100+64, 512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),  # 4x4

            spectral_norm(nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),  # 8x8

            spectral_norm(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),  # 16x16

            Self_Attn(128),
            spectral_norm(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # 32x32

            spectral_norm(nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)),
            nn.Tanh()  # 64x64
            )

        self.apply(weights_init)

    def forward(self, noise, condition):
        batch_size = len(noise)
        condition = self.condition_transfer(condition)

        condition_noise = torch.cat([condition, noise], dim=-1)
        condition_noise = condition_noise.view(batch_size, -1, 1, 1)

        fake_image = self.generate(condition_noise)
        return fake_image
