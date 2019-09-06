import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

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

                                  spectral_norm(nn.Conv2d(256, 1, kernel_size=4, stride=1)),
                                  nn.Sigmoid()
                                  )


    def forward_froze(self, image):
        features = self.conv(image)
        logit = features.view(len(image), -1)
        return logit

    def forward_update(self, batch_image, batch_fake):
        logit_image = self.forward_froze(batch_image)
        logit_fake = self.forward_froze(batch_fake)

        return  torch.mean(logit_image - logit_fake)

    def forward(self, image, fake_image=False):
        logits = self.forward_froze(image)

        if fake_image is False:
            return nn.BCELoss()(logits, torch.ones(len(image), 1).to(logits.device)) # torch.mean(logits)
        else:
            return nn.BCELoss()(logits, torch.zeros(len(image), 1).to(logits.device)) # -torch.mean(logits)