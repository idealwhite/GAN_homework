import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(3, 32, kernel_size=4, stride=2),
                                  nn.LeakyReLU(),

                                  nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                                  nn.LeakyReLU(),

                                  nn.Conv2d(64, 128, kernel_size=4, stride=2),
                                  nn.LeakyReLU(),

                                  nn.Conv2d(128, 256, kernel_size=4, stride=2)
                                  )
        self.dense = nn.Linear(1024, 1)

    def forward_froze(self, image):
        features = self.conv(image)
        features = features.view(len(image), -1)
        logit = self.dense(features)
        return logit

    def forward_update(self, batch_image, batch_fake):
        logit_image = self.forward_froze(batch_image)
        logit_fake = self.forward_froze(batch_fake)

        return  torch.mean(logit_image - logit_fake)

    def forward(self, image, fake_image=False):
        loss = self.forward_froze(image)

        if fake_image is False:
            return torch.mean(loss)
        else:
            return -torch.mean(loss)