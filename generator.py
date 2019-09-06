import torch.nn as nn
from torch.nn.utils import spectral_norm

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

                                      spectral_norm(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(),  # 32x32

                                      spectral_norm(nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)),
                                      nn.Tanh() # 64x64
                                      )

    def forward(self, noise):
        noise = noise.view(-1, 100, 1, 1)
        fake_image = self.generate(noise)
        return fake_image