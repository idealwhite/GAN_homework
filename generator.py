import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.transform = nn.Sequential(nn.Linear(100, 128*16*16),
                                nn.ReLU())
        self.generate = nn.Sequential(nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(),
                                      nn.ConvTranspose2d(64, 3, kernel_size=4, stride=1, padding=1),
                                      nn.Tanh()
                                      )

    def forward(self, noise):
        noise = self.transform(noise)
        noise = noise.view(-1, 128, 16, 16)
        fake_image = self.generate(noise)
        return fake_image

def generator_loss(predicted_fake_label):
    return -torch.mean(predicted_fake_label)