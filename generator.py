import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.transform = nn.Sequential(nn.Linear(100, 128*16*16),
                                nn.ReLU())
        self.generate = nn.Sequential(nn.ConvTranspose2d(128, 128, kernel_size=4),
                                      nn.ReLU(),

                                      )

def generator_loss(predicted_fake_label):
    pass