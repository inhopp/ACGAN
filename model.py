import torch
import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    def __init__(self, channels=1, img_size=32, n_classes = 10):
        super(Generator, self).__init__()
        self.img_shape = (channels, img_size, img_size)
        self.label_emb = nn.Embedding(n_classes, n_classes)

        def block(input_dim, output_dim, normalize=True):
            layers = [nn.Linear(input_dim, output_dim)]

            if normalize:
                layers.append(nn.BatchNorm1d(output_dim, 0.8))

            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers
        
        self.model = nn.Sequential(
            *block(100 + n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        # concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), z), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape) # (batch_size, img_shape...)
        return img


class Discriminator(nn.Module):
    def __init__(self, channels=1, img_size=32, n_classes=10):
        super(Discriminator, self).__init__()
        self.img_shape = (channels, img_size, img_size)
        self.label_embedding = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.Linear(n_classes + int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)

        return validity