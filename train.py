from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data import generate_loader
from option import get_option
from model import Generator
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image

class Solver():
    def __init__(self, opt):
        self.opt = opt
        self.img_size = opt.input_size
        self.n_classes = opt.n_classes
        self.dev = torch.device("cuda:{}".format(opt.gpu) if torch.cuda.is_available() else "cpu")
        print("device: ", self.dev)

        self.generator = Generator(channels=1, img_size=self.img_size, n_classes=self.n_classes).to(self.dev)
        self.discriminator = Discriminator(channels=1, img_size=self.img_size, n_classes=self.n_classes).to(self.dev)

        if opt.multigpu:
            self.generator = nn.DataParallel(self.generator, device_ids=self.opt.device_ids).to(self.dev)
            self.discriminator = nn.DataParallel(self.discriminator, device_ids=self.opt.device_ids).to(self.dev)

        print("# Generator params:", sum(map(lambda x: x.numel(), self.generator.parameters())))
        print("# Discriminator params:", sum(map(lambda x: x.numel(), self.discriminator.parameters())))

        self.loss_fn = nn.MSELoss()

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        
        self.train_loader = generate_loader(opt)
        print("train set ready")

    def fit(self):
        opt = self.opt
        print("start training")

        for epoch in range(opt.n_epoch):
            loop = tqdm(self.train_loader)

            for i , (imgs, labels) in enumerate(loop):
                # Adversarial ground truths (real=1, fake=0)
                real = Variable(torch.ones(imgs.size(0), 1)).to(self.dev)
                fake = Variable(torch.zeros(imgs.size(0), 1)).to(self.dev)

                real_imgs = Variable(imgs).to(self.dev)
                labels = Variable(labels).to(self.dev)

                # train Generator
                self.optimizer_G.zero_grad()
                z = Variable(torch.randn((imgs.size(0), 100))).to(self.dev)
                gen_labels = torch.tensor(np.random.randint(0, self.n_classes, imgs.size(0)))
                gen_labels = Variable(gen_labels).to(self.dev)
                generated_imgs = self.generator(z, gen_labels)
                g_loss = self.loss_fn(self.discriminator(generated_imgs, gen_labels), real)
                g_loss.backward()
                self.optimizer_G.step()

                # train Discriminator
                self.optimizer_D.zero_grad()
                real_loss = self.loss_fn(self.discriminator(real_imgs, labels), real)
                fake_loss = self.loss_fn(self.discriminator(generated_imgs.detach(), gen_labels), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                self.optimizer_D.step()

                if (epoch+1) % 50 == 0:
                    save_image(generated_imgs[:25], f"data{epoch}.png", nrow=5, normalize=True)
                
            print(f"[Epoch {epoch+1}/{opt.n_epoch}] [D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}]")

def main():
    opt = get_option()
    torch.manual_seed(opt.seed)
    solver = Solver(opt)
    solver.fit()

if __name__ == "__main__":
    main()