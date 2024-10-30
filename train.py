import argparse
import os

import matplotlib
import numpy as np
import math
import itertools
import time
import datetime
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from imageDataset import *
from model import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="facades", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument(
    "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

# os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
# os.makedirs("save/%s" % opt.dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

# Loss functions
criterion_GAN = torch.nn.BCEWithLogitsLoss()
criterion_pixelwise = torch.nn.SmoothL1Loss()
criterion_ab = torch.nn.MSELoss()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 10

patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

discriminator = Discriminator()
input_shape = (opt.channels, opt.img_height, opt.img_width)

## 创建生成器，判别器对象
generator = GeneratorResNet(input_shape, 9)
if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()
    criterion_ab.cuda()

is_pretrain=True
# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Configure dataloaders
transforms_ = [
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

## Training data loader
dataloader = DataLoader(  ## 改成自己存放文件的目录
    ImageDatasetP2P("/dataset/CEDAR/", transforms_=transforms_, train=True),
    ## "./datasets/facades" , unaligned:设置非对其数据
    batch_size=opt.batch_size,  ## batch_size = 1
    shuffle=True,
    num_workers=opt.n_cpu,
)
## Test data loader
val_dataloader = DataLoader(
    ImageDatasetP2P("/dataset/CEDAR/", transforms_=transforms_, train=True),
    batch_size=5,
    shuffle=True,
    num_workers=1,
)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    # real_A = Variable(imgs["B"].type(Tensor))
    real_B = Variable(imgs["B"].type(Tensor))
    real_A = Variable(imgs["A"].type(Tensor))
    fake_B = generator(real_A, real_A)

    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)
    # Arange images along y-axis
    ## 把以上图像都拼接起来，保存为一张大图片
    image_grid = torch.cat((real_A, real_B, fake_B), 1)

    save_image(image_grid, "imagesTestP2P/%s.png" % (batches_done), normalize=False)

matplotlib.use('Agg')
# ----------
#  Training
# ----------

prev_time = time.time()
Loss_D_list = []
Loss_G_list = []
alist = []
blist = []
for epoch in range(opt.epoch, opt.n_epochs):
    a = 0
    b = 0
    D_list = []
    G_list = []
    for i, batch in enumerate(dataloader):

        # Model inputs
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # GAN loss
        fake_B = generator(real_A, real_B)

        pred_fake1 = discriminator(fake_B, real_A)
        pred_fake2 = discriminator(fake_B, real_B)
        loss_GAN = criterion_GAN(pred_fake1, valid)
        loss_ab = criterion_ab(pred_fake1, pred_fake2)

        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_B, real_A)

        # Total loss
        loss_G = loss_GAN + lambda_pixel * loss_pixel + loss_ab

        loss_G.backward()

        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator(real_B, real_A)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake1 = discriminator(fake_B.detach(), real_A)
        loss_fake1 = criterion_GAN(pred_fake1, fake)
        pred_fake2 = discriminator(fake_B.detach(), real_B)
        loss_fake2 = criterion_GAN(pred_fake2, fake)
        
        # Total loss
        loss_fake = loss_fake1 + loss_fake2
        loss_D = loss_real + loss_fake1 + loss_fake2

        loss_D.backward()
        optimizer_D.step()

        sample_images(1)

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f, loss_fake: %f] [G loss: %f, pixel: %f, adv: %f, ab: %f]"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_fake.item(),
                loss_G.item(),
                loss_pixel.item(),
                loss_GAN.item(),
                loss_ab.item(),
                # time_left,
            )
        )

        # 计算判别器和生成器的损失
        Loss_D_list.append(loss_D.item())
        Loss_G_list.append(loss_G.item())
        D_list.append(loss_D.item())
        G_list.append(loss_G.item())
        a = a + loss_G.item()
        b = b + loss_D.item()

        # 每个epoch就保存一组测试集中的图片
    sample_images(epoch)
    print('\nloss_G:{}, loss_D:{}'.format(a / len(dataloader), b / len(dataloader)))

    alist.append(a / len(dataloader))
    blist.append(b / len(dataloader))

    ## 训练结束后，保存模型
    torch.save(generator.state_dict(), "saveTestP2P/generator.pth")
    torch.save(discriminator.state_dict(), "saveTestP2P/discriminator.pth")
    print("save my model finished !!")

    # 绘制每个epoch损失曲线
    plt.figure()
    plt.plot(D_list, label='Loss_D')
    plt.plot(G_list, label='Loss_G')
    plt.title("GAN Training Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("lossTestP2P/Training Loss - %d" % (epoch))

    # 绘制总损失曲线
    plt.figure()
    plt.plot(Loss_D_list, label='Loss_D')
    plt.title("Discriminator Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss/Training Loss_D")

    plt.figure()
    plt.plot(Loss_G_list, label='Loss_G')
    plt.title("Generator Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss/Training Loss_G")

    plt.figure()
    plt.plot(Loss_D_list, label='Loss_D')
    plt.plot(Loss_G_list, label='Loss_G')
    plt.title("Training Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss/Training All Loss(iter)")

    plt.figure()
    plt.plot(blist, label='Loss_D')
    plt.plot(alist, label='Loss_G')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss/Training All Loss(Epoch)")
