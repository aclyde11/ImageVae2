import argparse
import datetime
import logging

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from torch import nn, optim
from torchvision.utils import save_image

from dataloader import MoleLoader
from model import GeneralVae, PictureDecoder, PictureEncoder
from utils import AverageMeter, MS_SSIM

logger = logging.getLogger('cairosvg')
logger.setLevel(logging.CRITICAL)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-w', '--workers', default=16, type=int,
                    metavar='N', help='mini-batch size per process (default: 256)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size per process (default: 256)')
parser.add_argument('-g', '--grad-clip', default=2.0, type=float,
                    metavar='N', help='mini-batch size per process (default: 256)')
parser.add_argument('-d', default='moses/data', help='folder with train and test smiles files')
parser.add_argument('-mp', help='model save path', default='models/')
parser.add_argument('-op', help='output files path', default='output/')
args = parser.parse_args()

starting_epoch = 1
epochs = 500
no_cuda = False
seed = 42
data_para = True
log_interval = 25
LR = 1.0e-3
model_load = None  # {'decoder' : '/homes/aclyde11/imageVAE/im_im_small/model/decoder_epoch_128.pt', 'encoder':'/homes/aclyde11/imageVAE/im_im_small/model/encoder_epoch_128.pt'}
cuda = True
data_size = 1400000
torch.manual_seed(seed)
output_dir = args.op
save_files = args.mp
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': args.workers, 'pin_memory': True} if cuda else {}

smiles_lookup_train = pd.read_csv(f"{args.d}/train.csv")
print(smiles_lookup_train.head())
smiles_lookup_test = pd.read_csv(f"{args.d}/test.csv")
print(smiles_lookup_test.head())


class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

        self.crispyLoss = MS_SSIM()

    def compute_kernel(self, x, y):
        x_size = x.shape[0]
        y_size = y.shape[0]
        dim = x.shape[1]

        tiled_x = x.view(x_size, 1, dim).repeat(1, y_size, 1)
        tiled_y = y.view(1, y_size, dim).repeat(x_size, 1, 1)

        return torch.exp(-torch.mean((tiled_x - tiled_y) ** 2, dim=2) / dim * 1.0)

    def compute_mmd(self, x, y):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)

    def forward(self, x_recon, x, mu, logvar, epoch):
        loss_MSE = self.mse_loss(x_recon, x)
        # loss_mmd = self.compute_mmd(x_samples, z)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss_cripsy = self.crispyLoss(x_recon, x)

        if epoch > 10:
            return loss_MSE + loss_KLD + loss_cripsy
        else:
            return loss_MSE + (0.1 * epoch) * loss_KLD


model = None
encoder = None
decoder = None
encoder = PictureEncoder(rep_size=512)
decoder = PictureDecoder(rep_size=512)
# checkpoint = torch.load( save_files + 'epoch_' + str(42) + '.pt', map_location='cpu')
# encoder.load_state_dict(checkpoint['encoder_state_dict'])
# decoder.load_state_dict(checkpoint['decoder_state_dict'])

model = GeneralVae(encoder, decoder, rep_size=512).cuda()

print("LR: {}".format(LR))
optimizer = optim.Adam(model.parameters(), lr=LR)
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


for param_group in optimizer.param_groups:
    param_group['lr'] = LR

# sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=8.0e-5, last_epoch=-1)

if data_para and torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

loss_picture = customLoss()

if data_para and torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    loss_picture = nn.DataParallel(loss_picture)
loss_picture.cuda()
val_losses = []
train_losses = []


def get_batch_size(epoch):
    return args.batch_size


def clip_gradient(optimizer, grad_clip=1.0):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


train_data = MoleLoader(smiles_lookup_train)
train_loader_food = torch.utils.data.DataLoader(
    train_data,
    batch_size=args.batch_size, shuffle=True, drop_last=True,
    **kwargs)

val_data = MoleLoader(smiles_lookup_test)
val_loader_food = torch.utils.data.DataLoader(
    val_data,
    batch_size=args.batch_size, shuffle=True, drop_last=True,
    **kwargs)


def train(epoch):
    print("Epoch {}: batch_size {}".format(epoch, get_batch_size(epoch)))
    model.train()
    loss_meter = AverageMeter()
    for batch_idx, (_, data, _) in enumerate(train_loader_food):
        data = data.float().cuda()
        optimizer.zero_grad()

        recon_batch, mu, logvar, _ = model(data)

        loss2 = loss_picture(recon_batch, data, mu, logvar, epoch)
        loss2 = torch.sum(loss2)
        loss_meter.update(loss2.item(), int(recon_batch.shape[0]))

        loss2.backward()

        clip_gradient(optimizer, grad_clip=args.grad_clip)
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} {}'.format(
                epoch, batch_idx * len(data), len(train_loader_food.dataset),
                       100. * batch_idx / len(train_loader_food),
                loss_meter.avg, datetime.datetime.now()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, loss_meter.avg))
    return loss_meter.avg





def interpolate_points(x, y, sampling):
    ln = LinearRegression()
    data = np.stack((x, y))
    data_train = np.array([0, 1]).reshape(-1, 1)
    ln.fit(data_train, data)

    return ln.predict(sampling.reshape(-1, 1)).astype(np.float32)


def test(epoch):
    model.eval()
    losses = AverageMeter()
    test_loss = 0
    with torch.no_grad():
        for i, (_, data, _) in enumerate(val_loader_food):
            data = data.float().cuda()

            recon_batch, mu, logvar, _ = model(data)
            print('recon', recon_batch.shape, mu.shape, logvar.shape, data.shape)
            loss2 = loss_picture(recon_batch, data, mu, logvar, epoch)
            loss2 = torch.sum(loss2)
            losses.update(loss2.item(), int(data.shape[0]))
            test_loss += loss2.item()
            if i == 0:
                ##
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(get_batch_size(epoch), 3, 256, 256)[:n]])
                save_image(comparison.cpu(),
                           output_dir + 'reconstruction_' + str(epoch) + '.png', nrow=n)

                del recon_batch

                n_image_gen = 10
                images = []
                n_samples_linspace = 20
                print(data.shape)
                if data_para:
                    data_latent = model.module.encode_latent_(data[:25, ...])
                else:
                    data_latent = model.encode_latent_(data)
                print(data_latent.shape)
                print(data.shape)
                for i in range(n_image_gen):
                    pt_1 = data_latent[i * 2, ...].cpu().numpy()
                    pt_2 = data_latent[i * 2 + 1, ...].cpu().numpy()
                    sample_vec = interpolate_points(pt_1, pt_2,
                                                    np.linspace(0, 1, num=n_samples_linspace, endpoint=True))
                    sample_vec = torch.from_numpy(sample_vec).to(device)
                    if data_para:
                        images.append(model.module.decode(sample_vec).cpu())
                    else:
                        images.append(model.decode(sample_vec).cpu())

                save_image(torch.cat(images), output_dir + 'linspace_' + str(epoch) + '.png',
                           nrow=n_samples_linspace)

    test_loss /= len(val_loader_food.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    print('loss', losses.avg)

    val_losses.append(test_loss)


for epoch in range(starting_epoch, epochs):
    if epoch != starting_epoch and epoch % 15 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(LR * 0.9, 5.0e-5)

    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = LR
    for param_group in optimizer.param_groups:
        print("Current learning rate is: {}".format(param_group['lr']))

    loss = train(epoch)
    test(epoch)

    torch.save({
        'epoch': epoch,
        'encoder_state_dict': model.module.encoder.state_dict(),
        'decoder_state_dict': model.module.decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, save_files + 'epoch_' + str(epoch) + '.pt')
    torch.save(model.module, "model_inf.pt")
    with torch.no_grad():
        sample = torch.randn(64, 512).to(device)
        sample = model.module.decode(sample).cpu()
        save_image(sample.view(64, 3, 256, 256),
                   output_dir + 'sample_' + str(epoch) + '.png')
