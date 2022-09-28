import argparse
from asyncore import write
from csv import writer
import os
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import cv2
import numpy as np
import torch.onnx
import math

from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable

from utils import *
from tqdm import tqdm

from dataset import SR_Dataset, SR_Dataset_x3, SR_Dataset_x3_hrd, NR_Dataset, TSSR_Dataset_x3
from model.sr_qat import SrNet, NrNet, NrNet_D
from model.rcan import RCAN

# LPIPS
# import LPIPS.models.dist_model as dm
#
# model_LPIPS = dm.DistModel()
# model_LPIPS.initialize(model='net-lin', net='alex', use_gpu=True)

# # Setup warnings
import warnings

warnings.filterwarnings(action='ignore', category=DeprecationWarning, module=r'.*')
warnings.filterwarnings(action='default', module=r'torch.quantization')


def Huber(input, target, delta=0.01, reduce=True):
    abs_error = torch.abs(input - target)
    quadratic = torch.clamp(abs_error, max=delta)

    linear = (abs_error - quadratic)
    losses = 0.5 * torch.pow(quadratic, 2) + delta * linear

    if reduce:
        return torch.mean(losses)
    else:
        return losses


def L1_Loss(input, target, reduce=True):
    abs_error = torch.abs(input - target)
    if reduce:
        return torch.mean(abs_error)
    else:
        return abs_error


def get_features(input, md, is_target):
    if is_target:
        model = NrNet(ch_in=net_inp_ch, ch_out=net_out_ch, ch=opt.ch, skip=opt.skip, scale=opt.scale).to(device).eval()
        model.state_dict(md.state_dict())
    else:
        model = md
    features = []
    input, scale_in = model._modules['quant_in'](input, None)
    for _, layer in enumerate(model.features.children()):
        input, scale_in = layer(input, scale_in)
        features.append(input)

    return features


def self_dist_loss(HQ_features, LQ_features):
    loss = 0
    for idx in range(len(HQ_features)):
        loss += F.l1_loss(HQ_features[idx], LQ_features[idx])
    return loss

if __name__ == '__main__':
    cudnn.benchmark = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_hr', type=str, nargs='+')
    parser.add_argument('--dir_hrd', type=str)
    parser.add_argument('--dir_lr', type=str, nargs='+')
    parser.add_argument('--dir_hlr_', type=str, nargs='+')
    parser.add_argument('--nrsr', default='sr', type=str)
    parser.add_argument('--dir_out', type=str)
    parser.add_argument('--in_format', type=str, default='RGB')
    parser.add_argument('--out_format', type=str, default='RGB')
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--ch', type=int, default=128)
    parser.add_argument('--skip', type=str, default='on', choices=['on', 'off'])
    parser.add_argument('--perchannel', type=str, choices=['on', 'off'])
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma_tv', type=float, default=1e-3)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--isquant', type=int, default=0)
    parser.add_argument('--isPerceptual', type=int, default=0)
    parser.add_argument('--q_epoch', type=int, default=25)
    parser.add_argument('--p_weight', type=float)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # RCAN Model specifications
    parser.add_argument('--model', default='RCAN',
                        help='model name')

    parser.add_argument('--act', type=str, default='relu',
                        help='activation function')
    parser.add_argument('--pre_train', type=str, default='.',
                        help='pre-trained model directory')
    parser.add_argument('--extend', type=str, default='.',
                        help='pre-trained model directory')
    parser.add_argument('--n_resblocks', type=int, default=10,
                        help='number of residual blocks')
    parser.add_argument('--n_feats', type=int, default=64,
                        help='number of feature maps')
    parser.add_argument('--res_scale', type=float, default=3,
                        help='residual scaling')
    parser.add_argument('--shift_mean', default=True,
                        help='subtract pixel mean from the input')
    parser.add_argument('--precision', type=str, default='single',
                        choices=('single', 'half'),
                        help='FP precision for test (single | half)')
    parser.add_argument('--n_resgroups', type=int, default=10,
                        help='number of residual groups')
    parser.add_argument('--reduction', type=int, default=16,
                        help='number of feature maps reduction')

    parser.add_argument('--rgb_range', type=int, default=255,
                        help='maximum value of RGB')
    parser.add_argument('--n_colors', type=int, default=3,
                        help='number of color channels to use')
    parser.add_argument('--noise', type=str, default='.',
                        help='Gaussian noise std.')
    parser.add_argument('--chop', action='store_true',
                        help='enable memory-efficient forward')

    opt = parser.parse_args()

    parserNR = argparse.ArgumentParser()
    parserNR.add_argument('--model', default='RCAN',
                        help='model name')

    parserNR.add_argument('--act', type=str, default='relu',
                        help='activation function')
    parserNR.add_argument('--pre_train', type=str, default='.',
                        help='pre-trained model directory')
    parserNR.add_argument('--extend', type=str, default='.',
                        help='pre-trained model directory')
    parserNR.add_argument('--n_resblocks', type=int, default=10,
                        help='number of residual blocks')
    parserNR.add_argument('--n_feats', type=int, default=64,
                        help='number of feature maps')
    parserNR.add_argument('--res_scale', type=float, default=1,
                        help='residual scaling')
    parserNR.add_argument('--shift_mean', default=True,
                        help='subtract pixel mean from the input')
    parserNR.add_argument('--precision', type=str, default='single',
                        choices=('single', 'half'),
                        help='FP precision for test (single | half)')
    parserNR.add_argument('--n_resgroups', type=int, default=10,
                        help='number of residual groups')
    parserNR.add_argument('--reduction', type=int, default=16,
                        help='number of feature maps reduction')

    parserNR.add_argument('--rgb_range', type=int, default=255,
                        help='maximum value of RGB')
    parserNR.add_argument('--n_colors', type=int, default=3,
                        help='number of color channels to use')
    parserNR.add_argument('--noise', type=str, default='.',
                        help='Gaussian noise std.')
    parserNR.add_argument('--chop', action='store_true',
                        help='enable memory-efficient forward')

    optNR = parserNR.parse_args()

    torch.manual_seed(opt.seed)
    gamma_lr = 1.

    root_dir = './youtubedb'
    opt.dir_hr = [root_dir + '/hr/']
    opt.dir_lr = [root_dir + '/lr/']
    opt.dir_hlr = [root_dir + '/hlr/']

    if opt.perchannel == 'on':
        opt.perchannel = True
    else:
        opt.perchannel = False

    if opt.skip == 'on':
        opt.skip = True
    else:
        opt.skip = False

    if opt.isquant == 1:
        opt.isquant = True
    else:
        opt.isquant = False

    if opt.isPerceptual == 1:
        opt.isPerceptual = True
    else:
        opt.isPerceptual = False

    opt.dir_out = 'result/train/weight_RF_SD_' + str(opt.p_weight)
    writer = SummaryWriter('runs/weight_RF_SD_' + str(opt.p_weight))

    print(opt.dir_out)
    if not os.path.exists(opt.dir_out):
        os.makedirs(opt.dir_out)

    #### SR
    opt.isGAN = False
    opt.isPerceptual = True

    gain_hr = 1.0
    gain_dhr = 1.0
    gamma_adv = 0.01
    gamma_tv = 0. / ((opt.patch_size - 1) * opt.patch_size) / 2.

    transforms_train = transforms.Compose([transforms.ToTensor()])
    criterion = nn.L1Loss()

    net_inp_ch = 1 if opt.in_format == 'L' else 3
    net_out_ch = 1 if opt.out_format == 'L' else 3

    print("RF model created")
    #model_NR = NrNet(ch_in=net_inp_ch, ch_out=net_out_ch, ch=opt.ch, skip=opt.skip, scale=1).to(device)
    #model_SR = SrNet(ch_in=net_inp_ch, ch_out=net_out_ch, ch=opt.ch, skip=opt.skip, scale=opt.scale).to(device)
    model_NR = RCAN(args=optNR).to(device)
    model_SR = RCAN(args=opt).to(device)

    opt_model_NR = optim.Adam(model_NR.parameters(), lr=opt.lr)
    opt_model_SR = optim.Adam(model_SR.parameters(), lr=opt.lr)
    dataset = SR_Dataset_x3(dir_hr=opt.dir_hr, dir_lr=opt.dir_lr, dir_su = opt.dir_hlr, in_img_format=opt.in_format,
                            out_img_format=opt.out_format, transforms=transforms_train, patch_size=opt.patch_size)
    dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers,
                            pin_memory=True, drop_last=True)
    print("Data loading completed")

    epoch_loss_NR = AverageMeter()
    epoch_loss_SR = AverageMeter()

    n_mix = 0

    for epoch in range(opt.num_epochs + 1):
        model_NR.train()
        model_SR.train()
        with tqdm(total=(len(dataset) - len(dataset) % opt.batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {:3d}/{}'.format(epoch + 1, opt.num_epochs))
            for idx, (lr, hr, hlr) in enumerate(dataloader):
                lr = lr.to(device)
                hr = hr.to(device)
                hlr = hlr.to(device)

                # print(lr[0].size())
                # print(hr[0].size())
                #
                opt_model_NR.zero_grad()
                nr = model_NR(lr)

                # Calculate Pixel Loss - Generator

                # Calculate Dist Loss
                # HQ_features = get_features(hr, model, True)
                # LQ_features = get_features(lr, model, False)
                # loss_Dist = self_dist_loss(HQ_features, LQ_features)
                loss_NR = criterion(nr, hlr)

                # loss = loss_NR
                loss_NR.backward()
                opt_model_NR.step()

                opt_model_SR.zero_grad()
                sr = model_SR(hlr)
                loss_SR = criterion(sr, hr)
                loss_SR.backward()

                opt_model_SR.step()

                epoch_loss_NR.update(loss_NR.item(), len(lr))
                epoch_loss_SR.update(loss_SR.item(), len(lr))
                # epoch_loss_Dist.update(loss_Dist.item(), len(lr))

                _tqdm.set_postfix_str(s=f'NR: {epoch_loss_NR.avg:.6f}, SR: {epoch_loss_SR.avg:.6f}')
                _tqdm.update(len(lr))

        writer.add_scalar('NR.', epoch_loss_NR.avg, epoch + 1)
        writer.add_scalar('SR.', epoch_loss_SR.avg, epoch + 1)

        if epoch % 1 == 0:
            torch.save(model_NR.state_dict(), os.path.join(opt.dir_out, f'epoch_NR_{(epoch):03d}.pth'))
            torch.save(model_SR.state_dict(), os.path.join(opt.dir_out, f'epoch_SR_{(epoch):03d}.pth'))
        writer.close()
