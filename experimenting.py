from models import networks3D
import torch
import torch.nn as nn
from torch.nn import init
from utils.NiftiDataset import *
import utils.NiftiDataset as NiftiDataset
from torch.utils.data import DataLoader

netW = networks3D.define_W('normal', 0.02, [0])
netG = networks3D.define_G(1, 1, 64, 'unet_256_ddm', 'instance',
                                      True, 'normal', 0.02, [0], 
                                      **{'dim': 64, 
                                         'dim_mults': (1,2,4,8), 
                                         'init_dim': 64, 
                                         'resnet_groups': 8})

min_pixel = int(0.1 * ((128 * 128 * 64) / 100))
trainTransforms = [
                NiftiDataset.Resample((0.45, 0.45, 0.45), False),
                NiftiDataset.Augmentation(),
                NiftiDataset.Padding((128, 128, 64)),
                NiftiDataset.RandomCrop((128, 128, 64), 0, min_pixel)
                ]
# train_set = NiftiDataSet('/media/hdd/levibaljer/ExperimentingKhula', which_direction='AtoB', transforms=trainTransforms, shuffle_labels=False, train=True, outputIndices=True)

# testImage = train_set.__getitem__(0)[0].unsqueeze(0).to('cuda:0')
testImage = torch.randn((1, 1, 64, 64, 64), dtype=torch.float32).to('cuda:0')

disc_out = 0.5 + 0.001 * torch.randn((1, 1, 6, 6, 6), dtype=torch.float32)
outW = netW(disc_out)

print(outW.shape)

noisy_Input = testImage * (1 + outW)
print(noisy_Input.shape)

# print('hi')

outG = netG(noisy_Input, outW)

print(outG.shape)
