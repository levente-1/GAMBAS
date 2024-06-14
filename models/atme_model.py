import os
import torch
from .base_model import BaseModel
from . import networks3D
import random
from utils.NiftiDataset import *
# from util.image_pool import DiscPool
import utils.utils as util
from itertools import chain
# from data import create_dataset
from torch.utils.data import Dataset

# Have to see if ImagePool can function when the output of W(D) is dependent on the input 
# (i.e. the output of the previous W(D) is used as input to the next W(D)

# class ImagePool():
#     def __init__(self, pool_size):
#         self.pool_size = pool_size
#         if self.pool_size > 0:
#             self.num_imgs = 0
#             self.images = []

#     def query(self, images):
#         if self.pool_size == 0:
#             return images
#         return_images = []
#         for image in images:
#             image = torch.unsqueeze(image.data, 0)
#             if self.num_imgs < self.pool_size:
#                 self.num_imgs = self.num_imgs + 1
#                 self.images.append(image)
#                 return_images.append(image)
#             else:
#                 p = random.uniform(0, 1)
#                 if p > 0.5:
#                     random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
#                     tmp = self.images[random_id].clone()
#                     self.images[random_id] = image
#                     return_images.append(tmp)
#                 else:
#                     return_images.append(image)
#         return_images = torch.cat(return_images, 0)
#         return return_images

class DiscPool(Dataset):
    """This class implements a buffer that stores the previous discriminator map for each image in the dataset.

    This buffer enables us to recall the outputs of the discriminator in the previous epoch
    """

    def __init__(self, opt, device, isTrain=True, disc_out_size=[6, 6, 6]):
        """Initialize the DiscPool class

        Parameters:
            opt: stores all the experiment flags; needs to be a subclass of BaseOptions
            device: the device used
            isTrain: whether this class is instanced during the train or test phase
            disc_out_size: the size of the ouput tensor of the discriminator
        """
        import utils.NiftiDataset as NiftiDataset
        self.dataset_len = dataset_len = len(NiftiDataSet_atme(opt.data_path, which_direction='AtoB', shuffle_labels=False, train=True, outputIndices=True, repeats=4))

        if isTrain:
            # Initially the discriminator doesn't know real/fake because is not trained yet
            self.disc_out = torch.rand((dataset_len, 1, disc_out_size[0], disc_out_size[1], disc_out_size[2]), dtype=torch.float32)
        else:
            # At the end, the discriminator is expected to be near its maximum entropy state (D_i = 1/2)
            self.disc_out = 0.5 + 0.001 * torch.randn((dataset_len, 1, disc_out_size[0], disc_out_size[1], disc_out_size[2]), dtype=torch.float32)
        
        self.disc_out = self.disc_out.to(device)
    
    def __getitem__(self, _):
        raise NotImplementedError('DiscPool does not support this operation')

    def __len__(self):
        return self.dataset_len
    
    def query(self, img_idx):
        """Return the last discriminator map from the pool, corresponding to given image indices.

        Parameters:
            img_idx: indices of the images that the discriminator just processed

        Returns discriminator map from the buffer.
        """
        return self.disc_out[img_idx]
    
    def insert(self, disc_out, img_idx):
        """Insert the last discriminator map in the pool, corresponding to given image index.

        Parameters:
            disc_out: output from the discriminator in the backward pass of generator
            img_idx: indices of the images that the discriminator just processed
        """
        self.disc_out[img_idx] = disc_out

class AtmeModel(BaseModel):
    def name(self):
        return 'atmemodel'

    """ This class implements the ATME model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet_256_attn' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    atme paper: https://arxiv.org/pdf/x.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For atme, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with instance norm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='instance', netG='unet_256_ddm', netD='basic', dataset_mode='aligned')
        parser.add_argument('--n_save_noisy', type=int, default=0, help='number of D_t and W_t to keep track of')
        parser.add_argument('--mask_size', type=int, default=256)
        parser.add_argument('--dim', type=int, default=64, help='dim for the ddm UNet')
        parser.add_argument('--dim_mults', type=tuple, default=(1,2,4,8), help='dim_mults for the ddm UNet')
        parser.add_argument('--groups', type=int, default=8, help='number of groups for GroupNorm within ResnetBlocks')
        parser.add_argument('--init_dim', type=int, default=64, help='output channels after initial conv2d of x_t')
        parser.add_argument('--learned_sinusoidal_cond', type=bool, default=False, help='learn fourier features for positional embedding?')
        parser.add_argument('--random_fourier_features', type=bool, default=False, help='random fourier features for positional embedding?')
        parser.add_argument('--learned_sinusoidal_dim', type=int, default=16, help='twice the number of fourier frequencies to learn')
        parser.add_argument('--time_dim_mult', type=int, default=4, help='dim * time_dim_mult amounts to output channels after time-MLP')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def initialize(self, opt):
        """Initialize the atme class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.initialize(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'Disc_B', 'noisy_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D', 'W']
        else:  # during test time, only load G
            self.model_names = ['G', 'W'] 
        # define networks (both generator and discriminator)

        # self.netG = networks3D.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
        #                               not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, 
        #                               **{'dim': opt.dim, 
        #                                  'dim_mults': opt.dim_mults, 
        #                                  'init_dim': opt.init_dim, 
        #                                  'resnet_groups': opt.groups})
        self.netG = networks3D.define_G(1, 1, 64, 'unet_256_ddm', 'instance',
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, 
                                      **{'dim': 64, 
                                         'dim_mults': (1,2,4,8), 
                                         'init_dim': 64, 
                                         'resnet_groups': 8})

        self.netW = networks3D.define_W(opt.init_type, opt.init_gain, self.gpu_ids)
        self.disc_pool = DiscPool(opt, self.gpu_ids[0], isTrain=self.isTrain)
    
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            use_sigmoid = opt.no_lsgan
            self.netD = networks3D.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
        
        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks3D.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(chain(self.netW.parameters(), self.netG.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        
               
        # prepare to save D_t and W_t history
        self.save_noisy = True if opt.n_save_noisy > 0 else False
        if self.save_noisy:
            self.save_DW_idx = torch.randint(len(NiftiDataSet(opt.data_path, which_direction='AtoB', shuffle_labels=False, train=True, outputIndices=True)), (opt.n_save_noisy,))
            self.img_DW_dir = os.path.join(opt.checkpoints_dir, opt.name, 'images_noisy')
            util.mkdir(self.img_DW_dir)

    def _save_DW(self, visuals):
        to_save = (self.batch_indices.view(1, -1) == self.save_DW_idx.view(-1, 1)).any(dim=0)
        if any(to_save) > 0:
            idx_to_save = torch.nonzero(to_save)[0]
            for label, images in visuals.items():
                for idx, image in zip(idx_to_save, images[to_save]):
                    img_idx = self.batch_indices[idx].item()
                    image_numpy = util.tensor2im(image[None])
                    img_path = os.path.join(self.img_DW_dir, f'epoch_{self.epoch:03d}_{label}_{img_idx}.png')
                    util.save_image(image_numpy, img_path)

    def set_input(self, input, epoch=None):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.epoch = epoch
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input[0 if AtoB else 1].to(self.device)
        self.real_B = input[1 if AtoB else 0].to(self.device)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.batch_indices = input[2]
        self.disc_B = self.disc_pool.query(self.batch_indices)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.Disc_B = self.netW(self.disc_B)
        self.noisy_A = self.real_A * (1 + self.Disc_B)
        self.fake_B = self.netG(self.noisy_A, self.Disc_B)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        self.disc_B = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(self.disc_B, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients 
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
        # Save discriminator output
        self.disc_pool.insert(self.disc_B.detach(), self.batch_indices)
        if self.save_noisy: # Save images corresponding to disc_B and Disc_B
            self._save_DW({'D': torch.sigmoid(self.disc_B), 'W': self.Disc_B})