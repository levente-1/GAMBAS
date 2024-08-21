import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
# import utils.utils as util
from .base_model import BaseModel
from . import networks3D
import random

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images


class EaGANModel(BaseModel):
    def name(self):
        return 'EaGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=300.0, help='weight for L1 loss') # According to paper, this is fixed at 300
            parser.add_argument('--lambda_sobel', type=float, default=100.0, help='lambda for sobel l1 loss') # Sobel starts at 0, linearly increases in first 15% of epochs, then stays at 100
            parser.add_argument('--rise_sobelLoss', action='store_true', help='indicate to rise sobel lambda')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D', 'G_GAN', 'G_L1', 'G_sobelL1']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names = ['real_A', 'fake_B', 'real_B']

        self.visual_names = visual_names
        # define tensors
        # self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
        #                            opt.fineSize, opt.fineSize, opt.fineSize)
        # self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
        #                            opt.fineSize, opt.fineSize, opt.fineSize)
         # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        if self.opt.rise_sobelLoss:            
            self.sobelLambda = 0
        else:
            self.sobelLambda = self.opt.lambda_sobel

        # load/define networks
        self.netG = networks3D.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,   # nc number channels
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            self.D_channel = opt.input_nc + opt.output_nc + 1
            use_sigmoid = True 
            self.netD = networks3D.define_D(self.D_channel, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
                
        # if not self.isTrain or opt.continue_train:
        #     self.load_network(self.netG, 'G', opt.which_epoch)
        #     if self.isTrain:
        #         self.load_network(self.netD, 'D', opt.which_epoch)
        # if not self.isTrain:
        #     self.netG.eval()

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks3D.GANLoss(use_lsgan=False).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            # print('---------- Networks initialized -------------')
            # networks.print_network(self.netG)
            # networks.print_network(self.netD)
            # print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input[0 if AtoB else 1].to(self.device)
        self.real_B = input[1 if AtoB else 0].to(self.device)

    def forward(self):
        # self.real_A = Variable(self.input_A)
        # self.fake_B = self.netG.forward(self.real_A)
        # self.real_B = Variable(self.input_B)
        self.fake_B = self.netG(self.real_A)
        self.fake_sobel = networks3D.sobelLayer(self.fake_B)
        self.real_sobel = networks3D.sobelLayer(self.real_B).detach() 

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B          
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B, self.fake_sobel), 1))
        self.pred_fake = self.netD.forward(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

        # Real           
        real_AB = torch.cat((self.real_A, self.real_B, self.real_sobel), 1)
        self.pred_real = self.netD.forward(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator       
        # fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B, self.fake_sobel), 1)) # authors used imagepool again, this should just be the current output of G
        fake_AB = torch.cat((self.real_A, self.fake_B, self.fake_sobel), 1)
        pred_fake = self.netD.forward(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        # Third, sobel(G(A)) = sobel(B)
        self.loss_G_sobelL1 = self.criterionL1(self.fake_sobel, self.real_sobel) * self.sobelLambda

        # combined loss
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_sobelL1
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    # def get_current_visuals(self):
    #     real_A = util.tensor2array(self.real_A.data)
    #     fake_B = util.tensor2array(self.fake_B.data)
    #     real_B = util.tensor2array(self.real_B.data)
    #     return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    # def get_current_img(self):
    #     real_A = util.tensor2im(self.real_A.data)
    #     fake_B = util.tensor2im(self.fake_B.data)
    #     real_B = util.tensor2im(self.real_B.data)
    #     return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])


        
    #     return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])


    # def save(self, label):
    #     self.save_network(self.netG, 'G', label, self.gpu_ids)
    #     self.save_network(self.netD, 'D', label, self.gpu_ids)

    # Learning rate scheduler defined in networks3D and called in base_model
    # def update_learning_rate(self):
    #     lrd = self.opt.lr / self.opt.niter_decay
    #     lr = self.old_lr - lrd
    #     for param_group in self.optimizer_D.param_groups:
    #         param_group['lr'] = lr
    #     for param_group in self.optimizer_G.param_groups:
    #         param_group['lr'] = lr
    #     print('update learning rate: %f -> %f' % (self.old_lr, lr))
    #     self.old_lr = lr

    def update_sobel_lambda(self, epochNum):
        self.sobelLambda = self.opt.lambda_sobel/150*epochNum
        print('update sobel lambda: %f' % (self.sobelLambda))