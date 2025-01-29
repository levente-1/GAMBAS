import torch
from collections import OrderedDict
from torch.autograd import Variable
# import util.util as util
from .base_model import BaseModel
from . import networks3D
from torchvision import models
import random
# from generative.losses import PerceptualLoss

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


class I2IMambaOneModel(BaseModel):
    def name(self):
        return "I2IMambaOneModel"

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_f', type=float, default=0.9, help='momentum term for f')
            parser.add_argument('--lambda_A', type=float, default=100.0, help='lambda for sobel l1 loss')
            parser.add_argument('--lambda_adv', type=float, default=1.0, help='weight for adversarial loss')
            parser.add_argument('--vit_name', type=str, default='Res-ViT-B_16',help='vit type')
            parser.add_argument('--imageSize', type=int, default=256, help='size of largest axis from input 3D volume (if all equal, then this is the size of all axes)')
            parser.add_argument('--pre_trained_resnet', type=int, default=0,help='Pre-trained residual CNNs or not')
            parser.add_argument('--pre_trained_path', type=str, default='/media/hdd/levibaljer/ResViT/checkpoints/khula_Res_CNN/latest_net_G.pth', help='path to the pre-trained resnet architecture')
            parser.add_argument('--pre_trained_transformer', type=int, default=0,help='Pre-trained ViT or not')
            parser.add_argument('--lambda_perc', type=float, default=1.0, help='weight for perceptual loss')
            # parser.add_argument('--lambda_L1', type=float, default=300.0, help='weight for L1 loss') # According to paper, this is fixed at 300
            # parser.add_argument('--lambda_sobel', type=float, default=100.0, help='lambda for sobel l1 loss') # Sobel starts at 0, linearly increases in first 15% of epochs, then stays at 100
            # parser.add_argument('--rise_sobelLoss', action='store_true', help='indicate to rise sobel lambda')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D', 'G_GAN', 'G_L1']
        # self.loss_names = ['D', 'G_GAN', 'G_L1', 'G_perc']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names = ['real_A', 'fake_B', 'real_B']

        self.visual_names = visual_names
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']


        # load/define networks
        self.netG = networks3D.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, 
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                      **{'vit_name': opt.vit_name, 
                                         'img_size': (opt.imageSize, opt.imageSize), 
                                         'pre_trained_resnet': opt.pre_trained_resnet, 
                                         'pre_trained_path': opt.pre_trained_path,
                                         'pre_trained_transformer': opt.pre_trained_transformer
                                         }
                                      )

        if self.isTrain:
            self.lambda_f = opt.lambda_f
            use_sigmoid = opt.no_lsgan
            self.netD = networks3D.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD, 
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, "G", opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, "D", opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks3D.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # self.criterionPerc = PerceptualLoss(spatial_dims=3, network_type="alex").to(self.device)
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


        # print("---------- Networks initialized -------------")
        # networks.print_network(self.netG)
        # if self.isTrain:
        #     networks.print_network(self.netD)
        # print("-----------------------------------------------")

    def set_input(self, input):
        AtoB = self.opt.which_direction == "AtoB"
        self.real_A = input[0 if AtoB else 1].to(self.device)
        self.real_B = input[1 if AtoB else 0].to(self.device)

    def forward(self):
        self.fake_B = self.netG(self.real_A)
        # self.fake_sobel = networks3D.sobelLayer(self.fake_B)
        # self.real_sobel = networks3D.sobelLayer(self.real_B).detach() 

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        # fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B, self.fake_sobel), 1))
        self.pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        # real_AB = torch.cat((self.real_A, self.real_B, self.real_sobel), 1)
        self.pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        # fake_AB = torch.cat((self.real_A, self.fake_B, self.fake_sobel), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
        # Third, perceptual loss: (G(A)) = (B)
        # self.loss_G_perc = self.criterionPerc(self.fake_B, self.real_B) * self.opt.lambda_perc

        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        # self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_perc
        
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
    
    # def update_sobel_lambda(self, epochNum):
    #     self.sobelLambda = self.opt.lambda_sobel/150*epochNum
    #     print('update sobel lambda: %f' % (self.sobelLambda))



