import torch
import itertools
import random
from .base_model import BaseModel
from . import networks3D


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


class PUPGANModel(BaseModel):
    def name(self):
        return 'PUPGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_1', type=float, default=10.0, help='weight for cycle loss (A -> B -> A) in paired setting, identity in unpaired')
            parser.add_argument('--lambda_2', type=float, default=5.0, help='weight for identity in paired setting, cycle loss in unpaired')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D_A', 'D_B', 'D_C', 'D_D', 'G_A', 'G_B', 'cycle_A', 'cycle_B', 'idt_A', 'idt_B', 'G_dual_A', 'G_dual_B']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_2 > 0.0:
            visual_names_A.append('idt_A')
            visual_names_B.append('idt_B')

        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'D_C', 'D_D']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks3D.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,   # nc number channels
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks3D.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks3D.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks3D.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_C = networks3D.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_D = networks3D.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks3D.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_1 = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_2 = torch.optim.Adam(itertools.chain(self.netD_C.parameters(), self.netD_D.parameters()),
                                                lr=(opt.lr*0.5), betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_1)
            self.optimizers.append(self.optimizer_D_2)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input[0 if AtoB else 1].to(self.device)
        self.real_B = input[1 if AtoB else 0].to(self.device)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)

        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
    
    def backward_D_C(self, opt):
        if opt.super_train == 1:
            self.loss_D_C = self.backward_D_basic(self.netD_C, torch.cat((self.real_A, self.real_B), 1), torch.cat((self.fake_A, self.real_B), 1))
        else:
            self.loss_D_C = self.backward_D_basic(self.netD_C, torch.cat((self.real_A, self.fake_B), 1), torch.cat((self.rec_A, self.fake_B), 1))
    
    def backward_D_D(self, opt):
        if opt.super_train == 1:
            self.loss_D_D = self.backward_D_basic(self.netD_D, torch.cat((self.real_B, self.real_A), 1), torch.cat((self.fake_B, self.real_A), 1))
        else:
            self.loss_D_D = self.backward_D_basic(self.netD_D, torch.cat((self.real_B, self.fake_A), 1), torch.cat((self.rec_B, self.fake_A), 1))

    def backward_G(self, opt):
        lambda_1 = self.opt.lambda_1
        lambda_2 = self.opt.lambda_2

        # G_A should be identity if real_B is fed.
        self.idt_A = self.netG_A(self.real_B)
        self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_2
        # G_B should be identity if real_A is fed.
        self.idt_B = self.netG_B(self.real_A)
        self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_2

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)

        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # GAN loss: supervised = D_C(G_A(A), B) | unsupervised = D_C(G_B(G_A(A)), A)
        if opt.super_train == 1:
            pred_fake_D_C = self.netD_C(torch.cat((self.fake_A, self.real_B), 1))
        else:
            pred_fake_D_C = self.netD_C(torch.cat((self.rec_A, self.fake_B), 1))
        self.loss_G_dual_A = self.criterionGAN(pred_fake_D_C, True)

        # GAN loss: supervised = D_D(G_B(B), A) | unsupervised = D_D(G_A(G_B(B)), B)
        if opt.super_train == 1:
            pred_fake_D_D = self.netD_D(torch.cat((self.fake_B, self.real_A), 1))
        else:
            pred_fake_D_D = self.netD_D(torch.cat((self.rec_B, self.fake_A), 1))
        self.loss_G_dual_B = self.criterionGAN(pred_fake_D_D, True)
        
        # Forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_1
        # Backward cycle loss
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_1

        # combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_G_dual_A + self.loss_G_dual_B
        self.loss_G.backward()

    def optimize_parameters(self, opt):
        # forward
        self.forward()
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_C, self.netD_D], False)
        self.optimizer_G.zero_grad()
        self.backward_G(opt)
        self.optimizer_G.step()
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_C, self.netD_D], True)
        self.optimizer_D_1.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D_1.step()
        # D_C and D_D
        self.optimizer_D_2.zero_grad()
        self.backward_D_C(opt)
        self.backward_D_D(opt)
        self.optimizer_D_2.step()
