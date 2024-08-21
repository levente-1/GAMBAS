from .base_model import BaseModel
from . import networks3D
from .cycle_gan_model import CycleGANModel
from .pix2pix_model import Pix2PixModel
from .ea_gan_model import EaGANModel
from .deprecated.pup_gan_model import PUPGANModel
from .deprecated.atme_model import ATMEModel
from .resvit_one_model import ResViTOneModel
import torch


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        assert not is_train, 'TestModel cannot be used in train mode'
        # parser = EaGANModel.modify_commandline_options(parser, is_train=False)
        parser = ResViTOneModel.modify_commandline_options(parser, is_train=False)
        parser.set_defaults(dataset_mode='single')

        parser.add_argument('--model_suffix', type=str, default='',
                            help='In checkpoints_dir, [which_epoch]_net_G[model_suffix].pth will'
                            ' be loaded as the generator of TestModel')

        return parser

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = []
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['G' + opt.model_suffix]
        # Add net W if testing ATME
        # self.model_names = ['G' + opt.model_suffix, 'W' + opt.model_suffix]

        # self.netG = networks3D.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
        #                               opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        
        self.netG = networks3D.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, 
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                      **{'vit_name': 'Res-ViT-B_16', 
                                         'img_size': (256, 256), 
                                         'pre_trained_resnet': 0, 
                                         'pre_trained_path': None,
                                         'pre_trained_transformer': 0
                                         }
                                      )
        
        # Use DDPM model if testing ATME
        # self.netG = networks3D.define_G(1, 1, 64, 'unet_256_ddm', 'instance',
        #                               not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, 
        #                               **{'dim': 64, 
        #                                  'dim_mults': (1,2,4,8), 
        #                                  'init_dim': 64, 
        #                                  'resnet_groups': 8})

        # Add net W if testing ATME
        # self.netW = networks3D.define_W(opt.init_type, opt.init_gain, self.gpu_ids)

        # Initialise noise if testing ATME
        # self.disc_B = 0.5 + 0.001 * torch.randn((1, 1, 6, 6, 6), dtype=torch.float32)

        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see BaseModel.load_networks
        setattr(self, 'netG' + opt.model_suffix, self.netG)

        # Add net W if testing ATME
        # setattr(self, 'netW' + opt.model_suffix, self.netW)

    def set_input(self, input):
        # we need to use single_dataset mode
        self.real_A = input.to(self.device)  # the torch tensor patch in the inference function
        

    def forward(self):
        self.fake_B = self.netG(self.real_A)

        # If testing ATME, add noise
        # self.Disc_B = self.netW(self.disc_B)
        # self.noisy_A = self.real_A * (1 + self.Disc_B)
        # self.fake_B = self.netG(self.noisy_A, self.Disc_B)
