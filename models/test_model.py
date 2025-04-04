from .base_model import BaseModel
from . import networks3D
from .cycle_gan_model import CycleGANModel
from .pix2pix_model import Pix2PixModel
from .ea_gan_model import EaGANModel
from .resvit_model import ResViTModel
from .gambas_model import GambasModel
import torch


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        assert not is_train, 'TestModel cannot be used in train mode'
        # parser = EaGANModel.modify_commandline_options(parser, is_train=False)
        parser = GambasModel.modify_commandline_options(parser, is_train=False)
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

        # self.netG = networks3D.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
        #                               opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        
        if opt.netG == 'gambas':
            self.netG = networks3D.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, 
                                            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                        **{'img_size': (128, 128, 128)})
        elif opt.netG == 'res_cnn' or opt.netG == 'resvit':
            self.netG = networks3D.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, 
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                      **{'vit_name': 'Res-ViT-B_16', 
                                         'img_size': (128, 128, 128), 
                                         'pre_trained_resnet': 1, 
                                         'pre_trained_path': 'pretrained path',
                                         'pre_trained_transformer': 1
                                         }
                                      )
        else:
            self.netG = networks3D.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, 
                                            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)


        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see BaseModel.load_networks
        setattr(self, 'netG' + opt.model_suffix, self.netG)

    def set_input(self, input):
        # we need to use single_dataset mode
        self.real_A = input.to(self.device)  # the torch tensor patch in the inference function
        

    def forward(self):
        self.fake_B = self.netG(self.real_A)
