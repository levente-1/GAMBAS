from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument("--image", type=str, default='/media/hdd/levibaljer/Khula_GANs/paired/flirt_test/images/30.nii.gz')
        parser.add_argument("--result", type=str, default='/media/hdd/levibaljer/Khula_GANs/paired/flirt_test/results/30_ea.nii', help='path to the .nii result to save')
        parser.add_argument('--phase', type=str, default='test', help='test')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument("--stride_inplane", type=int, nargs=1, default=32, help="Stride size in 2D plane")
        parser.add_argument("--stride_layer", type=int, nargs=1, default=32, help="Stride size in z direction")

        parser.set_defaults(model='test')
        self.isTrain = False
        return parser