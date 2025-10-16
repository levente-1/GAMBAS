from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument("--image", type=str, default='/path/to/test/image.nii.gz', help='path to the .nii image to be tested')
        parser.add_argument("--result", type=str, default='/path/to/test/result.nii.gz', help='path to the .nii result to save')
        parser.add_argument('--phase', type=str, default='test', help='test')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument("--stride_inplane", type=int, nargs=1, default=32, help="Stride size in 2D plane")
        parser.add_argument("--stride_layer", type=int, nargs=1, default=32, help="Stride size in z direction")
        parser.add_argument('--input', type=str, default='/workspace/input/Testing_data', help='Path to input test data (only for Docker inference)')
        parser.add_argument('--output', type=str, default='/workspace/output/Predictions', help='Path to output predictions (only for Docker inference)')

        parser.set_defaults(model='test')
        self.isTrain = False
        return parser