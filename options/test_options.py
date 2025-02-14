import flywheel
from pathlib import Path
from base_options import BaseOptions  # BaseOptions is defined elsewhere

class TestOptions(BaseOptions):
    def initialize(self, parser):
        # Initialize parser from BaseOptions
        parser = BaseOptions.initialize(self, parser)

        # Initialize Flywheel context
        context = flywheel.GearContext()
        config = context.config

        # Define default paths
        input_dir = Path("/flywheel/v0/input")
        output_dir = Path("/flywheel/v0/output")

        # Find the first available NIfTI file in input directory
        input_files = list(input_dir.glob("*.nii.gz"))
        if not input_files:
            raise FileNotFoundError("No NIfTI image found in the input directory.")

        input_file = input_files[0]  # Select first available file
        input_filename = input_file.name

        # Assign dynamic paths
        image_path = str(input_file)
        result_path = str(output_dir / f"{input_filename.split('.')[0]}_gambas.nii.gz")

        # Parse arguments from manifest/config.json
        parser.add_argument("--image", type=str, default=image_path, help="Path to input NIfTI image")
        parser.add_argument("--result", type=str, default=result_path, help="Path to save the result NIfTI file")
        parser.add_argument("--phase", type=str, default=config.get("phase", "test"), help="Test phase")
        parser.add_argument("--which_epoch", type=str, default=config.get("which_epoch", "latest"), help="Epoch to load")
        parser.add_argument("--stride_inplane", type=int, default=int(config.get("stride_inplane", 32)), help="Stride size in 2D plane")
        parser.add_argument("--stride_layer", type=int, default=int(config.get("stride_layer", 32)), help="Stride size in Z direction")

        parser.set_defaults(model='test')
        self.isTrain = False

        return parser
