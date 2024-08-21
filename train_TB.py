import sys
from utils.NiftiDataset import *
import utils.NiftiDataset as NiftiDataset
from torch.utils.data import DataLoader
from options.train_options import TrainOptions
# from logger import *
import time
from models import create_model
from utils.visualizer import Visualizer
# from test import inference
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':

    class TensorBoardLogger():
        def __init__(self, log_dir, **kwargs):
            self.log_dir = log_dir
            self.writer = SummaryWriter(log_dir, **kwargs)
            
        
        def __call__(self, phase, step, **kwargs):
            for key, value in kwargs.items():
                self.writer.add_scalar(f'{key}/{phase}', value, step)

    # -----  Loading the init options -----
    opt = TrainOptions().parse()

    # -----  Setting up TensorBoard logger -----

    outputDir = os.path.join(opt.checkpoints_dir, opt.name)
    TBLogger = TensorBoardLogger(log_dir = os.path.join(outputDir, 'tb_logs'))

    # -----  Transformation and Augmentation process for the data  -----
    min_pixel = int(opt.min_pixel * ((opt.patch_size[0] * opt.patch_size[1] * opt.patch_size[2]) / 100))
    trainTransforms = [
                NiftiDataset.Resample(opt.new_resolution, opt.resample),
                NiftiDataset.Augmentation(),
                NiftiDataset.Padding((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2])),
                NiftiDataset.RandomCrop((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]), opt.drop_ratio, min_pixel)
                ]

    valTransforms = [
                NiftiDataset.RandomCrop((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]), opt.drop_ratio, min_pixel)
                ]

    if opt.model == 'atme':
        # DiscPool for ATME requires indices of input images to be stored
        train_set = NiftiDataSet(opt.data_path, which_direction='AtoB', transforms=trainTransforms, shuffle_labels=False, train=True, outputIndices=True)
    else:
        train_set = NiftiDataSet(opt.data_path, which_direction='AtoB', transforms=trainTransforms, shuffle_labels=False, train=True)
    val_set = NiftiDataSet(opt.val_path, which_direction='AtoB', transforms=valTransforms, shuffle_labels=False, train=False, test=True)
    print('length train set:', len(train_set))
    print((train_set[0][1].shape))
    print('length val set:', len(val_set))
    print((train_set[0][1].shape))
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, pin_memory=True)  # Here are then fed to the network with a defined batch size
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, pin_memory=True)

    # -----------------------------------------------------
    model = create_model(opt)  # creation of the model
    model.setup(opt)
    if opt.epoch_count > 1:
        model.load_networks(opt.epoch_count)
    visualizer = Visualizer(opt)
    total_steps = 0

    train_losses = []
    val_losses = []
    loss_names = model.loss_names

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        train_losses.append(0)
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(train_loader):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()

            total_losses = 0
            for loss_name in loss_names:
                total_losses += model.get_current_losses()[loss_name]
            train_losses[-1] += total_losses

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

        if opt.model == 'ea_gan' and epoch <= 150:
            model.update_sobel_lambda(epoch)

        TBLogger(phase='train', step=epoch, loss=train_losses[-1])

        # model.eval()
        val_losses.append(0)
        print("----------------- RUNNING VALIDATION -----------------")
        for i, data in enumerate(val_loader):
            model.set_input(data)
            model.test()

            total_losses = 0
            for loss_name in loss_names:
                total_losses += model.get_current_losses()[loss_name]
            val_losses[-1] += total_losses

        TBLogger(phase='val', step=epoch, loss=val_losses[-1])
            









