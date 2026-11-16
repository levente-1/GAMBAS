import sys
from utils.NiftiDataset import *
import utils.NiftiDataset as NiftiDataset
from torch.utils.data import DataLoader
from options.train_options import TrainOptions
# from logger import *
import time
import random
import numpy as np
import torch
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
    # Validation: fixed order and (below) a fixed seed, so every validation uses the same crops
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, pin_memory=True)

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

        print("----------------- RUNNING VALIDATION -----------------")
        # FIXED: the old loop logged get_current_losses() after model.test(), but test() only runs forward() and the
        # losses are computed in backward(), so it re-logged the last TRAINING losses. Now: mean L1 between the
        # generator output and the target on the validation crops (fixed seed -> the same crops every time).
        val_l1, n_val = 0.0, 0
        cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(1234)
            np_state, py_state = np.random.get_state(), random.getstate()
            np.random.seed(1234); random.seed(1234)
            for i, data in enumerate(val_loader):
                model.set_input(data)
                model.test()
                if hasattr(model, 'fake_B') and hasattr(model, 'real_B'):
                    val_l1 += float((model.fake_B - model.real_B).abs().mean()) * model.real_B.shape[0]
                    n_val += model.real_B.shape[0]
            np.random.set_state(np_state); random.setstate(py_state)
        if n_val:
            val_losses.append(val_l1 / n_val)
            print('Validation L1 (generator vs target, [-1, 1] units): %.4f' % val_losses[-1])
            TBLogger(phase='val', step=epoch, L1=val_losses[-1])
        TBLogger.writer.flush()
            









