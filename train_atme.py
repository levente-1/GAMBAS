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

if __name__ == '__main__':

    # -----  Loading the init options -----
    opt = TrainOptions().parse()

    # -----  Transformation and Augmentation process for the data  -----
    min_pixel = int(opt.min_pixel * ((opt.patch_size[0] * opt.patch_size[1] * opt.patch_size[2]) / 100))
    trainTransforms = [
                NiftiDataset.DeterministicCrop((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]), 77, 4)
                ]

    if opt.model == 'atme':
        # DiscPool for ATME requires indices of input images to be stored
        train_set = NiftiDataSet_atme(opt.data_path, which_direction='AtoB', transforms=trainTransforms, shuffle_labels=False, train=True, outputIndices=True, repeats=4)
    else:
        train_set = NiftiDataSet(opt.data_path, which_direction='AtoB', transforms=trainTransforms, shuffle_labels=False, train=True)
    print('lenght train list:', len(train_set))
    # print((train_set[0][1].shape))
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, pin_memory=True)  # Here are then fed to the network with a defined batch size

    # -----------------------------------------------------
    model = create_model(opt)  # creation of the model
    model.setup(opt)
    if opt.epoch_count > 1:
        model.load_networks(opt.epoch_count)
    visualizer = Visualizer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
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










