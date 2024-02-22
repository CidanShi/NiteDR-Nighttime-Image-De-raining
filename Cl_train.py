# First_stage Training for Cleaning Task
import datetime
import math
import time
import torch.distributed as dist

import numpy as np
import torch
import argparse
import random
import os
import logging

from CLFU_dataset import TrainDataset, TestDataset
from basicsr.data import create_dataloader
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import create_model
from basicsr.utils import get_time_str, tensor2img
from basicsr.utils.logger import get_root_logger, init_tb_logger, MessageLogger


def init_loggers(use_tb_logger=True):
    args = parse_options()
    # Create a log file path using the provided log_file argument
    log_file = os.path.join(args.logger_path,
                        f"train_{'De-raining'}_{get_time_str()}.log")
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    # Create a logger for terminal logging
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info('Logging to file: {}'.format(log_file))

    # Create a TensorBoard logger if use_tb_logger is True
    tb_logger = None
    if use_tb_logger:
        tb_logger = init_tb_logger(log_dir=os.path.join(args.experiments_root, 'tb_logger', args.exp_name, os.path.basename(log_file)))

    return logger, tb_logger

def check_resume(args, resume_iter):
    """Check resume states and pretrain_network paths.

    Args:
        args (argparse.Namespace): Command-line arguments.
        resume_iter (int): Resume iteration.
    """
    logger = get_root_logger()
    if args.resume_state:
        # Get all the networks
        networks = [key for key in args.__dict__.keys() if key.startswith('network_')]
        flag_pretrain = False
        for network in networks:
            if getattr(args, f'pretrain_{network}', None) is not None:
                flag_pretrain = True
        if flag_pretrain:
            logger.warning('pretrain_network path will be ignored during resuming.')
        # Set pretrained model paths
        for network in networks:
            name = f'pretrain_{network}'
            basename = network.replace('network_', '')
            if getattr(args, 'ignore_resume_networks', None) is None or (
                    basename not in args.ignore_resume_networks):
                setattr(args, name, os.path.join(args.models, f'net_{basename}_{resume_iter}.pth'))
                logger.info(f"Set {name} to {getattr(args, name)}")


def clean_train():
    # parse options, set distributed setting, set ramdom seed
    args = parse_options()

    torch.backends.cudnn.benchmark = True
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)

    # automatic resume ..
    state_folder_path = f'experiments/De-raining/training_states/'
    try:
        states = os.listdir(state_folder_path)
    except:
        states = []

    resume_state = None
    if len(states) > 0:
        max_state_file = f'{max([int(x[0:-6]) for x in states])}.state'
        resume_state = os.path.join(state_folder_path, max_state_file)
        args.resume_state = resume_state

    # load resume states if necessary
    if args.resume_state:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(args.resume_state, map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    # mkdir for experiments and logger
    if resume_state is None:
        # Create experiments root directory
        experiments_root = args.experiments_root
        if not os.path.exists(experiments_root):
            os.makedirs(experiments_root)

        # Create logger directory
        log_dir = os.path.join(experiments_root, 'tb_logger', args.exp_name)
        if args.use_tb_logger and 'debug' not in args.exp_name and args.local_rank == 0:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

     # initialize loggers
    logger, tb_logger = init_loggers(args)

    train_dataset = TrainDataset(train_root_dir=args.train_root_dir,
                                 scale=args.scale,
                                 gt_size=args.gt_size,
                                 patch_size=args.patch_size,
                                 geometric_augs=True,
                                 mean=None,
                                 std=None,
                                 transform=None,
                                 is_train=True)
    dataset_enlarge_ratio = args.dataset_enlarge_ratio

    train_sampler = EnlargedSampler(train_dataset, args.world_size,
                                            args.rank, dataset_enlarge_ratio)

    # train_dataloader = DataLoader(train_dataset, batch_size=args.num_worker_per_gpu, #shuffle=True,
    #                               num_workers=args.batch_size_per_gpu, sampler=train_sampler, pin_memory=True)
    train_dataloader = create_dataloader(dataset=train_dataset, phase='train',
                                         num_worker_per_gpu=args.num_worker_per_gpu,
                                         batch_size_per_gpu=args.batch_size_per_gpu,
                                         num_gpu=2,
                                         dist=True,  # Set to True for distributed training
                                         sampler=train_sampler,
                                         seed=args.seed,
                                         )

    num_iter_per_epoch = math.ceil(
            len(train_dataset) * dataset_enlarge_ratio /
            (args.batch_size_per_gpu * args.world_size))
    total_iters = args.total_iter
    total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
    logger.info(
        'Training statistics:'
        f'\n\tNumber of train images: {len(train_dataset)}'
        f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
        f'\n\tBatch size per gpu: {args.batch_size_per_gpu}'
        f'\n\tWorld size (gpu number): {args.world_size}'
        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

    val_dataset = TestDataset(test_root_dir=args.test_root_dir,
                              mean=None,
                              std=None,
                              transform=None)
    # val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    val_dataloader = create_dataloader(
        dataset=val_dataset,
        phase='val',
        num_worker_per_gpu=0,
        batch_size_per_gpu=1,
        num_gpu=2,
        dist=True,  # Set to True for distributed training
        sampler=None,
        seed=args.seed
    )

    logger.info(
        f'Number of val images/folders in ValSet: '
        f'{len(val_dataset)}')

    # Create model
    if args.resume_state:  # Resume training
        check_resume(args, args.resume_state['iter'])
        model = create_model(args)
        model.resume_training(args.resume_state)  # Handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {args.resume_state['epoch']}, "
                    f"iter: {args.resume_state['iter']}.")
        start_epoch = args.resume_state['epoch']
        current_iter = args.resume_state['iter']
    else:
        model = create_model(args)
        start_epoch = 0
        current_iter = 0

    # Create message logger (formatted outputs)
    msg_logger = MessageLogger(args, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = args.prefetch_mode
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_dataloader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_dataloader, args)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.'
                         "Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(
        f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()

    # for epoch in range(start_epoch, total_epochs + 1):

    iters = args.iters
    batch_size = args.batch_size_per_gpu
    mini_batch_sizes = args.mini_batch_sizes
    gt_size = args.gt_size
    mini_gt_sizes = args.gt_sizes

    groups = np.array([sum(iters[0:i + 1]) for i in range(0, len(iters))])

    logger_j = [True] * len(groups)

    scale = args.scale

    epoch = start_epoch
    best_val_metric = 0
    while current_iter <= total_iters:
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_time = time.time() - data_time

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(
                current_iter, warmup_iter=-1)

            ### ------Progressive learning ---------------------
            j = ((current_iter > groups) != True).nonzero()[0]
            if len(j) == 0:
                bs_j = len(groups) - 1
            else:
                bs_j = j[0]

            mini_gt_size = mini_gt_sizes[bs_j]
            mini_batch_size = mini_batch_sizes[bs_j]

            if logger_j[bs_j]:
                logger.info('\n Updating Patch_Size to {} and Batch_Size to {} \n'.format(mini_gt_size,
                                                                                          mini_batch_size * torch.cuda.device_count()))
                logger_j[bs_j] = False

            lq = train_data['vis_input']
            gt = train_data['vis_target']


            if mini_batch_size < batch_size:
                indices = random.sample(range(0, batch_size), k=mini_batch_size)
                lq = lq[indices]
                gt = gt[indices]

            if mini_gt_size < gt_size:
                x0 = int((gt_size - mini_gt_size) * random.random())
                y0 = int((gt_size - mini_gt_size) * random.random())
                x1 = x0 + mini_gt_size
                y1 = y0 + mini_gt_size
                lq = lq[:, :, x0:x1, y0:y1]
                gt = gt[:, :, x0 * scale:x1 * scale, y0 * scale:y1 * scale]

            ###-------------------------------------------


            # model.feed_train_data({'lq': lq, 'gt': gt})
            model.feed_train_data({'vis_input': lq, 'vis_target': gt})

            model.optimize_parameters(current_iter)

            iter_time = time.time() - iter_time

            # log
            if current_iter % args.logger_print_freq == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # save models and training states
            if current_iter % args.logger_save_checkpoint_freq == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            # validation
            if current_iter % args.val_freq == 0:
                rgb2bgr = True
                # wheather use uint8 image to compute metrics
                use_image = True
                save_img = False
                # model.validation(val_dataloader, current_iter, tb_logger,
                #                  save_img, rgb2bgr, use_image)

                current_metric = model.validation(val_dataloader, current_iter, tb_logger,
                                                  save_img, rgb2bgr, use_image)

                # Compare current validation metric with best validation metric
                if current_metric > best_val_metric:  # Update this condition based on your metric
                    logger.info('Saving best model based on validation metric.')
                    model.save_best(epoch, current_iter, current_metric)
                    best_val_metric = current_metric

            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()
        # end of iter
        epoch += 1

    # end of epoch

    consumed_time = str(
        datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    save_img = True
    model.validation(val_dataloader, current_iter, tb_logger,
                     save_img)
    if tb_logger:
        tb_logger.close()

def parse_options():
    parser = argparse.ArgumentParser(description='information cleaning')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm'],
                        default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', action='store_true', help='Enable distributed training')
    parser.add_argument('--rank', type=int, default=0, help='Rank of current process in distributed training')
    parser.add_argument('--world_size', type=int, default=2, help='Total number of processes in distributed training')
    parser.add_argument('--seed', type=int, default=100, help='Seed for random number generation')
    parser.add_argument('--is_train', action='store_true',
                        help='Flag indicating training mode')
    parser.add_argument('--prefetch_mode', choices=[None, 'cpu', 'cuda'], default=None, help='Dataloader prefetch mode')

    parser.add_argument('--batch_size_per_gpu', type=int, default=8, help='Batch size for training')
    parser.add_argument('--mini_batch_sizes', type=int, nargs='+', default=[4], help='The mini batch sizes.')
    parser.add_argument('--iters', type=int, nargs='+', default=[300000], help='The total training iterations.')
    parser.add_argument('--gt_size', type=int, default=64, help='The image target size.')
    parser.add_argument('--gt_sizes', type=int, nargs='+', default=[64], help='The image target sizes.')
    parser.add_argument('--scale', type=int, default=1, help='Scale factor')
    parser.add_argument('--patch_size', type=int, default=64, help='Size of the infrared patch')
    parser.add_argument('--num_worker_per_gpu', type=int, default=8, help='Multi-threaded loading training data')
    parser.add_argument('--total_iter', type=int, default=300000, help='Number of iters for training')

    # learning rate settings
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--scheduler_type', type=str, default='CosineAnnealingRestartCyclicLR',
                        help='Type of scheduler.')
    parser.add_argument('--scheduler_periods', nargs='+', type=int, default=[92000, 208000],
                        help='Periods for the scheduler.')
    parser.add_argument('--scheduler_restart_weights', nargs='+', type=float, default=[1.0, 1.0],
                        help='Restart weights for the scheduler.')
    parser.add_argument('--scheduler_eta_mins', nargs='+', type=float, default=[0.0003, 0.000001],
                        help='Eta mins for the scheduler.')

    parser.add_argument('--optim_type', type=str, default='AdamW', help='Type of optimizer for the generator.')
    parser.add_argument('--optim_weight_decay', type=float, default=1e-4,
                        help='Weight decay for the generator optimizer.')
    parser.add_argument('--optim_betas', type=float, nargs='+', default=[0.9, 0.999],
                        help='Coefficients used for computing running averages of gradient and its square')
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='')

    parser.add_argument('--mixup', action='store_true', help='Whether to use mixup data augmentation.')
    parser.add_argument('--mixup_beta', type=float, default=1.2,
                        help='Mixup beta parameter for linear interpolation of samples.')
    parser.add_argument('--use_identity', action='store_true',
                        help='Whether to use original samples in addition to mixup samples during training.')
    parser.add_argument('--dataset_enlarge_ratio', type=int, default=1.0, help='Ratio of dataset enlargement')

    parser.add_argument('--pixel_type', type=str, default='L1Loss', help='Type of pixel loss function.')
    parser.add_argument('--pixel_loss_weight', type=float, default=1.0, help='Weight for the pixel loss function.')
    parser.add_argument('--pixel_reduction', type=str, default='mean',
                        help='Reduction method for the pixel loss function.')
    parser.add_argument('--use_grad_clip', action='store_true', help='Whether to use gradient clipping.')

    parser.add_argument('--resume_state', type=str, default=None, help='Path to resume state file')
    parser.add_argument('--exp_name', type=str, default='De-raining', help='Name of the experiment')
    parser.add_argument('--train_root_dir', type=str, default='./dataset/RoadScene-rain/train/',
                        help='Path to the training data')
    parser.add_argument('--test_root_dir', type=str, default='./dataset/RoadScene-rain/test/',
                        help='Path to the test data')

    parser.add_argument('--pretrain_network_g_path', default=None, type=str, help='Path to pre-trained network_g weights.')
    parser.add_argument('--experiments_root', type=str, default='./experiments/de-raining_MDFFN',
                        help='Path to saving experiments results')
    parser.add_argument('--checkpoint_save_dir', type=str, default='./experiments/de-raining_MDFFN/checkpoints/',
                        help='Path to saving checkpoints')
    parser.add_argument('--training_states_path', type=str, default='./experiments/de-raining_MDFFN/training_states/',
                        help='Path to saving training states')
    parser.add_argument('--visualization_path', type=str, default='./experiments/de-raining_MDFFN/visualization/',
                        help='Path to saving val results')
    parser.add_argument('--logger_path', type=str, default='./experiments/de-raining_MDFFN/logger/')
    parser.add_argument('--window_size', type=int, default=8, help='Size of patch when validation')
    parser.add_argument('--val_freq', type=float, default=1000.0, help='Validation frequency.')
    parser.add_argument('--logger_print_freq', type=int, default=1000, help='Print frequency during training.')
    parser.add_argument('--logger_save_checkpoint_freq', type=float, default=1000.0,
                        help='Save checkpoint frequency during training.')
    parser.add_argument('--use_tb_logger', action='store_true', help='Whether to use Tensorboard logger.')


    args = parser.parse_args()

    return args

if __name__ == '__main__':
    # Initialize the distributed communication backend
    # dist.init_process_group(backend='nccl')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend='nccl')
    clean_train()

