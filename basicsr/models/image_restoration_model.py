import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm

from Cl_train import parse_options
from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial

args = parse_options()
network_g = {
            'type': 'CLformer',
            'inp_channels': 3,
            'out_channels': 3,
            'dim': 48,
            'num_blocks': [4, 6, 6, 8],
            'heads': [1, 2, 4, 8],
            'ffn_expansion_factor': 2.66,
            'bias': False,
            'LayerNorm_type': 'WithBias'
        }
class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1,1)).item()
    
        r_index = torch.randperm(target.size(0)).to(self.device)
    
        target = lam * target + (1-lam) * target[r_index, :]
        input_ = lam * input_ + (1-lam) * input_[r_index, :]
    
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments)-1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_

class ImageCleanModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, args):
        super(ImageCleanModel, self).__init__(args)
        self.best_val_metric = 0
        # define network
        self.mixing_flag = args.mixup = False
        if self.mixing_flag:
            mixup_beta = args.mixup_beta = 1.2
            use_identity = args.use_identity = False
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(network_g))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = args.pretrain_network_g_path
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              strict=True,
                              param_key='params')

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()

        self.ema_decay =args.ema_decay = 0
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            self.net_g_ema = define_network(network_g).to(
                self.device)
            load_path =args.pretrain_network_g_path
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                 strict=True, param_key='params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        pixel_type = args.pixel_type
        cri_pix_cls = getattr(loss_module, pixel_type)
        self.cri_pix = cri_pix_cls(args.pixel_loss_weight, reduction=args.pixel_reduction).to(
            self.device)


        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = args.optim_type
        # if optim_type == 'Adam':
        #     self.optimizer_g = torch.optim.Adam(optim_params, args.lr, args.optim_weight_decay, args.optim_betas)
        # elif optim_type == 'AdamW':
        #     self.optimizer_g = torch.optim.AdamW(optim_params, args.lr, args.optim_weight_decay, args.optim_betas)
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, lr=args.lr, weight_decay=args.optim_weight_decay,
                                                betas=(args.optim_betas[0], args.optim_betas[1]), eps=1e-8)
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=args.optim_weight_decay,
                                                 betas=(args.optim_betas[0], args.optim_betas[1]), eps=1e-8)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')

        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data['vis_input'].to(self.device)
        if 'vis_target' in data:
            self.gt = data['vis_target'].to(self.device)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        self.lq = data['vis_input'].to(self.device)
        if 'vis_target' in data:
            self.gt = data['vis_target'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        preds = self.net_g(self.lq)
        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        loss_dict = OrderedDict()
        # pixel loss
        l_pix = 0.
        for pred in preds:
            l_pix += self.cri_pix(pred, self.gt)

        loss_dict['l_pix'] = l_pix

        l_pix.backward()
        if args.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pad_test(self, window_size):
        scale = args.scale
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq      
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.get_filenames()
        # 定义metrics字典
        metrics = {
            'psnr': {
                # 'type': 'calculate_psnr',
                'crop_border': 0,
                'test_y_channel': True
            }
        }

        with_metrics = True
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in metrics.keys()
            }
        window_size = args.window_size

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['vis_input_path'][0]))[0]

            self.feed_data(val_data)
            test()

            visuals = self.get_current_visuals()
            input_img = tensor2img([visuals['vis_input']], rgb2bgr=rgb2bgr)
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if args.is_train:

                    save_img_path = osp.join(args.visualization_path,
                                             img_name,
                                             f'{img_name}_{current_iter}.png')

                    save_gt_img_path = osp.join(args.visualization_path,
                                                img_name,
                                                f'{img_name}_{current_iter}_gt.png')
                    # save_input_img_path = osp.join(args.visualization_path,
                    #                             img_name,
                    #                             f'{img_name}_{current_iter}_input.png')
                else:

                    save_img_path = osp.join(
                        args.visualization_path, dataset_name,
                        f'{img_name}.png')
                    save_gt_img_path = osp.join(
                        args.visualization_path, dataset_name,
                        f'{img_name}_gt.png')
                    
                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)
                # imwrite(input_img, save_input_img_path)

            if with_metrics:
                opt_metric = deepcopy(metrics)

                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type', 'calculate_psnr')  # Use 'calculate_psnr' as default if 'type' is not specified
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type', 'calculate_psnr')  # Use 'calculate_psnr' as default if 'type' is not specified
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)
            cnt += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['vis_input'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

    def save_best(self, epoch, current_iter, current_val_metric):
        if current_val_metric > self.best_val_metric:
            self.best_val_metric = current_val_metric
            if self.ema_decay > 0:
                self.save_best_network([self.net_g, self.net_g_ema],
                                       'best_net_g',
                                       current_iter,
                                       param_key=['params', 'params_ema'])
            else:
                self.save_best_network(self.net_g, 'best_net_g', current_iter)
            self.save_training_state(epoch, current_iter)
            # 更新最佳指标值和当前iter值到文件
            # best_metric_file = './experiments/de-raining/best_val_metric.txt'
            best_metric_file = os.path.join(args.experiments_root, 'best_val_metric.txt')
            # if not os.path.exists(best_metric_file):
            with open(best_metric_file, 'w') as f:
                f.write(f'Best Metric: {self.best_val_metric:.4f}\n')
                f.write(f'Best Iter: {current_iter}')