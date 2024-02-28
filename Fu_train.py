# Second-stage Training for Fusion Task
import os
from runpy import run_path
import cv2
import torch.nn.functional as F
from skimage import img_as_ubyte


import argparse
import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader

from CLFU_dataset import TrainDataset, TestDataset

from Dense import DenseNet
from basicsr.utils import get_time_str
from vgg import vgg16
from torch.optim import Adam, lr_scheduler
from information_fusion import information_fusion
from utils import YCrCb2rgb, rgb2YCrCb
import  random
import matplotlib
import matplotlib.pyplot as plt
from refine import Refinement
from torch.utils.tensorboard import SummaryWriter

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# 导入get_weights_and_parameters函数和其他必要的函数和库
def get_weights_and_parameters(parameters):
    weights = os.path.join(opt.derained_checkpoint_dir, 'best_net_g.pth')
    return weights, parameters

def clean_process(img, model, device):

    img_multiple_of = 8

    # 处理TrainDataset的图像
    with torch.no_grad():
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

        input_ = img.to(device)

        # Pad the input if not_multiple_of 8
        height, width = input_.shape[2], input_.shape[3]
        H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, ((width + img_multiple_of) // img_multiple_of) * img_multiple_of
        padh = H - height if height % img_multiple_of != 0 else 0
        padw = W - width if width % img_multiple_of != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

        ## Testing on the original resolution image
        restored = model(input_)

        restored = torch.clamp(restored, 0, 1)

        # Unpad the output
        restored = restored[:, :, :height, :width]

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])
        return restored

def get_patch(patch_size, input_img):

    h, w = input_img.shape[:2]
    stride = patch_size

    x = random.randint(0, w - stride)
    y = random.randint(0, h - stride)

    input_img = input_img[y:y + stride, x:x + stride, :]

    return input_img


def cascaded_train(args):
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    EPS = 1e-8

    data_transfrom = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                                     std=[0.5, 0.5, 0.5])])
    train_dataset = TrainDataset(train_root_dir=args.train_root_dir,
                                 scale=args.scale,
                                 gt_size=args.gt_size,
                                 patch_size=args.patch_size,
                                 geometric_augs=True,
                                 mean=None,
                                 std=None,
                                 transform=data_transfrom,
                                 is_train=True)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=4, pin_memory=False, collate_fn=None)

    val_dataset = TestDataset(test_root_dir=args.test_root_dir,
                              mean=None,
                              std=None,
                              transform=data_transfrom)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型参数和权重
    parameters = {'inp_channels': 3, 'out_channels': 3, 'dim': 48, 'num_blocks': [4, 6, 6, 8],
                  'heads': [1, 2, 4, 8], 'ffn_expansion_factor': 2.66, 'bias': False, 'LayerNorm_type': 'WithBias'}
    weights, parameters = get_weights_and_parameters(parameters)

    load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'CLformer_arch.py'))
    clean_model = load_arch['CLformer'](**parameters)

    clean_model.to(device)

    checkpoint = torch.load(weights)
    clean_model.load_state_dict(checkpoint['params'])

    # fusion
    fusion_model = DenseNet().to(device)
    feature_model = vgg16().to(device)
    feature_model.load_state_dict(torch.load('./model/vgg16-397923af.pth'))
    fusion_optimizer = Adam(fusion_model.parameters(), lr=args.fusion_lr)
    fusion_scheduler = lr_scheduler.ExponentialLR(fusion_optimizer, gamma=0.9)  # learning rate
    mse_loss = nn.MSELoss(reduction='mean').to(device)
    # refine
    refine_model = Refinement(args.stage).to(device)
    # init
    refine_model.balance.in_conv.apply(refine_model.weights_init)
    refine_model.balance.conv.apply(refine_model.weights_init)
    refine_model.balance.out_conv.apply(refine_model.weights_init)
    refine_model.adjust.in_conv.apply(refine_model.weights_init)
    refine_model.adjust.convs.apply(refine_model.weights_init)
    refine_model.adjust.out_conv.apply(refine_model.weights_init)
    refine_optimizer = Adam(refine_model.parameters(), lr=args.refine_lr, betas=(0.9, 0.99), weight_decay=3e-4)

    # loss
    fusion_loss = []
    refinement_loss = []

    # tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, f"train_{'fusion'}_{get_time_str()}.log"))

    for epoch in range(args.epochs):
        # epoch
        ep_floss1 = []
        ep_floss2 = []
        ep_loss = []
        ep_iloss = []
        loop = tqdm(train_dataloader, leave=True, desc="Epoch{}".format(epoch))
        for _, data_dict in enumerate(loop):
            vis_input_1 = data_dict['vis_input_1']  # vis_input_1 is now available
            inf_input = data_dict['inf_input']  # inf_input is now available

            # generating clean-visible output from stage 1
            clean_model.eval()
            vis_clean = clean_process(vis_input_1, clean_model, device)
            vis_clean = cv2.cvtColor(vis_clean, cv2.COLOR_RGB2BGR)
            vis_clean = cv2.cvtColor(vis_clean, cv2.COLOR_BGR2YCrCb)

            vis_clean = get_patch(args.patch_size, vis_clean)
            v_img = data_transfrom(vis_clean)
            v_img = torch.unsqueeze(v_img, dim=0)
            l_img = inf_input

            for _ in range(opt.cascaded_size):
                v_img = v_img.to(device)
                l_img = l_img.to(device)
                v_img_y = v_img[:, 0:1, :, :]  # only y channel
                l_img_y = l_img[:, 0:1, :, :]  # [-1 ,1]

                # fusion training
                fused_img_y, fused_loss_1, _, fused_loss = information_fusion(
                    v_img_y, l_img_y, device, fusion_model,
                    feature_model, fusion_optimizer, mse_loss,
                    ep_floss1, ep_floss2, ep_loss
                )  # fusion_img [0~1]
                img_cr = torch.cat((v_img[:, 1:2, :, :], l_img[:, 1:2, :, :]), dim=0).to(device)
                img_cb = torch.cat((v_img[:, 2:3, :, :], l_img[:, 2:3, :, :]), dim=0).to(device)
                w_cr = (torch.abs(img_cr) + EPS) / torch.sum(torch.abs(img_cr) + EPS, dim=0)
                w_cb = (torch.abs(img_cb) + EPS) / torch.sum(torch.abs(img_cb) + EPS, dim=0)
                fused_img_cr = torch.sum(w_cr * img_cr, dim=0, keepdim=True).clamp(-1, 1)
                fused_img_cb = torch.sum(w_cb * img_cb, dim=0, keepdim=True).clamp(-1, 1)

                fused_img = torch.cat((fused_img_y, fused_img_cr, fused_img_cb), dim=1)
                fused_img = (fused_img + 1) / 2
                fused_img = fused_img.data

                rgb_img = YCrCb2rgb(fused_img[:, 0:1, :, :], fused_img[:, 1:2, :, :], fused_img[:, 2:3, :, :])  # rgb
                # refine training
                refine_optimizer.zero_grad()
                _, refine_img, _, _ = refine_model(rgb_img, l_img)  # rgb img, l_img
                ref_loss = refine_model._loss(rgb_img, l_img)  # loss step
                ep_iloss.append(ref_loss.item())
                ref_loss.backward()
                nn.utils.clip_grad_norm_(refine_model.parameters(), 5)
                refine_optimizer.step()
                refine_img = refine_img[0]
                refine_img = rgb2YCrCb(refine_img[:, 0:1, :, :], refine_img[:, 1:2, :, :], refine_img[:, 2:3, :, :])  # yCrcb
                v_img = refine_img.data  # 增强融合的图像


        # validation
        # 创建当前epoch的文件夹
        if opt.val_path is not None:
            current_epoch_val_path = os.path.join(opt.val_path, f'epoch_{epoch}')
            if not os.path.exists(current_epoch_val_path):
                os.makedirs(current_epoch_val_path)

        with torch.no_grad():
            fusion_model.eval()
            refine_model.eval()
            filenames = val_dataset.get_filenames()
            for batch, data_dict in enumerate(val_dataloader):
                vis_input_val = data_dict['vis_input']
                inf_input_val = data_dict['inf_input']
                clean_model.eval()
                vis_clean_val_img = clean_process(vis_input_val, clean_model, device)

                vis_clean_val = cv2.cvtColor(vis_clean_val_img, cv2.COLOR_RGB2BGR)
                vis_clean_val = cv2.cvtColor(vis_clean_val, cv2.COLOR_BGR2YCrCb)

                v_img = data_transfrom(vis_clean_val)
                v_img_val = torch.unsqueeze(v_img, dim=0)
                v_img_val = v_img_val.to(device)
                l_img_val = inf_input_val.to(device)
                v_img_val_y = v_img_val[:, 0:1, :, :]
                l_img_val_y = l_img_val[:, 0:1, :, :]
                fused_img_val_cr = v_img_val[:, 1:2, :, :]
                fused_img_val_cb = v_img_val[:, 2:3, :, :]

                fused_img_val_y = fusion_model(v_img_val_y, l_img_val_y)
                fused_img_val = torch.cat((fused_img_val_y, fused_img_val_cr, fused_img_val_cb), dim=1)
                fused_img_val = (fused_img_val + 1) / 2
                rgb_img_val = YCrCb2rgb(fused_img_val[:, 0:1, :, :], fused_img_val[:, 1:2, :, :], fused_img_val[:, 2:3, :, :])

                _, refine_img_val, _, _ = refine_model(rgb_img_val, l_img_val)

                # saving validation results
                if opt.val_path is not None:
                    filename = os.path.splitext(os.path.basename(filenames[batch]))[0]

                    save_img((os.path.join(current_epoch_val_path, f'{filename}_clean.png')), vis_clean_val_img)
                    save_image(rgb_img_val, os.path.join(current_epoch_val_path, f'{filename}_fusion.png'))
                    save_image(refine_img_val[0],
                               os.path.join(current_epoch_val_path, f'{filename}_refine.png'))

        fusion_scheduler.step()

        fusion_loss.append(np.mean(fused_loss))
        refinement_loss.append(np.mean(ref_loss.item()))
        state = {
            'fusion_model': fusion_model.state_dict(),
            'refine_model': refine_model.state_dict(),
            'fusion_loss': fusion_loss,
            'refinement_loss': refinement_loss,
            'fusion_lr': fusion_optimizer.param_groups[0]['lr'],
            'ref_lr': refine_optimizer.param_groups[0]['lr']
        }
        torch.save(state, args.model_path + args.fusion_refine_model)
        # print
        matplotlib.use('Agg')
        fig1 = plt.figure()
        plot_loss_list_1 = fusion_loss
        plt.plot(plot_loss_list_1)
        plt.savefig(os.path.join(args.plt_logs_dir, 'fusion_loss_curve.png'))
        fig2 = plt.figure()
        plot_loss_list_2 = refinement_loss
        plt.plot(plot_loss_list_2)
        plt.savefig(os.path.join(args.plt_logs_dir, 'refinement_loss_curve.png'))

    writer.close()


def option():
    parser = argparse.ArgumentParser(description='cross-sensor information fusion')

    parser.add_argument('--seed', type=int, default=100,
                        help='random seed')
    parser.add_argument('--epochs', type=int, default=500,
                        help='training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='training batch size')
    parser.add_argument('--cascaded_size', type=int, default=3,
                        help='cascaded size')
    parser.add_argument('--fusion_lr', type=int, default=1e-4,
                        help='fusion learning rate')
    parser.add_argument('--refine_lr', type=float, default=0.0003,
                        help='refine learning rate')
    parser.add_argument('--log_dir', type=str, default='./experiments/fusion/logs_fusion/',
                        help='log file path')
    parser.add_argument('--plt_logs_dir', type=str, default='./experiments/fusion/plt_logs_fusion/',
                        help='plt_logs file path')
    parser.add_argument('--train_root_dir', type=str, default='./dataset/RoadScene-rain/train/',
                        help='Path to the training data')
    parser.add_argument('--test_root_dir', type=str, default='./dataset/RoadScene-rain/test/',
                        help='Path to the test data')
    parser.add_argument('--derained_checkpoint_dir', type=str, default='./experiments/de-raining/checkpoints/',
                        help='best training model of the first stage')
    parser.add_argument('--save_path', type=str, default='./testing/test_results/',
                        help='testing results  dataset directory')
    parser.add_argument('--val_path', type=str, default='./experiments/fusion/validation/',
                        help='validation results  dataset directory')
    parser.add_argument('--model_path', type=str, default='./experiments/fusion/checkpoints/',
                        help='trained model directory')
    parser.add_argument('--fusion_refine_model', type=str, default='fusion.pth',
                        help='fusion model name')
    parser.add_argument('--stage', type=int, default=3, help='adjust steps')
    parser.add_argument('--scale', type=int, default=1, help='Scale factor')
    parser.add_argument('--patch_size', type=int, default=64, help='Size of the infrared patch')
    parser.add_argument('--gt_size', type=int, default=64, help='The image target size.')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = option()
    cascaded_train(opt)


