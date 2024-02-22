from runpy import run_path

import numpy as np
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import to_pil_image

from CLFU_dataset import TestDataset
from Dense import DenseNet
from refine import Finetunemodel
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
from utils import YCrCb2rgb, rgb2YCrCb
from Fu_train import option
import torch.nn.functional as F
from skimage import img_as_ubyte
import cv2

def get_weights_and_parameters(parameters):
    opt = option()
    weights = os.path.join(opt.derained_checkpoint_dir, 'best_net_g.pth')
    return weights, parameters

def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def test():
    opt = option()
    data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                                         std=[0.5, 0.5, 0.5])])
    test_dataset = TestDataset(test_root_dir=opt.test_root_dir,
                              mean=None,
                              std=None,
                              transform=data_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

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

    fusion_model = DenseNet().to(device)
    refine_model = Finetunemodel('experiments/fusion/checkpoints/fusion.pth').to(device)

    state = torch.load('experiments/fusion/checkpoints/fusion.pth')
    fusion_model.load_state_dict(state['fusion_model'])

    clean_model.eval()
    fusion_model.eval()
    refine_model.eval()
    with torch.no_grad():
        filenames =test_dataset.get_filenames()
        for batch, data_dict in enumerate(test_dataloader):
            if torch.cuda.is_available():
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()

            vis_input = data_dict['vis_input']  # vis_input_1 is now available
            inf_input = data_dict['inf_input']
            print('Processing picture No.{}'.format(batch + 1))

            img_multiple_of = 8

            input_ = vis_input.to(device)

            # Pad the input if not_multiple_of 8
            height, width = input_.shape[2], input_.shape[3]
            H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, (
                        (width + img_multiple_of) // img_multiple_of) * img_multiple_of
            padh = H - height if height % img_multiple_of != 0 else 0
            padw = W - width if width % img_multiple_of != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            ## Testing on the original resolution image
            restored = clean_model(input_)

            restored = torch.clamp(restored, 0, 1)

            # Unpad the output
            restored = restored[:, :, :height, :width]

            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
            restored = img_as_ubyte(restored[0])

            vis_clean = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
            vis_clean = cv2.cvtColor(vis_clean, cv2.COLOR_BGR2YCrCb)

            v_img = data_transform(vis_clean)
            v_img = torch.unsqueeze(v_img, dim=0)
            v_img = v_img.to(device)

            l_img = inf_input.to(device)
            v_img_y = v_img[:, 0:1, :, :]
            l_img_y = l_img[:, 0:1, :, :]
            fused_img_cr = v_img[:, 1:2, :, :]
            fused_img_cb = v_img[:, 2:3, :, :]

            fused_img_y = fusion_model(v_img_y, l_img_y)
            fused_img = torch.cat((fused_img_y, fused_img_cr, fused_img_cb), dim=1)
            fused_img = (fused_img + 1) / 2
            rgb_img = YCrCb2rgb(fused_img[:, 0:1, :, :], fused_img[:, 1:2, :, :],
                                    fused_img[:, 2:3, :, :])

            _, refine_img = refine_model(rgb_img, l_img)

            if opt.save_path is not None:
                if not os.path.exists(opt.save_path):
                    os.makedirs(opt.save_path)
                filename = os.path.splitext(os.path.basename(filenames[batch]))[0]

                # save_img((os.path.join(opt.save_path, filename + '_clean' + '.jpg')), restored)
                #
                # save_image(rgb_img, (opt.save_path + filename + '_fusion' + '.png'))
                # save_image(refine_img, (opt.save_path + filename + '_refine' + '.png'))
                save_image(refine_img, (opt.save_path + filename + '.jpg'))

        print('Finished testing!')

if __name__ == "__main__":
    test()