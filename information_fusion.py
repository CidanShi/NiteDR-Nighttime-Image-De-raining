import torch
from utils import features_grad
import torch.nn.functional as F
from pytorch_msssim import ssim

def information_fusion(v_img, l_img, device, model, feature_model, optimizer, mse_loss, ep_loss1, ep_loss2, ep_loss):
    c = 3200
    v_img_f = (v_img + 1) / 2#0~1
    v_img_f = v_img_f.to(device)
    l_img_f = (l_img + 1) / 2#0~1
    l_img_f = l_img_f.to(device)

    with torch.no_grad():#禁止后向传播
        feat_1 = torch.cat((v_img_f, v_img_f, v_img_f), dim=1)
        feat_1 = feature_model(feat_1)
        feat_2 = torch.cat((l_img_f, l_img_f, l_img_f), dim=1)
        feat_2 = feature_model(feat_2)

        for i in range(len(feat_1)):
            m1 = torch.mean(features_grad(feat_1[i]).pow(2), dim=[1, 2, 3])
            m2 = torch.mean(features_grad(feat_2[i]).pow(2), dim=[1, 2, 3])
            if i == 0:
                w1 = torch.unsqueeze(m1, dim=-1)
                w2 = torch.unsqueeze(m2, dim=-1)
            else:
                w1 = torch.cat((w1, torch.unsqueeze(m1, dim=-1)), dim=-1)
                w2 = torch.cat((w2, torch.unsqueeze(m2, dim=-1)), dim=-1)
        weight_1 = torch.mean(w1, dim=-1) / c
        weight_2 = torch.mean(w2, dim=-1) / c
        weight_list = torch.cat((weight_1.unsqueeze(-1), weight_2.unsqueeze(-1)), -1)
        weight_list = F.softmax(weight_list, dim=-1)

    v_img = v_img.to(device)
    l_img = l_img.to(device)

    optimizer.zero_grad()
    torch.cuda.synchronize()
    fusion_img = model(v_img, l_img)
    torch.cuda.synchronize()
    fusion_img = (fusion_img + 1) / 2#0~1

    v_img = (v_img + 1) / 2
    l_img = (l_img + 1) / 2

    # loss_1 = weight_list[:, 0] * (1 - ssim(fusion_img, v_img, nonnegative_ssim=True)) \
    #                      + weight_list[:, 1] * (1 - ssim(fusion_img, l_img, nonnegative_ssim=True))
    # loss_1 = torch.mean(loss_1)#ssim loss
    #
    # loss_2 = weight_list[:, 0] * mse_loss(fusion_img, v_img) \
    #                      + weight_list[:, 1] * mse_loss(fusion_img, l_img)
    # loss_2 = torch.mean(loss_2)#mse loss

    loss_1 = 0.5 * (1 - ssim(fusion_img, v_img, nonnegative_ssim=True)) \
             + 0.5 * (1 - ssim(fusion_img, l_img, nonnegative_ssim=True))
    loss_1 = torch.mean(loss_1)  # ssim loss

    loss_2 = 0.5 * mse_loss(fusion_img, v_img) \
             + 0.5 * mse_loss(fusion_img, l_img)
    loss_2 = torch.mean(loss_2)  # mse loss

    loss = loss_1 + 20 * loss_2
    ep_loss1.append(loss_1.item())
    ep_loss2.append(loss_2.item())
    ep_loss.append(loss.item())
    loss.backward()
    optimizer.step()

    return fusion_img, ep_loss1, ep_loss2, ep_loss