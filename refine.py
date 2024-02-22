import torch
import torch.nn as nn
from refinement_loss import LossFunction



class BalanceModule(nn.Module):
    def __init__(self, layers, channels):
        super(BalanceModule, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, f_input, l_input):
        input = torch.cat((f_input, l_input), dim=1)
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)

        illu = fea + f_input
        illu = torch.clamp(illu, 0.0001, 1)

        return illu


class AdjustModule(nn.Module):
    def __init__(self, layers, channels):
        super(AdjustModule, self).__init__()
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        self.layers = layers

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.convs)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)

        fea = self.out_conv(fea)
        delta = input - fea

        return delta



class Refinement(nn.Module):

    def __init__(self, stage=3):
        super(Refinement, self).__init__()
        self.stage = stage
        self.balance = BalanceModule(layers=1, channels=6)  # torch.cat
        self.adjust = AdjustModule(layers=3, channels=16)
        self._criterion = LossFunction()

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, f_input, l_input):

        ilist, rlist, inlist, attlist = [], [], [], []
        input_op = f_input
        for i in range(self.stage):
            inlist.append(input_op)
            i = self.balance(input_op, l_input)
            r = f_input / i
            r = torch.clamp(r, 0, 1)
            att = self.adjust(l_input)
            input_op = r + att
            ilist.append(i)
            rlist.append(r)
            attlist.append(torch.abs(att))

        return ilist, rlist, inlist, attlist

    def _loss(self, f_input, l_input):
        i_list, en_list, in_list, _ = self(f_input, l_input)
        loss = 0
        for i in range(self.stage):
            loss += self._criterion(in_list[i], i_list[i])
        return loss



class Finetunemodel(nn.Module):

    def __init__(self, weights):
        super(Finetunemodel, self).__init__()
        self.balance = BalanceModule(layers=1, channels=6)  # torch.cat
        self._criterion = LossFunction()

        base_weights = torch.load(weights, map_location='cpu')
        pretrained_dict = base_weights['refine_model']
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, f_input, l_input):
        i = self.balance(f_input, l_input)
        r = f_input / i
        r = torch.clamp(r, 0, 1)
        return i, r


    def _loss(self, input):
        i, r = self(input)
        loss = self._criterion(input, i)
        return loss

