import torch
import torch.nn as nn
import torch.nn.functional as F
# from dbb_transforms import *
import numpy as np


def transI_fusebn(kernel, bn):
    gamma = bn.weight
    std = (bn.running_var + bn.eps).sqrt()
    return kernel * ((gamma / std).reshape(-1, 1, 1, 1)), bn.bias - bn.running_mean * gamma / std


def transII_addbranch(kernels, biases):
    return sum(kernels), sum(biases)


def transIII_1x1_kxk(k1, b1, k2, b2, groups):
    if groups == 1:
        k = F.conv2d(k2, k1.permute(1, 0, 2, 3))  #
        b_hat = (k2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3))
    else:
        k_slices = []
        b_slices = []
        k1_T = k1.permute(1, 0, 2, 3)
        k1_group_width = k1.size(0) // groups
        k2_group_width = k2.size(0) // groups
        for g in range(groups):
            k1_T_slice = k1_T[:, g * k1_group_width:(g + 1) * k1_group_width, :, :]
            k2_slice = k2[g * k2_group_width:(g + 1) * k2_group_width, :, :, :]
            k_slices.append(F.conv2d(k2_slice, k1_T_slice))
            b_slices.append(
                (k2_slice * b1[g * k1_group_width:(g + 1) * k1_group_width].reshape(1, -1, 1, 1)).sum((1, 2, 3)))
        k, b_hat = transIV_depthconcat(k_slices, b_slices)
    return k, b_hat + b2


def transIV_depthconcat(kernels, biases):
    return torch.cat(kernels, dim=0), torch.cat(biases)


def transV_avg(channels, kernel_size, groups):
    input_dim = channels // groups
    k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
    k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
    return k


#   This has not been tested with non-square kernels (kernel.size(2) != kernel.size(3)) nor even-size kernels
def transVI_multiscale(kernel, target_kernel_size):
    H_pixels_to_pad = (target_kernel_size - kernel.size(2)) // 2
    W_pixels_to_pad = (target_kernel_size - kernel.size(3)) // 2
    return F.pad(kernel, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad])


def conv_bn(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
            padding_mode='zeros', batch_Norm=False):
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                           stride=stride, padding=padding, dilation=dilation, groups=groups,
                           bias=not batch_Norm, padding_mode=padding_mode)
    bn_layer = nn.BatchNorm2d(num_features=out_channels, affine=True) if batch_Norm else nn.Identity()
    se = nn.Sequential()
    se.add_module('conv', conv_layer)
    se.add_module('bn', bn_layer)  #####
    return se


class IdentityBasedConv1x1(nn.Conv2d):

    def __init__(self, channels, groups=1):
        super(IdentityBasedConv1x1, self).__init__(in_channels=channels, out_channels=channels, kernel_size=1, stride=1,
                                                   padding=0, groups=groups, bias=False)

        assert channels % groups == 0
        input_dim = channels // groups
        id_value = np.zeros((channels, input_dim, 1, 1))
        for i in range(channels):
            id_value[i, i % input_dim, 0, 0] = 1
        self.id_tensor = torch.from_numpy(id_value).type_as(self.weight)  # 恒等于1的卷积核
        nn.init.zeros_(self.weight)

    def forward(self, input):
        kernel = self.weight + self.id_tensor.to(self.weight.device)
        result = F.conv2d(input, kernel, None, stride=1, padding=0, dilation=self.dilation, groups=self.groups)
        return result

    def get_actual_kernel(self):
        return self.weight + self.id_tensor.to(self.weight.device)


class BNAndPadLayer(nn.Module):
    def __init__(self,
                 pad_pixels,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True,
                 batch_Norm=False):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine,
                                 track_running_stats) if batch_Norm else nn.Identity()
        self.pad_pixels = pad_pixels
        self.batch_Norm = batch_Norm

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.batch_Norm:
                if self.bn.affine:
                    pad_values = self.bn.bias.detach() - self.bn.running_mean * self.bn.weight.detach() / torch.sqrt(
                        self.bn.running_var + self.bn.eps)
                else:
                    pad_values = - self.bn.running_mean / torch.sqrt(self.bn.running_var + self.bn.eps)
                output = F.pad(output, [self.pad_pixels] * 4)
                pad_values = pad_values.view(1, -1, 1, 1)
                output[:, :, 0:self.pad_pixels, :] = pad_values
                output[:, :, -self.pad_pixels:, :] = pad_values
                output[:, :, :, 0:self.pad_pixels] = pad_values
                output[:, :, :, -self.pad_pixels:] = pad_values
            else:
                pad_values = torch.from_numpy(np.zeros((1, input.size(1), 1, 1))).type_as(input).view(1, -1, 1, 1).to(
                    input.device)
                output = F.pad(output, [self.pad_pixels] * 4)
                output[:, :, 0:self.pad_pixels, :] = pad_values
                output[:, :, -self.pad_pixels:, :] = pad_values
                output[:, :, :, 0:self.pad_pixels] = pad_values
                output[:, :, :, -self.pad_pixels:] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps


class DiverseBranchBlock(nn.Module):
    """
                               |
                               |
            ---------------------------------------------
            |           |               |               |
            |           |               |               |
           1x1         1x1             1x1             kxk
            |           |               |               |
            |           |               |               |
           bn          bn1             bn              bn
            |           |               |               |
            |           |               |               |
            |          kxk             avg              |
            |           |               |               |
            |           |               |               |
            |          bn2            avgbn             |
            --------------------Add----------------------
                               |
                               |
           Diverse Branch Block
           """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 internal_channels_1x1_3x3=None,
                 deploy=False, nonlinear=None, single_init=False, batch_Norm=False):
        super(DiverseBranchBlock, self).__init__()
        self.deploy = deploy
        self.batch_Norm = batch_Norm
        self.device = torch.device('cpu')

        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups
        assert padding == kernel_size // 2

        if deploy:  # 重参数化后
            self.dbb_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)

        else:  # 重参数化前
            ############################################################################################################
            # 主要卷积分支
            # 常规卷积( + batchnorm)  k=k  s=s
            self.dbb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, groups=groups,
                                      batch_Norm=batch_Norm)
            ############################################################################################################
            # 平均池化分支
            self.dbb_avg = nn.Sequential()
            if groups < out_channels:  # 主要卷积为常规卷积时
                # 常规卷积  k=1  s=1
                self.dbb_avg.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                          kernel_size=1, stride=1, padding=0, groups=groups,
                                                          bias=False))
                # (batchnorm + )填充
                self.dbb_avg.add_module('bn', BNAndPadLayer(pad_pixels=padding, num_features=out_channels,
                                                            batch_Norm=batch_Norm))  ######
                # 平均池化  k=k  s=s
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
            else:  # 主要卷积为分组卷积时
                # 平均池化  k=k  s=s
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))
            # (batchnorm)
            self.dbb_avg.add_module('avgbn', nn.BatchNorm2d(out_channels) if batch_Norm else nn.Identity())  #####

            #########################################################################################################
            # 1x1卷积分支(+ batchnorm)  k=1  s=s (只存在于s=1条件下)
            if stride == 1 and groups < out_channels:
                self.dbb_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                       stride=stride, padding=0, groups=groups, batch_Norm=batch_Norm)

            ############################################################################################################
            # 1x1_kxk卷积分支
            if internal_channels_1x1_3x3 is None:  # 主要卷积为常规卷积时 internal_channels_1x1_3x3=ic 否则为 2*ic
                internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels  # For mobilenet, it is better to have 2X internal channels

            self.dbb_1x1_kxk = nn.Sequential()
            if internal_channels_1x1_3x3 == in_channels:  # 主要卷积为常规卷积时
                # 1x1卷积 + 核为1的恒等卷积  k=1  s=1
                self.dbb_1x1_kxk.add_module('idconv1', IdentityBasedConv1x1(channels=in_channels, groups=groups))
            else:
                # 常规1x1卷积  k=1  s=1
                self.dbb_1x1_kxk.add_module('conv1',
                                            nn.Conv2d(in_channels=in_channels, out_channels=internal_channels_1x1_3x3,
                                                      kernel_size=1, stride=1, padding=0, groups=groups, bias=False))
            # (batchnorm + )填充
            self.dbb_1x1_kxk.add_module('bn1', BNAndPadLayer(pad_pixels=padding, num_features=internal_channels_1x1_3x3,
                                                             affine=True, batch_Norm=batch_Norm))  #####
            # 常规卷积  k=k  s=s
            self.dbb_1x1_kxk.add_module('conv2',
                                        nn.Conv2d(in_channels=internal_channels_1x1_3x3, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=0, groups=groups,
                                                  bias=not batch_Norm))
            # (batchnorm)
            self.dbb_1x1_kxk.add_module('bn2', nn.BatchNorm2d(out_channels) if batch_Norm else nn.Identity())  #####

        # The experiments reported in the paper used the default initialization of bn.weight (all as 1). But changing
        # the initialization may be useful in some cases.
        if single_init:
            #   Initialize the bn.weight of dbb_origin as 1 and others as 0. This is not the default setting.
            self.single_init()

    def get_equivalent_kernel_bias(self):
        # #############################################################################################################
        # 主要卷积分支  融合conv + bn
        if self.batch_Norm:
            k_origin, b_origin = transI_fusebn(self.dbb_origin.conv.weight, self.dbb_origin.bn)  # 融合卷积和batchnorm参数
        else:
            # k_origin, b_origin = self.dbb_origin.conv.weight, torch.zeros(self.dbb_origin.conv.weight.size(0)).type_as(self.dbb_origin.conv.weight).to(self.device)
            k_origin, b_origin = self.dbb_origin.conv.weight, self.dbb_origin.conv.bias

        # ############################################################################################################
        # 1x1卷积分支  融合conv + bn
        if hasattr(self, 'dbb_1x1'):
            if self.batch_Norm:
                k_1x1, b_1x1 = transI_fusebn(self.dbb_1x1.conv.weight, self.dbb_1x1.bn)
            else:
                # k_1x1, b_1x1 = self.dbb_1x1.conv.weight, torch.zeros(self.dbb_1x1.conv.weight.size(0)).type_as(self.dbb_1x1.conv.weight).to(self.device)
                k_1x1, b_1x1 = self.dbb_1x1.conv.weight, self.dbb_1x1.conv.bias
            k_1x1 = transVI_multiscale(k_1x1, self.kernel_size)  # 1x1卷积核周围填充0变为kxk卷积核
        else:
            k_1x1, b_1x1 = 0, 0

        # #############################################################################################################
        # 1x1_kxk卷积分支
        if hasattr(self.dbb_1x1_kxk, 'idconv1'):
            # 常规卷积时  融合恒等映射和1x1卷积
            k_1x1_kxk_first = self.dbb_1x1_kxk.idconv1.get_actual_kernel()
        else:
            k_1x1_kxk_first = self.dbb_1x1_kxk.conv1.weight
        if self.batch_Norm:
            k_1x1_kxk_first, b_1x1_kxk_first = transI_fusebn(k_1x1_kxk_first, self.dbb_1x1_kxk.bn1)
            k_1x1_kxk_second, b_1x1_kxk_second = transI_fusebn(self.dbb_1x1_kxk.conv2.weight, self.dbb_1x1_kxk.bn2)
        else:
            k_1x1_kxk_first, b_1x1_kxk_first = k_1x1_kxk_first, torch.zeros(k_1x1_kxk_first.size(0)).type_as(
                k_1x1_kxk_first).to(self.device)
            # k_1x1_kxk_second, b_1x1_kxk_second = self.dbb_1x1_kxk.conv2.weight, torch.zeros(
            # self.dbb_1x1_kxk.conv2.weight.size(0)).type_as(self.dbb_1x1_kxk.conv2.weight).to(self.device)
            k_1x1_kxk_second, b_1x1_kxk_second = self.dbb_1x1_kxk.conv2.weight, self.dbb_1x1_kxk.conv2.bias
        # 融合两个卷积核
        k_1x1_kxk_merged, b_1x1_kxk_merged = transIII_1x1_kxk(k_1x1_kxk_first, b_1x1_kxk_first, k_1x1_kxk_second,
                                                              b_1x1_kxk_second, groups=self.groups)

        # ###########################################################################################################
        # 平均池化分支
        # 平均池化操作转换为等价的卷积核
        k_avg = transV_avg(self.out_channels, self.kernel_size, self.groups)
        if self.batch_Norm:
            k_1x1_avg_second, b_1x1_avg_second = transI_fusebn(k_avg.to(self.dbb_avg.avgbn.weight.device),
                                                               self.dbb_avg.avgbn)
        else:
            k_1x1_avg_second, b_1x1_avg_second = k_avg.to(self.device), torch.zeros(k_avg.size(0)).type_as(k_avg).to(
                self.device)
        if hasattr(self.dbb_avg, 'conv'):
            if self.batch_Norm:
                k_1x1_avg_first, b_1x1_avg_first = transI_fusebn(self.dbb_avg.conv.weight, self.dbb_avg.bn)
            else:
                k_1x1_avg_first, b_1x1_avg_first = self.dbb_avg.conv.weight, torch.zeros(
                    self.dbb_avg.conv.weight.size(0)).type_as(self.dbb_avg.conv.weight).to(self.device)
                # k_1x1_avg_first, b_1x1_avg_first = self.dbb_avg.conv.weight, self.dbb_avg.conv.bias
            k_1x1_avg_merged, b_1x1_avg_merged = transIII_1x1_kxk(k_1x1_avg_first, b_1x1_avg_first, k_1x1_avg_second,
                                                                  b_1x1_avg_second, groups=self.groups)
        else:
            k_1x1_avg_merged, b_1x1_avg_merged = k_1x1_avg_second, b_1x1_avg_second

        # ##########################################################################################################
        # 融合所有分支
        return transII_addbranch((k_origin, k_1x1, k_1x1_kxk_merged, k_1x1_avg_merged),
                                 (b_origin, b_1x1, b_1x1_kxk_merged, b_1x1_avg_merged))

    def switch_to_deploy(self):
        if hasattr(self, 'dbb_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.dbb_reparam = nn.Conv2d(in_channels=self.dbb_origin.conv.in_channels,
                                     out_channels=self.dbb_origin.conv.out_channels,
                                     kernel_size=self.dbb_origin.conv.kernel_size, stride=self.dbb_origin.conv.stride,
                                     padding=self.dbb_origin.conv.padding, dilation=self.dbb_origin.conv.dilation,
                                     groups=self.dbb_origin.conv.groups, bias=True)
        self.dbb_reparam.weight.data = kernel
        self.dbb_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('dbb_origin')
        self.__delattr__('dbb_avg')
        if hasattr(self, 'dbb_1x1'):
            self.__delattr__('dbb_1x1')
        self.__delattr__('dbb_1x1_kxk')

    def forward(self, inputs):

        if hasattr(self, 'dbb_reparam'):
            return self.nonlinear(self.dbb_reparam(inputs))

        out = self.dbb_origin(inputs)
        if hasattr(self, 'dbb_1x1'):
            out += self.dbb_1x1(inputs)
        out += self.dbb_avg(inputs)
        out += self.dbb_1x1_kxk(inputs)
        return self.nonlinear(out)

    def init_gamma(self, gamma_value):
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, gamma_value)
        if hasattr(self, "dbb_1x1"):
            torch.nn.init.constant_(self.dbb_1x1.bn.weight, gamma_value)
        if hasattr(self, "dbb_avg"):
            torch.nn.init.constant_(self.dbb_avg.avgbn.weight, gamma_value)
        if hasattr(self, "dbb_1x1_kxk"):
            torch.nn.init.constant_(self.dbb_1x1_kxk.bn2.weight, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, 1.0)


if __name__ == "__main__":
    import time

    batch_Norm = False
    for l in range(100):
        net = nn.Sequential(DiverseBranchBlock(6, 6, 3, 1, 1, batch_Norm=batch_Norm, deploy=False),
                            DiverseBranchBlock(6, 6, 3, 2, 1, batch_Norm=batch_Norm, deploy=False),
                            DiverseBranchBlock(6, 6, 3, 2, 1, batch_Norm=batch_Norm, deploy=False),
                            DiverseBranchBlock(6, 6, 3, 1, 1, batch_Norm=batch_Norm, deploy=False)
                            ).cuda()
        # for module in net.modules():
        #     if isinstance(module, torch.nn.BatchNorm2d):
        #         nn.init.uniform_(module.running_mean, 0, 0.1)
        #         nn.init.uniform_(module.running_var, 0, 0.1)
        #         nn.init.uniform_(module.weight, 0, 0.1)
        #         nn.init.uniform_(module.bias, 0, 0.1)
        if l == 0:
            print(net)

        x = np.random.randint(0, 255, (1, 6, 80, 80)).astype(np.float32)/255.
        x = torch.cuda.FloatTensor(x)
        time1 = time.time()


        # train mode
        net = net.train()
        y_train = net(x)

        # eval mode
        net = net.eval()
        y_eval = net(x)


        time2 = time.time()
        # net = net.eval()

        # eval mode after rep
        for m in net.modules():
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()
        if l==0:
            print(net)
        time3 = time.time()
        y_rep = net(x)
        time4 = time.time()
        # print(time2 - time1, time4 - time3)
        print(((y_train - y_eval) ** 2).sum())
        print(((y_train - y_rep) ** 2).sum())
        print(((y_eval - y_rep) ** 2).sum())

        print("Is Match: ", np.allclose(y_rep.detach().cpu().numpy(), y_eval.detach().cpu().numpy(), atol=1e-5))
