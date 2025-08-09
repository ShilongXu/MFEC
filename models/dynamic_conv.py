import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class DynamicConvGenerator(nn.Module):
    """生成参数感知的动态卷积核"""

    def __init__(self, in_channels, out_channels, param_dim=3, group=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group = group
        self.out_c_pg = out_channels // group
        self.in_c_pg = in_channels // group
        assert out_channels % group == 0 and in_channels % group == 0

        # 每个组每个 3×3 核的权重数量：(out_c_pg × in_c_pg × 3 × 3)
        params_per_kernel = self.out_c_pg * self.in_c_pg * 3 * 3
        total_params = group * params_per_kernel  # 单个样本下，所有 group 的核共需多少参数

        print(f"初始化动态卷积: in={in_channels}, out={out_channels}, group={group}")
        print(f"每核参数: {params_per_kernel}, 总参数: {total_params}")

        # 定义 MLP，输入 param_dim → 输出 total_params
        self.mlp = nn.Sequential(
            nn.Linear(param_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, total_params)
        )

    def forward(self, params):
        """
        params: [B, param_dim]
        返回： [B, group, out_c_pg, in_c_pg, 3, 3]
        """
        B = params.size(0)
        # raw_kernels: [B, total_params]
        raw_kernels = self.mlp(params)

        # reshape 为 [B, group, out_c_pg, in_c_pg, 3, 3]
        return raw_kernels.view(
            B,
            self.group,
            self.out_c_pg,
            self.in_c_pg,
            3, 3
        )


    def visualize_kernels(self, params, save_path):
        """生成带对比的动态卷积核可视化"""
        with torch.no_grad():
            kernels = self.forward(params)

            # 创建对比布局
            fig = plt.figure(figsize=(15, 10))
            gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.05])

            # 显示原始参数核
            ax1 = fig.add_subplot(gs[0, 0])
            k1 = kernels[0].mean(dim=0).cpu().numpy()
            im1 = ax1.imshow(k1, cmap='viridis')
            ax1.set_title('Default Parameters')
            ax1.axis('off')

            # 显示真实参数核
            ax2 = fig.add_subplot(gs[0, 1])
            k2 = kernels[-1].mean(dim=0).cpu().numpy()
            im2 = ax2.imshow(k2, cmap='viridis')
            ax2.set_title('Real Parameters')
            ax2.axis('off')

            # 显示差异
            ax3 = fig.add_subplot(gs[1, :])
            diff = np.abs(k1 - k2)
            im3 = ax3.imshow(diff, cmap='hot')
            ax3.set_title('Absolute Difference')
            ax3.axis('off')

            # 添加色条
            cax = fig.add_subplot(gs[:, 2])
            fig.colorbar(im3, cax=cax)

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()


class ParamAwareConv(nn.Module):
    """
    基于 PyTorch 分组卷积的一次性并行实现 param-aware 卷积。
    假定：输入 x: [B, C_in, H, W],  params: [B, param_dim]
    输出： [B, C_out, H, W]
    """

    def __init__(self, in_channels, out_channels, param_dim=3, group=8):
        super().__init__()
        assert in_channels % group == 0 and out_channels % group == 0, \
            f"in_channels={in_channels}, out_channels={out_channels} must be divisible by group={group}"
        self.group = group
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_c_pg = in_channels // group
        self.out_c_pg = out_channels // group

        self.generator = DynamicConvGenerator(
            in_channels=in_channels,
            out_channels=out_channels,
            param_dim=param_dim,
            group=group
        )

    def forward(self, x, params):
        """
        x: [B, C_in, H, W]
        params: [B, param_dim]
        """
        B, C_in, H, W = x.size()
        G = self.group
        # =======1.由 generator 生成 kernel: [B, G, out_c_pg, in_c_pg, 3,3]
        kernels = self.generator(params)

        # =======2.动态卷积-下面的1）和2）选一个即可
        # 1）并行卷积，提高GPU 利用率
        weight = kernels.view(B * G * self.out_c_pg, self.in_c_pg, 3, 3)
        # 将 x reshape 为 [1, B*C_in, H, W] 并用 groups=B*G
        x_ = x.view(1, B * C_in, H, W)
        out = F.conv2d(x_, weight=weight, groups=B * G, padding=1)
        out = out.view(B, self.out_channels, H, W)
        # 2）循环卷积，防止显存爆炸
        # outs = []
        # for i in range(B):  # 按样本循环的动态卷积
        #     # 取第 i 个样本的 kernel
        #     k = kernels[i]  # [G, out_c_pg, in_c_pg, 3,3]
        #     # reshape 为标准分组卷积 weight
        #     weight = k.view(self.group * self.out_c_pg, self.in_c_pg, 3, 3)
        #     xi = x[i:i + 1]  # [1, C_in, H, W]
        #     # 标准分组卷积
        #     outi = F.conv2d(xi, weight=weight, bias=None,
        #                     stride=1, padding=1, groups=self.group)
        #     outs.append(outi)
        # return torch.cat(outs, dim=0)  # [B, out_channels, H, W]

        return out
