import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torchvision.models as tv_models
import torch.nn.functional as F
from models.dynamic_conv import ParamAwareConv
import traceback
from utils.pyramid_fusion import ParamAwarePyramidFusion


class MultiModalTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config['d_model']
        num_classes = config['num_classes']
        nhead = config['nhead']
        param_dim = config.get('param_dim', 3)
        conv_group = config.get('conv_group', 8)

        # ---------------（1）ResNet50 分支 ---------------
        resnet = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
        # 冻结resnet层
        for name, p in resnet.named_parameters():
            # p.requires_grad = name.startswith("layer3")   # 只有 layer3 的参数需要梯度，其它都冻结
            if name.startswith("layer3") or name.startswith("layer4"):  # layer3和layer4的参数需要梯度，冻结layer1和layer2
                p.requires_grad = True
            else:
                p.requires_grad = False
        # 将 layer4 置为 Identity，只用resnet的layer1-3，如果要使用layer1-4则将下面两行语句注释掉
        # resnet.layer4 = nn.Identity()
        # resnet.fc = nn.Identity()  # 去掉最后全连接层
        self.cnn = resnet

        # ---------------（2）动态卷积层 ---------------
        # 降到1024 + DynConv 1024→self.d_model
        self.reduce_conv = nn.Conv2d(1024, 1024, kernel_size=1)
        self.dynamic_conv = ParamAwareConv(1024, 1024, param_dim, group=conv_group)
        # 动态卷积替换为恒等映射
        # self.dynamic_conv = nn.Identity()
        # ---------------（3）FPN 层 ---------------
        # ****** self.img_proj_direct 是对self.fpn的替代*******
        # ////只有layer3输出到fpn
        # self.fpn = nn.ModuleDict({
        #     'lateral': nn.Conv2d(1024, 256, kernel_size=1),
        #     'smooth': nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     'upsample': nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # })
        # //// resnet的layer1-4均输入到fpn
        # 1x1 lateral convs for layer1, layer2, layer3′, layer4
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(256, self.d_model, kernel_size=1),  # layer1 输出通道256
            nn.Conv2d(512, self.d_model, kernel_size=1),  # layer2 输出通道512
            nn.Conv2d(1024, self.d_model, kernel_size=1),  # dyn_conv 后的 layer3 通道 = self.d_model
            nn.Conv2d(2048, self.d_model, kernel_size=1)  # layer4 输出通道2048
        ])
        # 3x3 平滑 convs
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(self.d_model, self.d_model, kernel_size=3, padding=1)
            for _ in range(4)
        ])

        # 替代 FPN 后直接投影
        # self.img_proj_direct = nn.Sequential(
        #     nn.Linear(1024, self.d_model),
        #     nn.GELU(),
        #     nn.Linear(self.d_model, self.d_model)
        # )

        # ---------------（4）多尺度融合 ---------------
        # 4尺度拼接后维度 = 4 * d_model
        fused_dim = 4 * self.d_model
        # 特征增强层(fused_dim → 2*fused_dim → fused_dim)
        # self.feature_diversity = nn.Sequential(
        #     nn.Linear(fused_dim, fused_dim * 2), nn.GELU(), nn.Dropout(0.3),
        #     nn.Linear(fused_dim * 2, fused_dim)
        # )
        # # 特征投影层(fused_dim → 2*fused_dim → fused_dim)
        # self.feature_enhance = nn.Sequential(
        #     nn.Linear(fused_dim, fused_dim * 2), nn.GELU(), nn.LayerNorm(fused_dim * 2), nn.Dropout(0.2),
        #     nn.Linear(fused_dim * 2, fused_dim)
        # )
        # ParamAwarePyramidFusion模块集成
        self.pyramid_fusion = ParamAwarePyramidFusion(
            param_dim=config.get('param_dim', 3),
            C=self.d_model,  # d_model 为 P1–P4 全部 lateral 后的通道数
            hidden_dim=64,
            temperature=1.1,
            ent_lambda=config.get('ent_loss_weight', 0.05)
        )

        # ---------------（5）图像投影层 ---------------
        # 取 self.cnn(flat_images)→[B*S,2048]，再投影到 d_model
        # self.img_proj = nn.Linear(2048, self.d_model)
        # 取 FPN 后统一维度→1024，再投影到 d_model
        # self.img_proj_after_fpn = nn.Linear(fused_dim, self.d_model)
        # ParamAwarePyramidFusion模块集成
        self.img_proj_after_fpn = nn.Linear(self.d_model, self.d_model)
        # ---------------（6）电信号分支（LSTM + Attention） ---------------
        self.signal_encoder = nn.LSTM(
            input_size=3,
            hidden_size=32,
            bidirectional=True,
            batch_first=True
        )
        self.signal_attention = nn.Sequential(
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
        self.signal_proj = nn.Linear(64, self.d_model)

        # ---------------（7）参数分支 ---------------
        self.param_encoder = nn.Sequential(
            nn.Linear(param_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, self.d_model)
        )
        # MetaAdapter 这里只占位
        # self.param_adapter = MetaAdapter(input_dim=d_model, output_dim=d_model)

        # ---------------（8）多模态融合---------------
        # self.attention = InterpretableAttention(self.d_model)
        self.fusion_module = FusionModule(self.d_model)
        # ---------------（9）融合后投影层 ---------------
        # 因为实际拼接是 [img_proj, param_proj, elec_proj] 三者，所以维度是  d_model + d_model + d_model = 3*d_model
        # 这里用一层线性变回 d_model
        # self.fusion_proj = nn.Sequential(
        #     nn.Linear(3 * self.d_model, self.d_model),
        #     nn.LayerNorm(self.d_model)
        # )

        # ---------------（10）Transformer 编码器 ---------------
        # 回归用 3 层 TransformerEncoder
        reg_layer = TransformerEncoderLayer(
            d_model=self.d_model, nhead=nhead, dim_feedforward=2048, dropout=0.1, batch_first=True
        )
        self.reg_transformer = TransformerEncoder(reg_layer, num_layers=3)

        # 分类用 2 层 TransformerEncoder
        cls_layer = TransformerEncoderLayer(
            d_model=self.d_model, nhead=nhead, dim_feedforward=1024, dropout=0.1, batch_first=True
        )
        self.cls_transformer = TransformerEncoder(cls_layer, num_layers=2)
        # 分类特征归一化层 + Dropout --- #
        self.cls_norm = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Dropout(0.1),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU()
        )
        # ---------------（11）输出头 ---------------
        # 回归头：输出电压、电流（用 sigmoid/hardtanh 等做物理约束）
        self.voltage_head = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        self.current_head = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            # nn.Hardtanh(min_val=0, max_val=8)  # 约束电流范围 [0,8]A
            # nn.ReLU()  # 只取 >= 0
        )
        # self.power_head = nn.Sequential(
        #     nn.Linear(self.d_model, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 1),
        #     nn.ReLU()
        # )
        # 功率预测由电压和电流相乘得到
        self.power_head = nn.Identity()
        # 分类头
        self.cls_head = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),  # 可适当添加，防过拟合
            nn.Linear(128, num_classes)
        )

        # ---------------（12）Feature Selector 占位 ---------------
        # self.feature_selector = AdaptiveFeatureSelector(1024)

        # ---------------（13）初始化权重 ---------------
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                if hasattr(m, 'weight'):
                    nn.init.constant_(m.weight, 1.0)
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def feature_maps(self, x):
        """返回CNN最后一层卷积的特征图（已压缩通道）"""
        # 输入x形状: [B,S,C,H,W]
        B, S = x.shape[:2]
        x = x.view(-1, *x.shape[2:])  # [B*S,C,H,W]

        with torch.no_grad():
            # 获取layer4输出 [B*S,2048,7,7]
            x = self.cnn.conv1(x)
            x = self.cnn.bn1(x)
            x = self.cnn.relu(x)
            x = self.cnn.maxpool(x)
            x = self.cnn.layer1(x)
            x = self.cnn.layer2(x)
            x = self.cnn.layer3(x)
            x = self.cnn.layer4(x)  # 直接使用layer4

            # 关键修改：压缩通道维度并调整尺寸
            heatmap = x.mean(dim=1)  # [B*S,7,7]
            heatmap = F.interpolate(heatmap.unsqueeze(1),
                                    size=(224, 224),
                                    mode='bilinear').squeeze()  # [B*S,224,224]

        return heatmap.view(B, S, 224, 224)  # [B,S,224,224]

    def get_image_features(self, images):
        """返回原始图像特征和投影特征"""
        B, S = images.shape[:2]
        flat_images = images.view(-1, *images.shape[2:])  # [B*S,3,224,224]

        # 获取完整的CNN特征
        with torch.no_grad():
            features = self.cnn(flat_images)  # [B*S,2048]
            projected = self.img_proj(features)  # [B*S,d_model]

        return (
            features.view(B, S, -1),  # [B,S,2048]
            projected.view(B, S, -1)  # [B,S,d_model]
        )

    def get_fused_features(self):
        if hasattr(self.attention, 'attention_weights') and self.attention.attention_weights is not None:
            return self.attention.attention_weights
        return None

    def _get_fused_features(self, images, params, elec_signals):
        # 这是一个辅助方法，用于获取融合特征
        B, S = images.shape[0], images.shape[1]
        flat_images = images.view(-1, *images.shape[2:])

        # 简化的特征提取流程
        with torch.no_grad():
            x = self.cnn.conv1(flat_images)
            x = self.cnn.bn1(x)
            x = self.cnn.relu(x)
            x = self.cnn.maxpool(x)
            x = self.cnn.layer1(x)
            x = self.cnn.layer2(x)
            base_feat = self.cnn.layer3(x)  # [B*S, 1024, 14, 14]

        batch_params = params.unsqueeze(1).expand(-1, S, -1).reshape(-1, 3)
        dyn_feat = self.dynamic_conv(base_feat, batch_params)
        p3 = self.fpn['lateral'](dyn_feat)
        p3 = self.fpn['smooth'](p3)
        p4 = self.fpn['upsample'](p3)

        assert base_feat.size(1) == 1024, f"基础特征通道应为1024，实际为{base_feat.size(1)}"
        assert dyn_feat.size(1) == 1024, f"动态特征通道应为1024，实际为{dyn_feat.size(1)}"
        assert p3.size(1) == 256, f"FPN特征通道应为256，实际为{p3.size(1)}"

        img_feat = p4.mean(dim=[2, 3])
        img_feat = self.feature_diversity(img_feat)
        img_feat = F.normalize(img_feat, p=2, dim=1)
        img_feat = self.feature_enhance(img_feat)

        img_proj = self.img_proj_after_fpn(img_feat)
        elec_proj = torch.zeros(B * S, self.config['d_model'] // 8, device=images.device)
        param_proj = torch.zeros(B * S, self.config['d_model'], device=images.device)

        fused = torch.cat([img_proj, param_proj, elec_proj], dim=-1)
        fused = self.fusion_proj(fused)
        return fused.view(B, S, -1)

    def forward(self, images, params, elec_signals=None):
        """
        inputs:
            images: [B, S, 3, 224, 224]
            params: [B, param_dim]
            elec_signals: [B, S, 3]  (voltage, current, power 序列)
        returns:
            reg_output: [B, 3]  (voltage, current, power)
            cls_output: [B, num_classes]
            ortho_loss: torch.tensor  (占位 0)
        """
        B, S, C, H, W = images.shape
        device = images.device

        # -----------------（1）图像特征提取 -----------------
        flat_images = images.view(B * S, C, H, W)  # [B*S,3,224,224]
        # ResNet 前向，冻结 layer1~layer2，只微调 layer3、4

        # --- ResNet 前四层特征提取 ---
        # 先经过 先经过 ResNet 的 stem（conv1→bn1→relu→maxpool）
        x = flat_images  # [B*S, 3, H, W]
        x = self.cnn.conv1(x)  # → [B*S, 64, H/2, W/2]
        x = self.cnn.bn1(x)  # BatchNorm
        x = self.cnn.relu(x)  # ReLU
        x = self.cnn.maxpool(x)  # → [B*S, 64, H/4, W/4]
        feat1 = self.cnn.layer1(x)  # → [B*S, 256, H/4, W/4]
        feat2 = self.cnn.layer2(feat1)  # [B*S, 512, H/8, W/8]
        feat3 = self.cnn.layer3(feat2)  # [B*S, 1024, H/16, W/16]
        feat4 = self.cnn.layer4(feat3)  # [B*S, 2048, H/32, W/32]

        # -----------------（2） layer3降维（1024->1024） + DynConv -----------------
        reduced3 = self.reduce_conv(feat3)  # [B*S, self.d_model, H/16, W/16]
        dyn_feat3 = self.dynamic_conv(
            reduced3,
            params.unsqueeze(1).expand(-1, S, -1).reshape(-1, params.shape[-1])
        )  # [B*S, self.d_model, H/16, W/16]

        # “消融”版：只做降维，不做动态卷积
        # reduced3 = self.reduce_conv(feat3)  # [B*S, d_model, H/16, W/16]
        # dyn_feat3 = reduced3  # 直接用降维后的特征，不加参数感知

        # --- 四路特征列表 ---
        feats = [feat1, feat2, dyn_feat3, feat4]

        # 1. lateral: 统一通道到 d_model
        latents = [l_conv(f) for l_conv, f in zip(self.lateral_convs, feats)]
        # 2. top-down 聚合 & smooth
        fpn_outs = [None] * 4
        prev = None
        # 从最高层（feat4）往回遍历
        for i in reversed(range(4)):
            if prev is None:
                merged = latents[i]
            else:
                up = F.interpolate(prev, size=latents[i].shape[-2:], mode='bilinear', align_corners=False)
                merged = latents[i] + up
            fpn_outs[i] = self.smooth_convs[i](merged)
            prev = merged

        # -----------------（3）FPN 多尺度融合 -----------------
        # p3 = self.fpn['smooth'](self.fpn['lateral'](dyn_feat))  # [B*S,256,7,7]
        # p4 = self.fpn['upsample'](p3)  # [B*S,256,14,14]
        # p3f = F.adaptive_avg_pool2d(p3, (1, 1)).view(B * S, 256)
        # p4f = F.adaptive_avg_pool2d(p4, (1, 1)).view(B * S, 256)
        #
        # # -----------------（4）图像特征全局池化 + 多样化 + 增强 + 投影-----------------
        # img_feat = torch.cat([p3f, p4f], dim=1)  # [B*S,512]
        # img_feat = self.feature_diversity(img_feat)  # identity stub
        # img_feat = F.normalize(img_feat, p=2, dim=1)
        # img_feat = self.feature_enhance(img_feat)  # identity stub
        # img_proj_after = self.img_proj_after_fpn(img_feat)  # [B*S, d_model]

        # ****** 替换fpn，直接投影到同一维度输出*************
        # 2）全局平均池化
        # bsz_s, C, H, W = dyn_feat.shape
        # img_vec = F.adaptive_avg_pool2d(dyn_feat, 1).view(bsz_s, C)  # [B*S,1024]
        #
        # # 3）直接投影到 d_model
        # img_proj_after = self.img_proj_direct(img_vec)  # [B*S, d_model]
        # fpn_outs = [P1, P2, P3, P4]，分别对应四个尺度
        # 下面可以按原有流程，对 P3、P4 做全局池化、拼接、投影等
        # p3, p4 = fpn_outs[2], fpn_outs[3]
        # p3f = F.adaptive_avg_pool2d(p3, 1).view(B * S, -1)
        # p4f = F.adaptive_avg_pool2d(p4, 1).view(B * S, -1)
        # img_feat = torch.cat([p3f, p4f], dim=1)  # [B*S, 2*d_model]

        # —— 取 P1, P2, P3, P4 并全局池化、拼接 —— #
        pooled = []
        for p in fpn_outs:
            v = F.adaptive_avg_pool2d(p, 1).view(B * S, -1)  # [B*S, d_model]
            pooled.append(v)
        # # //// 简单拼接所有尺度：得到 [B*S, 4*d_model]
        # img_feat = torch.cat(pooled, dim=1)
        #
        # # —— 多尺度特征增强与投影 —— #
        # img_feat = self.feature_diversity(img_feat)     # [B*S, fused_dim]
        # img_feat = F.normalize(img_feat, dim=-1)
        # img_feat = self.feature_enhance(img_feat)  # [B*S, fused_dim]
        # img_proj_after = self.img_proj_after_fpn(img_feat)  # [B*S, d_model]

        # batch_params: [B*S, param_dim]
        batch_params = params.unsqueeze(1).expand(-1, S, -1).reshape(-1, params.shape[-1])
        # 参数感知金字塔融合
        fused_feat, ent_loss, alpha, gamma = self.pyramid_fusion(pooled, batch_params)
        # —— 数值保护：清理 NaN/Inf，并限幅 fused 特征 —— #
        fused_feat = torch.nan_to_num(fused_feat, nan=0.0, posinf=5.0, neginf=-5.0)
        fused_feat = torch.clamp(fused_feat, min=-5.0, max=5.0)
        # 投影到同一维度
        img_proj_after = self.img_proj_after_fpn(fused_feat)  # [B*S, d_model]
        # 将 ent_loss 存到 self 以便后续损失聚合
        self.latest_ent_loss = ent_loss
        self.latest_alpha = alpha
        self.latest_gamma = gamma

        # -----------------（5）电信号处理 -----------------
        if elec_signals is not None:
            eo, _ = self.signal_encoder(elec_signals)  # [B, S, 64]
            # 注意力权重：对每个时间步打分
            attn_scores = self.signal_attention(eo)  # [B, S, 1]
            attn_scores = attn_scores.squeeze(-1)  # [B, S]
            attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # [B, S, 1]
            # 加权求和得到电信号特征
            ef = torch.sum(eo * attn_weights, dim=1)  # [B, 64]
            elec_proj = self.signal_proj(ef)  # [B, d_model]
            elec_proj = elec_proj.unsqueeze(1).expand(-1, S, -1).reshape(B * S, -1)
        else:
            elec_proj = torch.zeros(B * S, self.d_model, device=device)

        # -----------------（6）参数处理 -----------------
        param_feat = self.param_encoder(params)  # [B, d_model]
        param_proj = param_feat.unsqueeze(1).expand(-1, S, -1).reshape(B * S, -1)  # [B*S, d_model]

        # -----------------（7）多模态融合 -----------------
        # img_proj_after: [B*S, d_model]; param_proj: [B*S, d_model]; elec_proj: [B*S, d_model]
        fused, gate_w, attn_w = self.fusion_module(img_proj_after, param_proj, elec_proj)  # [B*S, d_model]
        # ==== 【数值保护】：防止 NaN/Inf 蔓延到 Transformer ====
        fused = torch.where(torch.isfinite(fused), fused, torch.zeros_like(fused))
        fused = torch.clamp(fused, min=-5.0, max=5.0)
        fused_seq = fused.view(B, S, self.d_model)  # [B, S, d_model]

        # -----------------（8）Transformer 编码 -----------------
        reg_feat = self.reg_transformer(fused_seq).mean(dim=1)  # [B, d_model]
        cls_feat = self.cls_transformer(fused_seq).mean(dim=1)  # [B, d_model]
        # ==== 【分类特征数值保护】防止送到 LayerNorm 的输入含 NaN/Inf ====
        cls_feat = torch.where(torch.isfinite(cls_feat), cls_feat, torch.zeros_like(cls_feat))
        cls_feat = torch.clamp(cls_feat, min=-5.0, max=5.0)
        cls_feat = self.cls_norm(cls_feat)  # 分类归一化 + Dropout

        # -----------------（9）输出预测 -----------------
        voltage = self.voltage_head(reg_feat)  # [B, 1]
        current = self.current_head(reg_feat)  # [B, 1]
        # power = self.power_head(reg_feat)  # [B, 1]
        # power = voltage * current  # [B, 1]# 不使用功率预测头
        cls_output = self.cls_head(cls_feat)  # [B, num_classes]
        reg_out = torch.cat([voltage, current], dim=1)  # [B,2]
        # 不再 catch 所有异常：若跑到这一步，说明 dynamic_conv etc 都没出错

        # ===== 数值稳定处理 =====
        # 1) 替换 nan/inf 为零或极值
        # reg_out = torch.nan_to_num(reg_out, nan=0.0, posinf=3.0, neginf=-3.0)
        # 2) 限幅到 [-3, +3] 范围，防止极端值
        # reg_out = torch.clamp(reg_out, min=-3.0, max=3.0)

        return reg_out, cls_output, gate_w, attn_w


class MetaAdapter(nn.Module):
    """动态参数适配器，增强模型对不同参数的适应能力"""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(3, 64),  # 输入原始参数[load, pressure, speed]
            nn.ReLU(),
            nn.Linear(64, input_dim * 2)  # 生成scale和shift参数
        )
        self.output_dim = output_dim

    def forward(self, x, params):
        """
        x: 已编码的参数特征 [B, input_dim]
        params: 原始参数 [B, 3]
        """
        # 动态生成适配参数
        adapter_params = self.adapter(params)
        scale = adapter_params[:, :self.output_dim]
        shift = adapter_params[:, self.output_dim:]

        # 应用适配
        return x * scale + shift


class InterpretableAttention(nn.Module):
    """可解释的多模态注意力融合机制"""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # 初始化权重为有效值
        self.attention_weights = torch.zeros(1, 3)

        # 创建查询和键的投影层
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)

        # 值投影层（可选）
        self.value_proj = nn.Linear(d_model, d_model)

        # 缩放因子
        self.scale = torch.sqrt(torch.FloatTensor([d_model]))

    def forward(self, img_feat, param_feat, elec_feat):
        """
        输入:
            img_feat: [B*S, d_model] 图像特征
            param_feat: [B*S, d_model] 参数特征
            elec_feat: [B*S, d_model] 电信号特征
        输出:
            fused: [B*S, d_model] 融合特征
        """
        # 确保缩放因子在正确设备上
        self.scale = self.scale.to(img_feat.device)

        # 准备模态特征
        modalities = torch.stack([img_feat, param_feat, elec_feat], dim=1)  # [B*S, 3, d_model]

        # 生成查询向量（以图像特征为主）
        queries = self.query_proj(img_feat).unsqueeze(1)  # [B*S, 1, d_model]

        # 生成键向量
        keys = self.key_proj(modalities)  # [B*S, 3, d_model]

        # 计算注意力分数
        attn_scores = torch.matmul(queries, keys.transpose(1, 2))  # [B*S, 1, 3]
        attn_scores = attn_scores / self.scale

        # 应用softmax获取注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B*S, 1, 3]
        if torch.isnan(attn_weights).any():
            print("注意力权重出现NaN，使用安全替代")
            attn_weights = torch.softmax(torch.rand(1, 3), dim=-1)

        # 保存可视化权重
        self.attention_weights = attn_weights.squeeze(1).detach()  # [B*S, 3]

        # 生成值向量
        values = self.value_proj(modalities) if hasattr(self, 'value_proj') else modalities

        # 加权求和
        fused = torch.matmul(attn_weights, values)  # [B*S, 1, d_model]

        return fused.squeeze(1)  # [B*S, d_model]


class FusionModule(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.qkv_proj = nn.Linear(3 * d_model, 3 * d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead,
                                          dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(3 * d_model, 3),  # 为每路特征生成 gate logit
            nn.Softmax(dim=-1)
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, img, param, elec):
        # img/param/elec: [N, d_model]
        cat = torch.cat([img, param, elec], dim=-1)  # [N, 3*d_model]
        # Gate 权重
        gate_w = self.gate(cat)  # [N,3]
        fused = img * gate_w[:, 0:1] + param * gate_w[:, 1:2] + elec * gate_w[:, 2:3]
        # 可选：用自注意力增强
        # 转为序列长度=1的 QKV
        qkv = self.qkv_proj(cat)  # [N,3*d_model]
        Q, K, V = qkv.split(self.d_model, dim=-1)
        # 以 N×1 seq 形式调用 MHA
        out_attn, attn_w = self.attn(Q.unsqueeze(1),
                                     torch.stack([K, K, K], dim=1),
                                     torch.stack([V, V, V], dim=1))
        out_attn = out_attn.squeeze(1)  # [N,d_model]
        # 最终融合
        out = self.out_norm(fused + self.dropout(out_attn))
        return out, gate_w, attn_w  # 返回融合特征、gate、attn 矩阵


class FusionGate(nn.Module):  # 多模态注意力门控
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, img_feat, param_feat, elec_feat):
        combined = torch.cat([img_feat, param_feat, elec_feat], dim=-1)
        gate_weights = self.gate(combined)
        return (gate_weights * img_feat +
                (1 - gate_weights) * (param_feat + elec_feat) / 2)


class TemporalConv(nn.Module):  # 时序感知卷积
    def __init__(self, in_c, out_c, k=3):
        super().__init__()
        self.conv = nn.Conv1d(in_c, out_c, kernel_size=k, padding=k // 2)

    def forward(self, x):
        # x: [B, Seq, C]
        x = x.permute(0, 2, 1)  # [B, C, Seq]
        x = F.relu(self.conv(x))
        return x.permute(0, 2, 1)  # 恢复维度


class AdaptiveFeatureSelector(nn.Module):
    """动态选择最相关特征"""

    def __init__(self, in_channels):
        super().__init__()
        self.selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.selector(x)
        return x * attn


class ModelDiagnostics:
    """模型诊断工具"""

    @staticmethod
    def check_gradients(model):
        """检查梯度流"""
        zero_grad_count = 0
        total_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                total_params += 1
                if param.grad is None:
                    print(f"参数 {name} 梯度为None")
                    zero_grad_count += 1
                elif torch.allclose(param.grad, torch.zeros_like(param.grad)):
                    print(f"参数 {name} 梯度全零")
                    zero_grad_count += 1

        print(f"零梯度参数比例: {zero_grad_count}/{total_params} ({zero_grad_count / total_params:.1%})")

    @staticmethod
    def check_outputs(outputs, labels):
        """检查分类输出"""
        pred_probs = F.softmax(outputs, dim=1)
        max_probs, pred_labels = torch.max(pred_probs, dim=1)

        print(f"分类输出统计:")
        print(f"  最大概率: {max_probs.min().item():.4f}-{max_probs.max().item():.4f}")
        print(f"  预测标签分布: {torch.bincount(pred_labels)}")
        print(f"  真实标签分布: {torch.bincount(labels)}")
        print(f"  准确率: {(pred_labels == labels).float().mean().item():.2%}")


class FeatureExtractor(nn.Module):
    """稳健的特征提取器"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 使用轻量级CNN
        self.cnn = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        self.cnn.fc = nn.Identity()

        # 特征适配层
        self.feature_adapter = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )

    def forward(self, images):
        """输入: [B, S, C, H, W] 输出: [B, S, 256]"""
        B, S, C, H, W = images.shape
        flat_images = images.view(B * S, C, H, W)

        # 提取特征
        with torch.no_grad():
            features = self.cnn(flat_images)

        # 适配维度
        features = self.feature_adapter(features)
        return features.view(B, S, -1)
