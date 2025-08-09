import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, WeightedRandomSampler, BatchSampler, SubsetRandomSampler
from data.terrain_dataset import TerrainEnergyDataset
from models.multimodal_transformer import MultiModalTransformer, ModelDiagnostics
from utils.Enhanced_early_stopper import EnhancedEarlyStopper
from config.defaults import CONFIG
from utils.metrics import calculate_metrics
from utils.logger import TrainingLogger
from tqdm import tqdm
import torch.optim as optim
import os
from sklearn.manifold import TSNE
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from visualization.visualize import plot_training_curves, visualize_feature_space, visualize_features, \
    plot_loss_components, plot_physical_constraints, generate_parameter_card, visualize_dynamic_kernels
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score
import traceback
from sklearn.metrics import confusion_matrix, precision_score, recall_score

import time
from collections import defaultdict


def get_train_transforms():
    """训练数据增强配置"""

    return A.Compose([
        # 1. 裁剪掉上1/4区域，只保留下方的3/4区域
        A.Crop(x_min=0, y_min=120, x_max=640, y_max=480),
        # 2. 常规增强（仅地面区域）
        A.HorizontalFlip(p=0.5),  # 平翻转图像
        A.RandomRotate90(p=0.5),  # 随机旋转 90 度的倍数（0°、90°、180°、270°）
        A.RandomBrightnessContrast(p=0.2),  # 亮度和对比度
        # 去掉或大幅降低以下两项，因为它们影响纹理
        # A.GaussianBlur(blur_limit=(3,7), p=0.1),#高斯模糊
        # A.CoarseDropout(max_holes=4, max_height=16, max_width=16, fill_value=0, p=0.2),#创建矩形遮挡区域
        # 3. 统一尺寸
        # A.Resize(256, 256),  # 调整为 256x256 像素大小
        A.RandomCrop(224, 224),  # 随机裁剪出 224x224 像素的区域
        # 4. 标准化
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
        ToTensorV2()  # 将图像和标签转换为 PyTorch 的张量格式（Tensor）
    ])


def get_val_transforms():
    """验证集数据转换（无数据增强）"""
    return A.Compose([
        A.Crop(x_min=0, y_min=120, x_max=640, y_max=480),
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


class EnergyPredictor:
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.visualize_freq = 5  # 每5个epoch可视化一次
        self.dyn_loss_weight = nn.Parameter(torch.tensor(0.1))  # 可学习权重
        # 初始化模型
        self.model = MultiModalTransformer(config).to(self.device)
        # 添加模型训练状态标志
        self.model.train()  # 初始设置为训练模式
        os.makedirs(self.config['checkpoint_path'], exist_ok=True)
        try:
            # 初始化训练集
            self.train_dataset = TerrainEnergyDataset(
                root_dir=config['train_root'],
                mode='train',
                seq_length=config['seq_length'],
                transform=get_train_transforms()
            )

            # 从训练数据集获取标准化器引用
            self.voltage_scaler = self.train_dataset.voltage_scaler
            self.current_scaler = self.train_dataset.current_scaler
            self.power_scaler = self.train_dataset.power_scaler
            self.param_scaler = self.train_dataset.param_scaler

            # 打印验证信息
            print("\n训练器标准化器验证:")
            print(f"电压标准化器: mean={self.voltage_scaler.mean_[0]:.2f}, scale={self.voltage_scaler.scale_[0]:.2f}")
            print(f"电流标准化器: mean={self.current_scaler.mean_[0]:.2f}, scale={self.current_scaler.scale_[0]:.2f}")

            # 从训练集获取标准化器
            scaler_dict = {
                'voltage': self.voltage_scaler,
                'current': self.current_scaler,
                'power': self.power_scaler,
                'params': self.param_scaler
            }

            # 初始化验证集（使用训练集的标准化器）
            self.val_dataset = TerrainEnergyDataset(
                root_dir=config['val_root'],
                mode='val',
                seq_length=config['seq_length'],
                transform=get_val_transforms(),
                scaler_dict=scaler_dict
            )

            print(f"训练集大小: {len(self.train_dataset)} 样本")
            print(f"验证集大小: {len(self.val_dataset)} 样本")

            # 自动根据 DynamicLabelEncoder 中的 classes_ 生成 class_counts
            # train_dataset.samples 中每个 sample 是 dict，ground_label 存了编码标签
            labels = [s['ground_label'].item() for s in self.train_dataset.samples]
            num_classes = len(self.train_dataset.ground_encoder.classes_)  # 当前类别数
            class_counts = np.bincount(labels, minlength=num_classes)
            class_weights = 1.0 / (class_counts + 1e-6)  # 防止除零
            sample_weights = [class_weights[l] for l in labels]

            # WeightedRandomSampler 保证每个 batch 里各类别均衡出现
            total_bs = config['batch_size']  # 仍表示「整个 batch 要多少样本」

            # 每类样本数 K = floor(total_bs / num_classes)
            K = total_bs // num_classes
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )

            # 将标准化参数转换为PyTorch张量
            self.voltage_mean = torch.tensor(self.voltage_scaler.mean_, dtype=torch.float32, device=self.device)
            self.voltage_std = torch.tensor(self.voltage_scaler.scale_, dtype=torch.float32, device=self.device)
            self.current_mean = torch.tensor(self.current_scaler.mean_, dtype=torch.float32, device=self.device)
            self.current_std = torch.tensor(self.current_scaler.scale_, dtype=torch.float32, device=self.device)
            self.power_mean = torch.tensor(self.power_scaler.mean_, dtype=torch.float32, device=self.device)
            self.power_std = torch.tensor(self.power_scaler.scale_, dtype=torch.float32, device=self.device)

        except Exception as e:
            print(f"数据集初始化失败: {str(e)}")
            traceback.print_exc()
            raise

        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        self.num_train_samples = len(self.train_loader.dataset)  # 数据集总样本数

        # （可选）检查一下冻结是否正确：
        num_frozen = sum(1 for _, p in self.model.named_parameters() if not p.requires_grad)
        num_total = sum(1 for _ in self.model.named_parameters())
        print(f"冻结参数 {num_frozen}/{num_total}； 解冻参数 {(num_total - num_frozen)}/{num_total}")

        # 实例化 early stopper
        self.early_stopper = EnhancedEarlyStopper(
            patience=config['early_stop_patience'],
            min_epochs=config['min_epochs'],
            improvement=config['improvement_threshold'],
            stop_metric='composite',  # 用 composite 指标
            higher_is_better=True  # composite 越大越好
        )

        # 检查模型参数初始化
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='leaky_relu')
            elif 'bias' in name:
                nn.init.constant_(param, 0.01)  # 小正值避免死神经元
        print("模型参数初始化完成")

        # 物理约束损失
        self.phy_criterion = nn.MSELoss()

        # 损失函数
        self.reg_criterion = nn.HuberLoss()
        all_labels = np.concatenate([b['ground_label'].numpy() for b in self.train_loader])
        counts = np.bincount(all_labels, minlength=config['num_classes'])
        total = counts.sum()
        class_weights = torch.tensor(total / (config['num_classes'] * counts),
                                     dtype=torch.float32).to(self.device)
        print("Class weights:", class_weights.tolist())
        self.cls_criterion = nn.CrossEntropyLoss(weight=class_weights)

        # ======分组设置学习率======
        # 1. 设置学习率
        lr_main = config['lr']
        lr_cls = config['lr'] * 10.0
        lr_min = config['min_lr']
        lr_current = config['lr'] * 2
        # 2. 分组参数列表
        # 1) layer3 + layer4 微调组（lr 最小）
        params_layer3_4 = list(self.model.cnn.layer3.parameters()) \
                          + list(self.model.cnn.layer4.parameters())
        # 2) current_head 微调组（lr_current）
        params_current = list(self.model.current_head.parameters())
        # 3) cls_head 独立组（lr_cls）
        params_cls = list(self.model.cls_head.parameters())
        # 4) rest：所有其它 requires_grad=True 的参数
        all_trainable = [p for p in self.model.parameters() if p.requires_grad]
        for group in (params_layer3_4, params_current, params_cls):
            new_list = []
            for p in all_trainable:
                if not any(p is q for q in group):
                    new_list.append(p)
            all_trainable = new_list
        params_rest = all_trainable

        # 验证参数数目一致
        total1 = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total2 = sum(p.numel() for g in [params_layer3_4, params_rest, params_current, params_cls] for p in g)
        assert total1 == total2, f"参数不匹配: 模型{total1} vs 分组{total2}"

        # 3. 构造 param_groups
        param_groups = [
            {'params': params_layer3_4, 'lr': lr_min},
            {'params': params_rest, 'lr': lr_main},
            {'params': params_current, 'lr': lr_current},
            {'params': params_cls, 'lr': lr_cls},
        ]

        # 4. 创建优化器
        self.optimizer = optim.AdamW(param_groups, weight_decay=config.get('weight_decay', 1e-4))

        self.scheduler = optim.lr_scheduler.OneCycleLR(  # 自适应学习率调度
            self.optimizer,
            max_lr=config['lr'] * 3,
            epochs=config['epochs'],
            steps_per_epoch=len(self.train_loader),
            anneal_strategy='cos'
        )
        self.writer = SummaryWriter(log_dir='runs/experiment1')
        # self.power_loss_weight = nn.Parameter(torch.tensor(config.get('init_power_weight', 0.3)))
        self.early_stop_metric = config.get('early_stop_metric', 'total')  # 默认基于总损失
        self.early_stop_patience = config.get('early_stop_patience', 5)  # 默认耐心值
        self.best_metrics = None

        # 混合精度
        self.scaler = GradScaler()

        # 集成可视化
        self.visualization_dir = "visualization_results"
        os.makedirs(self.visualization_dir, exist_ok=True)
        assert hasattr(self.model, 'voltage_head'), "Model missing voltage_head"
        assert hasattr(self.model, 'current_head'), "Model missing current_head"
        assert hasattr(self.model, 'cls_head'), "Model missing cls_head"

        # 日志
        self.logger = TrainingLogger(log_path="./training_log.csv")

    def train(self):
        # 初始化检查
        # print("模型参数检查:")
        # total_params = 0
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(f"{name}: {param.shape}")
        #         total_params += param.numel()
        # print(f"总可训练参数: {total_params:,}")

        # 验证数据加载器
        sample_batch = next(iter(self.train_loader))
        print("\n数据批次验证:")
        print(f"图像: {sample_batch['images'].shape}")
        print(f"参数: {sample_batch['params'].shape}")
        print(f"电压: {sample_batch['voltage'].shape}")
        print(f"标签: {sample_batch['ground_label'].shape}")

        last_epoch, last_val_metrics = None, None
        # —— 开启 Anomaly Detection —— #
        # torch.autograd.set_detect_anomaly(True)
        for epoch in range(self.config['epochs']):
            # 重置峰值统计
            torch.cuda.reset_peak_memory_stats()
            print(f"[MEM] start epoch {epoch + 1} reserved={torch.cuda.memory_reserved() / 1e9:.2f}GB")

            self.model.train()  # 确保模型处于训练模式
            epoch_metrics = {k: 0.0 for k in ['total', 'voltage', 'current', 'power', 'cls', 'phy']}
            grad_monitor = {'cnn': [], 'dynamic_conv': [], 'voltage_head': [], 'current_head': []}
            gate_means, attn_means = [], []

            # ===== 训练阶段：计时开始 =====
            start_time = time.perf_counter()
            train_metrics = defaultdict(float)
            for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")):
                images = batch['images'].to(self.device)  # [B, S, 3,224,224]
                params = batch['params'].to(self.device)  # [B, 3]
                # 取序列最后一帧作为回归目标
                voltage_true = batch['voltage'][:, -1].unsqueeze(1).to(self.device)  # [B,1]
                current_true = batch['current'][:, -1].unsqueeze(1).to(self.device)  # [B,1]
                labels = batch['ground_label'].to(self.device)  # [B]
                elec_signals = batch['elec_signals'].to(self.device)  # [B, S, 3]

                # —— AMP 前向 + 反向 （Forward + backward）—— #
                self.optimizer.zero_grad()
                with autocast():
                    reg_pred, cls_pred, gate_w, attn_w = self.model(images, params, elec_signals)
                    # ===== 分类输出数值稳定处理 =====
                    print(
                        f"[DEBUG] reg_pred range: V={reg_pred[:, 0].min().item():.4f}/{reg_pred[:, 0].max().item():.4f}, "
                        f"I={reg_pred[:, 1].min().item():.4f}/{reg_pred[:, 1].max().item():.4f}")
                    # 替换 nan/inf
                    cls_pred = torch.where(torch.isfinite(cls_pred), cls_pred, torch.zeros_like(cls_pred))
                    # 限幅到 [-20, +20]
                    cls_pred = torch.clamp(cls_pred, min=-20.0, max=20.0)
                    losses = self._compute_loss(
                        (reg_pred, cls_pred),
                        (voltage_true, current_true, labels),
                        params
                    )
                print(f"[DEBUG LOSS] v={losses['voltage']:.6f}, c={losses['current']:.6f}, "
                      f"p={losses['power']:.6f}, cls={losses['cls']:.6f}, phy={losses['phy']:.6f}, "
                      f"ent={losses.get('ent', 0):.6f}")
                if torch.isnan(losses['cls']) or torch.isnan(losses['phy']):
                    print(">>> Loss components contain NaN before total")
                # —— 数值保护：如果 ent_loss 本身有 NaN，直接重置——#
                if not torch.isfinite(losses['ent']):
                    losses['ent'] = torch.tensor(0.0, device=losses['ent'].device)
                loss = losses['total']
                if not torch.isfinite(loss):
                    raise ValueError(f"Invalid loss {loss}")
                # 缩放梯度并反向
                self.scaler.scale(loss).backward()
                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('grad_clip', 1.0))
                # 优化器步进
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # 学习率调度（放在 optimizer.step 之后）
                self.scheduler.step()
                # 收集 gate/attn：
                if batch_idx % 20 == 0:  # 每20次取一个结果
                    gate_means.append(gate_w.detach().cpu().mean().item())
                    attn_means.append(attn_w.detach().cpu().mean().item())
                del gate_w, attn_w
                # 记录梯度（仅第一 batch）
                if batch_idx == 0:
                    print(f"\n首批次数据验证:")
                    print(f"电压范围: {voltage_true.min().item():.2f}~{voltage_true.max().item():.2f}")
                    print(f"电流范围: {current_true.min().item():.2f}~{current_true.max().item():.2f}")
                    print(f"标签分布: {torch.bincount(labels)}")

                    ModelDiagnostics.check_outputs(cls_pred, labels)  # 第一个 batch 时输出分类诊断
                    # —— debug 电流分支 —— #
                    with torch.no_grad():
                        v0, c0 = reg_pred[0, 0].item(), reg_pred[0, 1].item()
                        tv0, tc0 = voltage_true[0, 0].item(), current_true[0, 0].item()
                    print(
                        f"[DEBUG] sample0 → v_pred={v0:.4f}, v_true={tv0:.4f}; c_pred={c0:.4f}, c_true={tc0:.4f}")
                    for n, p in self.model.current_head.named_parameters():
                        print(f"[GRAD-CHECK] {n}: grad_norm={p.grad.norm().item():.4e}")

                    ModelDiagnostics.check_gradients(self.model)
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            if 'cnn' in name:
                                grad_monitor['cnn'].append(param.grad.norm().item())
                            elif 'dynamic_conv' in name:
                                grad_monitor['dynamic_conv'].append(param.grad.norm().item())
                            elif 'voltage_head' in name:
                                grad_monitor['voltage_head'].append(param.grad.norm().item())
                            elif 'current_head' in name:
                                grad_monitor['current_head'].append(param.grad.norm().item())

                # 累加损失
                for k in epoch_metrics:
                    if k in losses:
                        epoch_metrics[k] += losses[k].item()

                # 收集每个 batch 的 gate/attn 平均
                # gate_means.append(gate_w.detach().cpu().mean().item())
                # attn_means.append(attn_w.detach().cpu().mean().item())
                # # （2）删掉所有大张量Tensor
                # del reg_pred, cls_pred, gate_w, attn_w, losses, loss
                # del images, params, voltage_true, current_true, labels, elec_signals

            num_batches = len(self.train_loader)
            # ===== 2. 训练阶段：计时结束，计算 FPS =====
            elapsed = time.perf_counter() - start_time
            # FPS 定义为每秒处理多少 样本（或帧）
            fps = self.num_train_samples / elapsed

            # 训练结束后，对 epoch_metrics 做平均
            num_train_batches = len(self.train_loader)
            for k in epoch_metrics:
                epoch_metrics[k] /= num_train_batches

            # 打印梯度信息
            print(f"\n梯度监控 (Epoch {epoch + 1}):")
            for module, grads in grad_monitor.items():
                if grads:
                    print(f"{module}: mean_grad={sum(grads) / len(grads):.4f}")
            print(f"Loss components (Epoch {epoch + 1}): "
                  + f"V={epoch_metrics['voltage']:.4f}, "
                  + f"I={epoch_metrics['current']:.4f}, "
                  + f"Cls={epoch_metrics['cls']:.4f},"
                  + f"Phy={epoch_metrics['phy']:.4f}")

            # 验证集评估
            # 验证前切换到 eval，锁定 BN / Dropout
            val_metrics = self.evaluate(epoch)

            # 验证后切换回 train
            self.model.train()
            # 1) 更新 EarlyStopper（记录是否 improved）
            last_epoch, last_val_metrics = epoch, val_metrics

            if val_metrics:
                improved = self.early_stopper.update(epoch, val_metrics)
            else:
                improved = False
            # 日志记录：传入 num_train_batches 以便平均
            # batch 循环结束后，计算 epoch 平均
            extras = {'gate_mean': sum(gate_means) / len(gate_means), 'attn_mean': sum(attn_means) / len(attn_means),
                      'fps': fps}
            if val_metrics:
                self.logger.log(epoch, epoch_metrics, val_metrics, num_train_batches, extras=extras)

            # 2) 如果 improved，则保存“最佳模型”
            if improved:
                best_path = os.path.join(self.config['checkpoint_path'], "best_model.pth")
                self._save_checkpoint(epoch, val_metrics, suffix="best")
                print(f"⊙ New best {self.early_stopper.stop_metric} "
                      f"{self.early_stopper.best_score:.4f} @ epoch {epoch + 1}, saved → {best_path}")
            # 3) 如果达到 patience，则停止
            if self.early_stopper.should_stop():
                print(f"✱ Early stopping at epoch {epoch + 1}. "
                      f"No improvement in {self.early_stopper.stop_metric} for "
                      f"{self.early_stopper.patience} epochs. "
                      f"Best {self.early_stopper.stop_metric}={self.early_stopper.best_score:.4f} "
                      f"at epoch {self.early_stopper.best_epoch + 1}")
                break
            # epoch 末尾统一清理显存
            torch.cuda.empty_cache()
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"[MEM] end epoch {epoch + 1} peak_alloc={peak:.2f}GB\n")

            # 安全执行特征可视化
            # try:
            #     # 特征可视化
            #     if (epoch + 1) % self.visualize_freq == 0:
            #         self._visualize_features(epoch)
            # except Exception as viz_err:
            #     print(f"特征可视化异常: {str(viz_err)}")
            #     traceback.print_exc()
            # # 功率预测可视化
            # if (epoch + 1) % self.visualize_freq == 0:
            #     self._visualize_power_predictions(epoch)
            # # 绘制损失分量变化曲线
            # if (epoch + 1) % 5 == 0:  # 每5个epoch
            #     self._plot_loss_components(epoch)
            #     self._visualize_dynamic_kernels()  # 扩展参数组合
            # # 检查功率物理一致性
            # if (epoch + 1) % 10 == 0:  # 每10个epoch
            #     self._plot_physical_constraints(epoch)
        #
        # except Exception as err:
        #     e = err  # 捕获异常
        #     print(f"训练过程中发生异常: {str(e)}")
        #     traceback.print_exc()
        #     # 尝试保存当前状态
        #     try:
        #         self._save_checkpoint(0, {"error": str(e)}, suffix="emergency")
        #         print("紧急检查点已保存")
        #     except:
        #         print("无法保存紧急检查点")
        # finally:
        #     plt.close('all')
        #     self.writer.close()
        #     # 只在没有异常时执行最终可视化
        #     if e is None:
        #         try:
        #             self.final_visualization()
        #         except Exception as viz_err:
        #             print(f"最终可视化失败: {str(viz_err)}")
        #     else:
        #         print("由于训练异常，跳过最终可视化")

        # 训练结束保存一份最终模型
        final_path = os.path.join(self.config['checkpoint_path'], "final_model.pth")
        if last_epoch is not None and last_val_metrics is not None:
            self._save_checkpoint(last_epoch, last_val_metrics, suffix="final")
            print(f"➤ Final model (epoch {last_epoch + 1}) saved to {final_path}")
        else:
            print("⚠️ 没有可保存的 final checkpoint（训练循环可能未执行任何一次）")

    def _ensure_grad(self, loss):
        """确保损失有梯度信息"""
        if not loss.requires_grad:
            # 创建新变量并设置需要梯度
            loss = loss.clone().detach().requires_grad_(True)

        # 检查梯度函数是否存在
        if loss.grad_fn is None:
            # 如果是标量，创建计算图
            if loss.dim() == 0:
                loss = loss + 0.0 * torch.sum(torch.cat([p.view(-1) for p in self.model.parameters()]))
            else:
                loss = loss.mean()

        return loss

    def evaluate(self, epoch):
        self.model.eval()  # 设置为评估模式
        metrics = {
            'total': 0.0, 'reg': 0.0, 'cls': 0.0,
            'voltage_mae': 0.0, 'current_mae': 0.0, 'power_mae': 0.0,
            'voltage_mse_sum': 0.0, 'current_mse_sum': 0.0, 'power_mse_sum': 0.0,  # 用于累加每个样本的平方误差
            'voltage_rmse': 0.0, 'current_rmse': 0.0, 'power_rmse': 0.0,  # 最终再根据 mse_sum 计算 rmse
            'acc': 0.0, 'f1': 0.0,
            'voltage_rel': 0.0, 'current_rel': 0.0,  # 确保初始化所有键
            'power_consistency_mae': 0.0,
            'correct': 0,  # 新增正确计数
            'total_samples': 0  # 新增总样本计数
        }

        total_samples = 0
        all_true, all_pred = [], []
        # ——【确保不创建新梯度图】——
        with torch.no_grad():
            for batch in self.val_loader:
                # 数据准备
                images = batch['images'].to(self.device)
                params = batch['params'].to(self.device)
                voltage = batch['voltage'][:, -1].unsqueeze(1).to(self.device)
                current = batch['current'][:, -1].unsqueeze(1).to(self.device)
                labels = batch['ground_label'].to(self.device)
                elec_signals = batch['elec_signals'].to(self.device)

                # 前向传播
                reg_pred, cls_pred, gate_w, attn_w = self.model(images, params, elec_signals)
                # 反标准化到物理量
                v_pred_phy = reg_pred[:, 0].unsqueeze(1) * self.voltage_std + self.voltage_mean
                c_pred_phy = reg_pred[:, 1].unsqueeze(1) * self.current_std + self.current_mean
                c_pred_phy = F.relu(c_pred_phy)
                p_pred_phy = v_pred_phy * c_pred_phy
                v_true_phy = voltage * self.voltage_std + self.voltage_mean
                c_true_phy = current * self.current_std + self.current_mean
                p_true_phy = v_true_phy * c_true_phy  # [W]

                # 原有 MAE 累加
                metrics['voltage_mae'] += F.l1_loss(v_pred_phy, v_true_phy).item()
                metrics['current_mae'] += F.l1_loss(c_pred_phy, c_true_phy).item()
                metrics['power_mae'] += F.l1_loss(p_pred_phy, p_true_phy).item()

                # 新增 MSE 累加（reduction='sum' 得到该 batch 所有样本的平方误差之和）
                batch_size = labels.size(0)
                metrics['voltage_mse_sum'] += F.mse_loss(v_pred_phy, v_true_phy, reduction='sum').item()
                metrics['current_mse_sum'] += F.mse_loss(c_pred_phy, c_true_phy, reduction='sum').item()
                metrics['power_mse_sum'] += F.mse_loss(p_pred_phy, p_true_phy, reduction='sum').item()

                # 分类指标
                pred_labels = torch.argmax(cls_pred, dim=1)
                metrics['correct'] += (pred_labels == labels).sum().item()
                metrics['total_samples'] += labels.size(0)

                # 收集混淆矩阵数据
                arr = labels.cpu().numpy()
                all_true.append(arr.reshape(-1))  # 保证一维
                ap = torch.argmax(cls_pred, dim=1).cpu().numpy()
                all_pred.append(ap.reshape(-1))
        # 打印混淆矩阵
        all_true = np.concatenate(all_true)
        all_pred = np.concatenate(all_pred)
        cm = confusion_matrix(all_true, all_pred, labels=range(self.config['num_classes']))
        print(f"Epoch {epoch + 1} Confusion Matrix:\n", cm)

        # 计算最终指标
        # MEA
        metrics['voltage_mae'] /= len(self.val_loader)
        metrics['current_mae'] /= len(self.val_loader)
        metrics['power_mae'] /= len(self.val_loader)  # 平均功率 MAE
        # RMSE
        N = metrics['total_samples']
        metrics['voltage_rmse'] = (metrics['voltage_mse_sum'] / N) ** 0.5
        metrics['current_rmse'] = (metrics['current_mse_sum'] / N) ** 0.5
        metrics['power_rmse'] = (metrics['power_mse_sum'] / N) ** 0.5
        # 分类 acc / f1 / precision / recall
        metrics['acc'] = metrics['correct'] / metrics['total_samples']
        metrics['total'] = metrics['voltage_mae'] + metrics['current_mae']
        metrics['f1'] = f1_score(all_true, all_pred, average='weighted')
        metrics['precision'] = precision_score(all_true, all_pred, average='weighted')
        metrics['recall'] = recall_score(all_true, all_pred, average='weighted')

        # —— 新增：相对误差 & 复合指标 —— #
        # 计算相对误差时，应使用物理域的最大最小值，而不是 batch 内瞬时 range
        # voltage_rel = MAE / range
        # v_range = v_true_phy.max().item() - v_true_phy.min().item()
        v_min = float((self.voltage_mean - 3 * self.voltage_std).item())
        v_max = float((self.voltage_mean + 3 * self.voltage_std).item())
        metrics['voltage_rel'] = metrics['voltage_mae'] / (v_max - v_min)
        # current_rel = MAE / range
        c_min, c_max = self.config['current_range']
        # c_range = c_true_phy.max().item() - c_true_phy.min().item()
        metrics['current_rel'] = metrics['current_mae'] / (c_max - c_min)
        # composite: 40% acc + 30% (1–voltage_rel) + 30% (1–current_rel)
        # 复合指标
        metrics['composite'] = (
                0.4 * metrics['acc'] +
                0.3 * (1.0 - metrics['voltage_rel']) +
                0.3 * (1.0 - metrics['current_rel'])
        )
        # print("DEBUG val_metrics keys:", metrics.keys())
        return metrics

    def _compute_loss(self, pred, true, batch_params):
        reg_pred, cls_pred = pred
        voltage_true, current_true, labels = true

        # —— 网络输出——
        v_pred_norm = reg_pred[:, 0].unsqueeze(1)  # 归一化域预测
        c_pred_norm = reg_pred[:, 1].unsqueeze(1)
        # 数值保护：nan/inf→0，限幅防爆
        v_pred_norm = torch.where(torch.isfinite(v_pred_norm), v_pred_norm, torch.zeros_like(v_pred_norm))
        c_pred_norm = torch.where(torch.isfinite(c_pred_norm), c_pred_norm, torch.zeros_like(c_pred_norm))
        # 根据数据的 normalized 范围设定
        v_pred_norm = torch.clamp(v_pred_norm, min=-5.0, max=5.0)
        c_pred_norm = torch.clamp(c_pred_norm, min=-5.0, max=5.0)

        # 真值 voltage_true, current_true 都是 dataset 标准化后的值
        v_true_norm = voltage_true
        c_true_norm = current_true

        # —— 1) 电压、电流 回归（normalized domain）——
        voltage_loss = F.l1_loss(v_pred_norm, v_true_norm) * self.config['voltage_loss_weight']
        current_loss = F.l1_loss(c_pred_norm, c_true_norm) * self.config['current_loss_weight']

        # —— 2) 功率回归（normalized domain）——
        p_true_norm = (v_true_norm * c_true_norm).detach()
        p_pred_norm = v_pred_norm * c_pred_norm
        power_loss = F.l1_loss(p_pred_norm, p_true_norm) * self.config['power_loss_weight']

        # —— 3) 分类 Loss ——
        cls_loss = self.cls_criterion(cls_pred, labels) * self.config['cls_loss_weight']

        # —— 4) 强化物理一致性（Huber）可选 加 stronger phy_loss ——
        phy_loss = F.huber_loss(p_pred_norm, p_true_norm) * self.config.get('phy_loss_weight', 0.1)

        # —— 5) 物理边界惩罚 —— soft constraints
        # 电压 25–27V 之外的惩罚（normalized domain）
        v_phy = v_pred_norm * self.voltage_std + self.voltage_mean
        pen_v = (F.relu(25 - v_phy) + F.relu(v_phy - 27)).mean() * self.config.get('volt_range_weight', 1.0)
        # 电流 <= 0 之外的惩罚
        c_phy = c_pred_norm * self.current_std + self.current_mean
        pen_c = F.relu(-c_phy).mean() * self.config.get('curr_nonneg_weight', 1.0)
        penalize_loss = pen_v + pen_c

        # 加入金字塔融合的熵正则
        ent_loss = getattr(self.model, 'latest_ent_loss', torch.tensor(0.0, device=voltage_loss.device))
        total = voltage_loss + current_loss + power_loss + cls_loss + phy_loss + penalize_loss + ent_loss

        # 可选安全检查（关闭也可）
        # if not torch.isfinite(total):
        #     total = torch.tensor(1.0, device=total.device)

        return {
            'voltage': voltage_loss,
            'current': current_loss,
            'power': power_loss,
            'cls': cls_loss,
            'phy': phy_loss,
            'ent': ent_loss,  # 熵正则损失
            'total': total
        }

    # 新增物理约束方法
    def _physical_constraint_loss(self, v_pred, c_pred, p_pred, batch_params):
        """物理定律约束：P=VI，考虑车辆参数"""
        # 功率一致性
        power_calc = v_pred * c_pred
        power_constraint = F.huber_loss(power_calc, p_pred.unsqueeze(1))

        # 电压范围约束(25-27V)
        voltage_lower = F.relu(25 - v_pred)  # 当v_pred<25时为正
        voltage_upper = F.relu(v_pred - 27)  # 当v_pred>27时为正
        voltage_range = (voltage_lower + voltage_upper).mean()

        # 电流非负性
        current_constraint = F.relu(-c_pred).mean()

        # 速度-功率相关性 (batch_params: [B, 3] -> 速度在索引2)
        speed = batch_params[:, 2].unsqueeze(1)  # [B, 1]
        # 速度增加时功率应增加（正相关）
        speed_corr = F.relu(-(p_pred.detach() - power_calc) * speed)

        return (
                power_constraint +
                0.5 * voltage_range +
                0.2 * current_constraint +
                0.1 * speed_corr.mean()
        )

    def _save_checkpoint(self, epoch, metrics, suffix=''):
        """保存模型检查点，自动创建目录"""
        os.makedirs(self.config['checkpoint_path'], exist_ok=True)  # 确保目录存在
        filename = f"model_epoch{epoch + 1}"
        if suffix:
            filename += f"_{suffix}"
        filename += ".pth"

        full_path = os.path.join(self.config['checkpoint_path'], filename)  # 拼接完整路径

        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            # 'optimizer_state': self.optimizer.state_dict(),
            # 'scaler_state': self.scaler.state_dict(),
            # 'scheduler_state': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        # 可选保存：encoder 类
        if hasattr(self.train_dataset, 'ground_encoder'):
            checkpoint['encoder_classes'] = self.train_dataset.ground_encoder.classes_.tolist()

        # 保存归一化器 mean/scale 以便还原
        if hasattr(self.train_dataset, 'param_scaler'):
            scalers = {
                'params': {'mean': self.train_dataset.param_scaler.mean_,
                           'scale': self.train_dataset.param_scaler.scale_},
                'voltage': {'mean': self.train_dataset.voltage_scaler.mean_,
                            'scale': self.train_dataset.voltage_scaler.scale_},
                'current': {'mean': self.train_dataset.current_scaler.mean_,
                            'scale': self.train_dataset.current_scaler.scale_}
            }
            # 如果有 power_scaler 再加一条
            if hasattr(self.train_dataset, 'power_scaler'):
                scalers['power'] = {
                    'mean': self.train_dataset.power_scaler.mean_,
                    'scale': self.train_dataset.power_scaler.scale_
                }
            checkpoint['scaler'] = scalers

        torch.save(checkpoint, full_path)
        print(f"Checkpoint saved: {full_path} | Metrics: {metrics}")

    def final_visualization(self):
        """训练结束后生成最终可视化结果"""
        # 训练曲线 - 直接传递logger对象
        plot_training_curves(self.logger)

        # 特征空间可视化
        visualize_feature_space(self.model, self.train_loader, self.device)

        # 动态卷积核可视化
        visualize_dynamic_kernels()

    def _get_terrain_type(self, param_tensor):
        """根据参数推断地形类型"""
        param = param_tensor.cpu().numpy()
        original = self.train_dataset.param_scaler.inverse_transform([param])[0]
        # 这里添加地形判断逻辑
        return "Mixed Terrain"


if __name__ == "__main__":
    predictor = EnergyPredictor(CONFIG)
    try:
        predictor.train()
    finally:
        # predictor.final_visualization()
        predictor.writer.close()  # 确保TensorBoard写入器关闭
