# 训练配置
CONFIG = {
    'num_classes': 3,
    'd_model': 256,
    'nhead': 8,
    'num_layers': 1,
    'lr': 1e-5,  # 学习率
    'min_lr': 1e-6,  # 最小学习率
    'weight_decay': 1e-3,  # 权重衰减
    'dropout_rate': 0.5,  # 全局dropout配置
    'grad_clip': 0.5,  # 梯度裁剪
    'epochs': 50,
    'batch_size': 8,
    'seq_length': 10,  # 滑动窗口序列长度
    'image_size': 224,
    'max_current': 8,  # 数据集中最大电流（A）
    'checkpoint_path': './checkpoints',
    'train_ratio': 0.8,
    'train_root': '/root/data/split_dataset/train',  # 训练数据集路径/data/dataset/train
    'val_root': '/root/data/split_dataset/verify',  # 验证数据集路径/data/dataset/verify
    # 损失权重参数化
    'reg_loss_weight': 1.0,
    'cls_loss_weight': 1.0,  # 分类权重
    'voltage_loss_weight': 1.0,
    'current_loss_weight': 1.0,
    'power_loss_weight': 1.0,
    'phy_loss_weight': 0.1,  # 物理约束损失权重
    'dyn_loss_weight': 0.05,
    'init_power_weight': 0.2,  # 功率动态权重
    'ent_loss_weight': 0.1,    # 熵正则化损失权重（参数感知金字塔融合的平滑约束）
    'power_loss_max': 0.3,  # 动态权重上限
    'power_loss_min': 0.1,  # 动态权重下限
    'early_stop_metric': 'total',  # 早停指标，可选 'total', 'acc', 'voltage_mae' 等
    'early_stop_patience': 5,  # 早停耐心值
    'improvement_threshold': 0.01,  # 早停改进阈值
    'min_epochs': 15,
    'feature_diversity_threshold': 0.5,  # 特征多样性阈值
    # 输出层参数约束
    'voltage_range': [25.0, 27.0],  # 实际输出物理范围约束
    'current_range': [1.0, 8.0],
    # 'power_range': [20, 200],
    # 特征多样性目标
    'min_feature_diversity': 0.3,
    # 标准化参数（与数据集统计一致）
    'voltage_norm': [25.94, 0.15],  # mean, std
    'current_norm': [3.71, 1.26],

}

# 测试配置
TEST_CONFIG = {
    'batch_size': 16,
    'seq_length': 10,
    'test_root': '/data/dataset/eval',
    'model_path': './checkpoints/best_model.pth',  # 与 train.py 保存的路径一致
    'raw_data_path': 'evaluation_data.npz',
    'out_json': 'evaluation_results.json',
    'metrics_desc': 'metrics_descriptions.json'
}
