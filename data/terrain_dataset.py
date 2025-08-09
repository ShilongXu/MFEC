import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from data.label_manager import DynamicLabelEncoder
from tqdm import tqdm


class TerrainEnergyDataset(Dataset):
    def __init__(self, root_dir, mode='train', seq_length=30, transform=None,
                 encoder_classes=None, scaler_dict=None):  # 增加scaler_dict参数
        self.root_dir = root_dir
        self.ground_encoder = DynamicLabelEncoder(encoder_classes)
        self.seq_length = seq_length
        self.transform = transform
        self.mode = mode

        # === 重构标准化器初始化 ===
        # 初始化标准化器实例
        self.voltage_scaler = StandardScaler()
        self.current_scaler = StandardScaler()
        self.power_scaler = StandardScaler()
        self.param_scaler = StandardScaler()

        if scaler_dict is not None:
            self.voltage_scaler = scaler_dict['voltage']
            self.current_scaler = scaler_dict['current']
            self.power_scaler = scaler_dict['power']
            self.param_scaler = scaler_dict['params']
        else:
            # 训练模式：收集数据并拟合标准化器
            self._collect_statistics()
        # 收集所有地形类型
        self.all_ground_types = self._collect_ground_types()
        # 拟合标签编码器
        if encoder_classes is None:
            self.ground_encoder.fit(self.all_ground_types)
        # 加载样本
        self.samples = []
        self._load_samples()

    def _collect_ground_types(self):
        """收集所有地形类型"""
        ground_types = set()
        for folder in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder)
            if os.path.isdir(folder_path):
                parts = folder.split('_')
                if parts:
                    ground_types.add(parts[0])
        print(f"Found ground types: {list(ground_types)}")
        return list(ground_types)

    def _load_samples(self):
        """加载样本：直接遍历 self.root_dir下所有子文件夹，不再做 split"""
        print(f"\nLoading data from: {self.root_dir}")

        # 获取所有文件夹
        all_folders = [f for f in os.listdir(self.root_dir)
                       if os.path.isdir(os.path.join(self.root_dir, f))]
        if not all_folders:
            raise ValueError(f"No valid folders found in {self.root_dir}")

        # 直接遍历 all_folders
        self.samples = []
        for folder in tqdm(all_folders, desc=f"Loading {'train' if self.mode == 'train' else 'val'} folders"):
            folder_path = os.path.join(self.root_dir, folder)
            try:
                # 解析文件夹
                parts = folder.split('_')
                ground_type = parts[0] if parts else None
                # 【参照 2.2.1 修正 numeric 提取】
                numeric = []
                for i in range(1, min(4, len(parts))):
                    part = parts[i]
                    for unit in ['kg', 'bar', 'm/s', 'ms']:
                        part = part.replace(unit, '')
                    try:
                        numeric.append(float(part))
                    except ValueError:
                        numeric.append(0.0)
                while len(numeric) < 3:
                    numeric.append(0.0)

                # ground_label 计算同前
                try:
                    ground_label = self.ground_encoder.transform([ground_type])[0]
                except ValueError:
                    self.ground_encoder.partial_fit([ground_type])
                    ground_label = self.ground_encoder.transform([ground_type])[0]

                data_file = os.path.join(folder_path, 'DataSum.csv')
                if not os.path.exists(data_file):
                    print(f"Warning: Missing DataSum.csv in {folder}")
                    continue
                power_df = pd.read_csv(data_file)
                if len(power_df) < self.seq_length:
                    print(f"Warning: Not enough rows ({len(power_df)}) in {folder}/DataSum.csv")
                    continue
                power_df = self._convert_units(power_df)

                # 标准化
                scaled_power = self.power_scaler.transform(
                    power_df['power'].values.reshape(-1, 1)).flatten()  # 得到 [N] 长度数组
                scaled_params = self.param_scaler.transform([numeric])[0]
                scaled_voltage = self.voltage_scaler.transform(power_df['voltage'].values.reshape(-1, 1)).flatten()
                scaled_current = self.current_scaler.transform(power_df['current'].values.reshape(-1, 1)).flatten()

                num_samples = len(power_df) - self.seq_length + 1
                step_size = max(1, self.seq_length // 4)
                for start_idx in range(0, num_samples, step_size):
                    sample = self._create_sample(
                        folder_path, power_df, start_idx, scaled_params,
                        scaled_voltage, scaled_current, scaled_power,
                        ground_label
                    )
                    if sample is not None:
                        self.samples.append(sample)
                print(f"Folder {folder}: {len(power_df)} rows -> {num_samples} samples (step={step_size})")
            except Exception as e:
                print(f"Error processing {folder}: {str(e)}")
                continue

        print(f"Mode: {self.mode}, Root Dir: {self.root_dir}")
        print(f"Successfully loaded {len(self.samples)} valid samples")

    def _collect_statistics(self):
        """收集全局统计数据并拟合标准化器（仅训练模式调用）"""
        # 重置数据容器
        self.all_params = []
        self.all_voltages = []
        self.all_currents = []
        self.all_powers = []

        # 遍历所有文件夹收集数据
        for folder in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            try:
                # 解析文件夹名获取参数
                parts = folder.split('_')
                ground_type = parts[0] if parts else "Unknown"

                # 提取数值参数
                numeric = []
                for i in range(1, min(4, len(parts))):
                    part = parts[i]
                    # 清理单位标识
                    for unit in ['kg', 'bar', 'm/s', 'ms']:
                        part = part.replace(unit, '')
                    try:
                        numeric.append(float(part))
                    except ValueError:
                        numeric.append(0.0)
                        print(f"Warning: Invalid numeric value in {folder} - {parts[i]}")

                # 补全缺失参数
                while len(numeric) < 3:
                    numeric.append(0.0)

                self.all_params.append(numeric)

                # 读取并处理数据文件
                data_file = os.path.join(folder_path, 'DataSum.csv')
                if os.path.exists(data_file):
                    power_df = pd.read_csv(data_file)
                    power_df = self._convert_units(power_df)

                    # 确保列存在
                    if 'voltage' in power_df.columns:
                        self.all_voltages.extend(power_df['voltage'].values)
                    if 'current' in power_df.columns:
                        self.all_currents.extend(power_df['current'].values)
                    if 'power' in power_df.columns:
                        self.all_powers.extend(power_df['power'].values)
                    else:
                        # 计算功率如果不存在
                        power_df['power'] = power_df['voltage'] * power_df['current']
                    self.all_powers.extend(power_df['power'].values)
            except Exception as e:
                print(f"Error in data collection for {folder}: {str(e)}")
                continue

        # 转换为numpy数组并拟合
        if self.all_voltages:
            self.all_voltages = np.array(self.all_voltages)
            self.voltage_scaler.fit(self.all_voltages.reshape(-1, 1))

        if self.all_currents:
            self.all_currents = np.array(self.all_currents)
            self.current_scaler.fit(self.all_currents.reshape(-1, 1))

        if self.all_powers:
            self.all_powers = np.array(self.all_powers)
            self.power_scaler.fit(self.all_powers.reshape(-1, 1))

        if self.all_params:
            self.all_params = np.array(self.all_params)
            self.param_scaler.fit(self.all_params)

        # 打印标准化器参数
        print("\nScaler parameters (auto-fitted):")
        if hasattr(self.voltage_scaler, 'mean_') and self.voltage_scaler.mean_ is not None:
            print(
                f"电压标准化参数 - mean: {self.voltage_scaler.mean_[0]:.2f}, scale: {self.voltage_scaler.scale_[0]:.2f}")
        if hasattr(self.current_scaler, 'mean_') and self.current_scaler.mean_ is not None:
            print(
                f"电流标准化参数 - mean: {self.current_scaler.mean_[0]:.2f}, scale: {self.current_scaler.scale_[0]:.2f}")

    def _parse_folder(self, folder_name):
        parts = folder_name.split('_')
        if len(parts) != 4:
            raise ValueError(f"Invalid folder name format: {folder_name}. "
                             "Expected format: <ground_type>_<load>kg_<pressure>bar_<speed>")

        try:
            return {
                'ground_type': parts[0],  # 地形类型
                'numeric': [
                    float(parts[1].replace('kg', '')),
                    float(parts[2].replace('bar', '')),
                    float(parts[3])
                ]
            }
        except ValueError as e:
            print(f"Error parsing folder {folder_name}: {str(e)}")
            raise

    def _convert_units(self, power_df):
        power_df = power_df.copy()

        # 数据验证
        if 'voltage' not in power_df.columns or 'current' not in power_df.columns:
            raise ValueError("CSV文件缺少电压或电流列")

        # 单位转换
        power_df['current'] = np.abs(power_df['current']) / 1000
        power_df['voltage'] = np.abs(power_df['voltage'])
        power_df['power'] = power_df['voltage'] * power_df['current']

        # 范围验证
        valid_voltage = (power_df['voltage'] > 20) & (power_df['voltage'] < 30)
        valid_current = (power_df['current'] < 10)
        valid_power = (power_df['power'] > 0) & (power_df['power'] < 200)

        valid_mask = valid_voltage & valid_current & valid_power

        if valid_mask.sum() < len(power_df) * 0.8:
            print(f"警告: 超过20%的数据被过滤 (原始:{len(power_df)} 有效:{valid_mask.sum()})")

        return power_df[valid_mask]

    def _create_sample(self, folder_path, power_df, start_idx, scaled_params,
                       scaled_voltage, scaled_current, scaled_power, ground_label):
        end_idx = start_idx + self.seq_length
        sample = {
            'images': [],
            'params': scaled_params,
            'voltage': scaled_voltage[start_idx:end_idx],
            'current': scaled_current[start_idx:end_idx],
            'power': scaled_power[start_idx:end_idx],
            'ground_label': np.array([ground_label], dtype=np.int64)
        }
        missing_count = 0
        for idx in range(start_idx, end_idx):
            timestamp = power_df.iloc[idx]['PictureName']
            img_path = os.path.join(folder_path, f"{timestamp}.jpg")
            if not os.path.exists(img_path):
                missing_count += 1
                blank_img = np.zeros((224, 224, 3), dtype=np.uint8)
                sample['images'].append(blank_img)
            else:
                img = cv2.imread(img_path)
                if img is None:
                    missing_count += 1
                    blank_img = np.zeros((224, 224, 3), dtype=np.uint8)
                    sample['images'].append(blank_img)
                else:
                    sample['images'].append(img)
        # 若缺失图像过多，则丢弃此样本(缺失比例30%)
        if missing_count > int(self.seq_length * 0.3):
            return None
        return sample

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 返回标准化后的数据
        return {
            'images': torch.stack([self._process_image(img) for img in sample['images']]),
            'params': torch.FloatTensor(sample['params']),
            'elec_signals': torch.FloatTensor(np.column_stack([
                sample['voltage'],
                np.abs(sample['current']),
                sample['power']
            ])),
            'ground_label': torch.from_numpy(sample['ground_label']).long().squeeze(),
            # 返回标准化值，非原始物理值
            'voltage': torch.FloatTensor(sample['voltage']),  # normalized V
            'current': torch.FloatTensor(np.abs(sample['current'])),  # normalized A
            'power': torch.FloatTensor(sample['power'])  # normalized W
        }

    def _process_image(self, img):
        # 1. 转换到 RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 2. 如果定义了 Albumentations transform，就全权交给它（包括 crop/resize）
        if self.transform:
            out = self.transform(image=img)['image']
            # Albumentations 已经返回 Tensor 并做了 Normalize
            return out.float()
        # 3. 否则，手动 resize 到 224×224
        h, w = img.shape[:2]

        if h != 224 or w != 224:
            img = cv2.resize(img, (224, 224))
        # 转为 Tensor & 归一化到 [0,1]
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
