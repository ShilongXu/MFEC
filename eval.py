import torch
import json
import numpy as np
import os
from torch.utils.data import DataLoader
from data.terrain_dataset import TerrainEnergyDataset
from models.multimodal_transformer import MultiModalTransformer
from config.defaults import CONFIG, TEST_CONFIG
from utils.metrics import calculate_metrics
from sklearn.preprocessing import StandardScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_test_transforms():
    """
        仅截取图片底部中心区域 (224x224)，滤除无关内容
        假设源分辨率为 640x480
        """
    return A.Compose([
        # 底部中心裁剪: y 从 256 到 480，x 从 208 到 432
        A.Crop(x_min=208, y_min=256, x_max=432, y_max=480),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

class Evaluator:
    """
    Evaluator for MultiModalTransformer models: performs batch-wise inference,
    computes metrics, and saves raw data for downstream visualization.
    """
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        # create output dirs
        os.makedirs(config.get('visualization_dir', 'visualization_results'), exist_ok=True)

        # load checkpoint
        ckpt = torch.load(config['model_path'], map_location=self.device)
        self.model = MultiModalTransformer(CONFIG).to(self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()

        # rebuild scalers
        sc = ckpt.get('scaler', {})
        def _make(s):
            ss = StandardScaler()
            ss.mean_ = np.array(s['mean'], dtype=float)
            ss.scale_ = np.array(s['scale'], dtype=float)
            return ss
        self.voltage_scaler = _make(sc['voltage'])
        self.current_scaler = _make(sc['current'])
        self.param_scaler = _make(sc['params'])
        self.power_scaler = _make(sc['power']) if 'power' in sc else None

        # prepare test dataset & loader
        self.test_dataset = TerrainEnergyDataset(
            root_dir=config['test_root'],
            mode='test',
            seq_length=config['seq_length'],
            transform=get_test_transforms(),
            scaler_dict={
                'voltage': self.voltage_scaler,
                'current': self.current_scaler,
                'power': self.power_scaler,
                'params': self.param_scaler
            },
            encoder_classes=ckpt.get('encoder_classes')
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 4)
        )

    def evaluate(self):
        """
        Run inference over test set, compute metrics, and save raw and summary data.
        Returns:
            metrics (dict): computed evaluation metrics.
        """
        # containers for raw values
        v_true, v_pred = [], []
        c_true, c_pred = [], []
        p_true, p_pred = [], []
        cls_true, cls_pred = [], []

        # inference
        with torch.no_grad():
            for batch in self.test_loader:
                imgs = batch['images'].to(self.device)
                params = batch['params'].to(self.device)
                elec = batch['elec_signals'].to(self.device)

                reg, cls_logits, _, _ = self.model(imgs, params, elec)

                # normalized preds
                v_norm = reg[:, 0].cpu().numpy()
                c_norm = reg[:, 1].cpu().numpy()
                # unscale
                v_p = self.voltage_scaler.inverse_transform(v_norm.reshape(-1,1)).ravel()
                c_p = self.current_scaler.inverse_transform(c_norm.reshape(-1,1)).ravel()
                p_p = v_p * c_p

                # true values
                v_t = self.voltage_scaler.inverse_transform(
                    batch['voltage'][:, -1].cpu().numpy().reshape(-1,1)
                ).ravel()
                c_t = self.current_scaler.inverse_transform(
                    batch['current'][:, -1].cpu().numpy().reshape(-1,1)
                ).ravel()
                p_t = v_t * c_t

                # collect
                v_true.extend(v_t.tolist()); v_pred.extend(v_p.tolist())
                c_true.extend(c_t.tolist()); c_pred.extend(c_p.tolist())
                p_true.extend(p_t.tolist()); p_pred.extend(p_p.tolist())

                cls_probs = torch.softmax(cls_logits, dim=1)
                cls_p = torch.argmax(cls_probs, dim=1).cpu().numpy()
                cls_true.extend(batch['ground_label'].cpu().numpy().tolist())
                cls_pred.extend(cls_p.tolist())

        # compute metrics
        scaler_tuples = {
            'voltage': (self.voltage_scaler.mean_[0], self.voltage_scaler.scale_[0]),
            'current': (self.current_scaler.mean_[0], self.current_scaler.scale_[0]),
            'power':  (self.power_scaler.mean_[0], self.power_scaler.scale_[0])
        }
        metrics = calculate_metrics(
            np.array(v_true), np.array(v_pred),
            np.array(c_true), np.array(c_pred),
            np.array(p_true), np.array(p_pred),
            np.array(cls_true), np.array(cls_pred),
            scalers=scaler_tuples
        )

        # save raw data for visualization
        raw_save_path = self.config.get('raw_data_path', 'evaluation_data.npz')
        np.savez_compressed(
            raw_save_path,
            v_true=np.array(v_true), v_pred=np.array(v_pred),
            c_true=np.array(c_true), c_pred=np.array(c_pred),
            p_true=np.array(p_true), p_pred=np.array(p_pred),
            cls_true=np.array(cls_true), cls_pred=np.array(cls_pred)
        )
        print(f"Raw evaluation data saved to {raw_save_path}")

        # save metrics summary
        metrics_path = self.config.get('out_json', 'evaluation_results.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Evaluation metrics saved to {metrics_path}")

        # save descriptions of each metric
        desc = {
            'voltage_mae': 'Mean Absolute Error between true and predicted voltage (V)',
            'current_mae': 'Mean Absolute Error between true and predicted current (A)',
            'power_mae'  : 'Mean Absolute Error between true and predicted power (W)',
            'voltage_rmse': 'Root Mean Squared Error for voltage (V)',
            'current_rmse': 'Root Mean Squared Error for current (A)',
            'power_rmse'  : 'Root Mean Squared Error for power (W)',
            'acc'         : 'Classification accuracy of terrain types',
            'f1'          : 'F1 score (weighted) for terrain classification',
            'voltage_rel' : 'Relative MAE for voltage: voltage_mae / voltage_range',
            'current_rel' : 'Relative MAE for current: current_mae / current_range',
            'power_rel'   : 'Relative MAE for power: power_mae / power_range'
        }
        desc_path = self.config.get('metrics_desc', 'metrics_descriptions.json')
        with open(desc_path, 'w') as f:
            json.dump(desc, f, indent=2)
        print(f"Metric descriptions saved to {desc_path}")

        # console print
        print("\n=== Final Evaluation Metrics ===")
        for k, v in metrics.items():
            print(f"{k}: {v}")

        return metrics

if __name__ == '__main__':
    evaluator = Evaluator(TEST_CONFIG)
    evaluator.evaluate()
