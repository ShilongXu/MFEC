import os
import csv
import json
import torch

class TrainingLogger:
    def __init__(self, log_path="training_log.csv"):
        self.history = []
        self.log_path = log_path
        # 如果已存在旧日志，先删掉
        if os.path.exists(self.log_path):
            os.remove(self.log_path)

    def log(self, epoch, train_metrics, val_metrics, num_train_batches, extras=None):
        """
        增加 num_train_batches 参数，用于平均损失，并安全处理 extras 中的各种类型数据
        """
        # 1. 平均 train loss
        avg = {k: v / num_train_batches for k, v in train_metrics.items()}
        # 2. 构造 log entry
        entry = {
            'epoch': epoch + 1,
            **{f"train_{k}": avg[k] for k in avg},
            **{f"val_{k}": val_metrics.get(k, 0) for k in [
                'total', 'voltage_mae', 'current_mae', 'power_mae',
                'acc', 'f1', 'precision', 'recall',
                'voltage_rel', 'current_rel', 'composite',
                'voltage_rmse', 'current_rmse', 'power_rmse'    #新增RMSE
            ]}
        }
        # 3. extras 转换
        if extras:
            for k, v in extras.items():
                # 安全处理多种数据类型
                if isinstance(v, torch.Tensor):
                    # 提取标量或均值
                    try:
                        val = v.detach().cpu()
                        # 对多维 tensor 取均值
                        entry[k] = val.mean().item()
                    except Exception:
                        entry[k] = val.item() if val.numel()==1 else val.tolist()
                elif isinstance(v, (list, tuple)):
                    # 列表或元组，尝试转为可序列化
                    try:
                        arr = torch.tensor(v) if not torch.is_tensor(v) else v
                        entry[k] = arr.mean().item()
                    except Exception:
                        entry[k] = list(v)
                elif isinstance(v, (int, float)):
                    entry[k] = v
                else:
                    # fallback to string
                    entry[k] = str(v)
        self.history.append(entry)

        # 4. 打印
        self._print_log(entry)
        # 5. 追加写 CSV
        self._write_csv(entry)

    def _print_log(self, e):
        print(f"\nEpoch {e['epoch']}")
        print(f"[Train] total={e['train_total']:.4f} | "
              f"voltage={e.get('train_voltage', 0):.4f} | current={e.get('train_current', 0):.4f}")
        print(f"[Val] total={e['val_total']:.4f} | "
              f"voltage_mae={e['val_voltage_mae']:.4f} | current_mae={e['val_current_mae']:.4f} | "
              f"acc={e['val_acc']:.2%}")
        print(f"f1={e['val_f1']:.4f} | precision={e['val_precision']:.4f} | recall={e['val_recall']:.4f} | "
              f"power_mae={e['val_power_mae']:.4f}")
        print(f"voltage_rel={e['val_voltage_rel']:.4f} | current_rel={e['val_current_rel']:.4f} | "
              f"composite={e['val_composite']:.4f}")

    def _write_csv(self, entry):
        # 首次调用时写 header
        write_header = not os.path.exists(self.log_path)
        with open(self.log_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(entry.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(entry)

    def flush(self):
        # 将 history 写成 JSON，然后清空
        with open(self.log_path.replace('.csv', '.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
        self.history.clear()
