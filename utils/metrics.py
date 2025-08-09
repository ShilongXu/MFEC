import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score


def calculate_metrics(v_true, v_pred, c_true, c_pred, p_true, p_pred,
                      cls_true, cls_pred, scalers=None):
    """
    Args:
        scalers: dict {'voltage': StandardScaler, 'current': StandardScaler, ...}
    """
    # 反归一化
    v_pred_unscaled = v_pred * scalers['voltage'][1] + scalers['voltage'][0]
    c_pred_unscaled = c_pred * scalers['current'][1] + scalers['current'][0]
    p_pred_unscaled = v_pred_unscaled * c_pred_unscaled

    metrics = {
        'voltage_mae': mean_absolute_error(v_true, v_pred_unscaled),
        'current_mae': mean_absolute_error(c_true, c_pred_unscaled),
        'power_mae': mean_absolute_error(p_true, p_pred_unscaled),
        'voltage_rmse': np.sqrt(mean_squared_error(v_true, v_pred_unscaled)),
        'current_rmse': np.sqrt(mean_squared_error(c_true, c_pred_unscaled)),  # 原代码乘1000
        'power_rmse': np.sqrt(mean_squared_error(p_true, p_pred_unscaled)),
        'acc': accuracy_score(cls_true, cls_pred),
        'f1': f1_score(cls_true, cls_pred, average='weighted'), # 权重f1分数
        'voltage_rel': 0.0,
        'current_rel': 0.0,
        'power_rel': 0.0
    }

    # 计算相对误差
    try:
        voltage_range = np.ptp(v_true)
        current_range = np.ptp(c_true)
        power_range = np.ptp(p_true)
        if voltage_range > 1e-6:
            metrics['voltage_rel'] = metrics['voltage_mae'] / voltage_range
        if current_range > 1e-6:
            metrics['current_rel'] = metrics['current_mae'] / current_range
        if power_range > 1e-6:
            metrics['power_rel'] = metrics['power_mae'] / power_range
    except Exception:
        pass

    return metrics
