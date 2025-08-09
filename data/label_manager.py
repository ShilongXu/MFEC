import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder


class DynamicLabelEncoder(BaseEstimator):
    """支持动态更新类别的标签编码器（追加时保持排序一致）"""

    def __init__(self, classes=None):
        self.encoder = LabelEncoder()
        if classes is not None:
            # 确保 classes 唯一且排序
            unique_classes = np.unique(classes)
            self.encoder.fit(unique_classes)

    def fit(self, y):
        """完整拟合编码器"""
        if len(y) > 0:
            unique_classes = np.unique(y)
            self.encoder.fit(unique_classes)
        return self

    def partial_fit(self, y):
        """动态更新编码器类别（追加时保持排序）"""
        y = np.array(y)
        if not hasattr(self.encoder, 'classes_') or len(self.encoder.classes_) == 0:
            unique_classes = np.unique(y)
            self.encoder.fit(unique_classes)
        else:
            new_classes = np.setdiff1d(np.unique(y), self.encoder.classes_)
            if len(new_classes) > 0:
                all_classes = np.concatenate([self.encoder.classes_, new_classes])
                unique_sorted = np.unique(all_classes)  ##### 修改点：确保排序
                self.encoder.fit(unique_sorted)

    def transform(self, y):
        """将标签转换为编码，遇到新类别自动 partial_fit 再次 transform"""
        y = np.array(y)
        if not hasattr(self.encoder, 'classes_') or len(self.encoder.classes_) == 0:
            unique_classes = np.unique(y)
            self.encoder.fit(unique_classes)
        try:
            return self.encoder.transform(y)
        except ValueError:
            # 发现新类别，动态更新后再 transform
            self.partial_fit(y)
            return self.encoder.transform(y)

    def inverse_transform(self, y):
        """将编码转换回标签"""
        return self.encoder.inverse_transform(y)

    @property
    def classes_(self):
        """获取当前所有类别"""
        return self.encoder.classes_

    @classes_.setter
    def classes_(self, value):
        """
        仅在从 checkpoint 恢复时使用。
        直接设置 classes_ 会同时设置内部 mapping；
        外部不要随意调用。
        """
        self.encoder.classes_ = np.array(value)
