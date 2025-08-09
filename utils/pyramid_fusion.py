import torch
import torch.nn as nn
import torch.nn.functional as F

class ParamAwarePyramidFusion(nn.Module):
    """
    参数感知金字塔加权融合 + 残差门控
    输入：
        feats: list of 4 张量, 每个形状 [N, C]，对应 P1-P4 全局池化后的特征
        params: 张量 [N, param_dim]，车辆状态
    输出：
        fused: 张量 [N, C]，融合后特征
        ent_loss: 熵正则化项（标量）
    """
    def __init__(self, param_dim: int, C: int, hidden_dim: int = 64, temperature: float = 1.0, ent_lambda: float = 0.01):
        super().__init__()
        self.temperature = temperature
        self.ent_lambda = ent_lambda

        # 1) 隐藏映射 W1: param_dim → hidden_dim
        self.fc1 = nn.Linear(param_dim, hidden_dim)
        # 2) Logits 映射 W2: hidden_dim → 4
        self.fc2 = nn.Linear(hidden_dim, 4)
        # 5.1) 门控映射 Wr: hidden_dim → 1
        self.fc_r = nn.Linear(hidden_dim, 1)

        # 保证特征维度一致
        assert C > 0
        self.C = C

    def forward(self, feats: list, params: torch.Tensor):
        """
        feats: [P1, P2, P3, P4], each [N, C]
        params: [N, param_dim]
        """
        N = params.size(0)
        # 1. 隐藏向量 h_t
        h = F.relu(self.fc1(params))               # [N, hidden_dim]

        # 2. logits o_t
        logits = self.fc2(h)                       # [N, 4]
        # —— 数值保护：将 logits 限幅在 [-10,10]，防止 softmax 溢出 ——#
        logits = torch.clamp(logits, min=-10.0, max=10.0)

        # 3. 带温度的 softmax 计算 α
        # (N,4) -> (N,4)
        alpha = F.softmax(logits / self.temperature, dim=1)

        # 4. 熵正则化
        # α_log = α * log(α+eps)
        eps = 1e-6
        log_alpha = (alpha + eps).log()  # [N,4]
        ent_per_sample = - (alpha * log_alpha).sum(dim=1)  # [N]
        ent_per_sample = torch.where(   # 过滤非数值项
            torch.isfinite(ent_per_sample),
            ent_per_sample,
            torch.zeros_like(ent_per_sample)
        )
        ent = ent_per_sample.mean()  # 标量
        ent_loss = self.ent_lambda * ent
        if not torch.isfinite(ent_loss):
            ent_loss = torch.tensor(0.0, device=params.device)
        # ent_loss = torch.nan_to_num(ent_loss, nan=0.0, posinf=0.0, neginf=0.0) # 最后一层数值保护：替换 nan/inf 为 0

        # 5. 金字塔加权融合 f_wtd = ∑ α_i * f_i
        # 将 feats 拼成 [N,4,C]
        stacked = torch.stack(feats, dim=1)        # [N,4,C]
        # (N,4,1) * (N,4,C) -> (N,4,C) -> sum -> (N,C)
        alpha_unsq = alpha.unsqueeze(-1)           # [N,4,1]
        f_wtd = (alpha_unsq * stacked).sum(dim=1)  # [N,C]

        # 5.1 门控系数 γ
        gamma = torch.sigmoid(self.fc_r(h))        # [N,1]

        # 5.2 残差式输出 f_out = γ f_wtd + (1-γ) f3
        f3 = feats[2]                              # P3 as基准尺度
        fused = gamma * f_wtd + (1 - gamma) * f3    # [N,C]

        return fused, ent_loss, alpha, gamma
