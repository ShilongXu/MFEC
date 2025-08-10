# MFEC — Energy cost prediction method for wheeled off-road vehicles based on multimodal fusion
**MFEC**（Multimodal Fusion for Energy Consumption）是一套用于越野轮式车辆能耗估计的开源实现，基于视觉（ResNet50）、车辆状态编码、与时序电信号（LSTM）等多模态输入，通过状态自适应模块（参数感知动态卷积、参数感知金字塔融合、门控自注意力融合等）实现对越野场景下车辆电压/电流/功率的实时估计。该项目包含训练、评估、消融实验与在真实平台（ROS）上的部署示例。

## 环境与依赖
### 推荐环境（与论文中一致）

- OS: Ubuntu 20.04

- Python: 3.8

- PyTorch: 1.13.1 (CUDA 11.7.1)

- CUDA: 11.7.1

- GPU: NVIDIA (建议 16GB 显存或更高，用于较大 batch)

### 快速安装
建议先创建并激活虚拟环境（conda 或 venv），然后按下列步骤安装依赖。

```bash
# 安装 PyTorch (conda 推荐)
conda create -n mfec python=3.8 -y
conda activate mfec
# PyTorch 1.13.1 + CUDA 11.7 (conda 安装)
conda install pytorch==1.13.1 torchvision torchaudio cudatoolkit=11.7 -c pytorch -c nvidia -y

# 安装其它 Python 包（使用清华镜像）
pip install opencv-python pandas scikit-learn seaborn albumentations==0.5.2 torchviz -i https://pypi.tuna.tsinghua.edu.cn/simple

```
## 数据集
数据集涉及国土资源安全，未经过相关部门批准，暂时不提供。

## 参数设置
在文件defaults.py中已经有了基础配置，如果要根据自己的情况进行配置，请见文章：“Energy Cost Prediction Method for Wheeled Off-road Vehicles Based on Multi-modal Fusion”实验部分。
