# Transformer Acoustic Analysis: MVDR + PINN/PCNN Classification

## 1. 项目简介
本项目基于多臂螺旋麦克风阵列采集变压器声纹信号，利用 MVDR 波束形成算法抑制风扇噪声与环境干扰，并通过深度学习模型（PINN/PCNN/VGGSE）实现变压器运行状态分类。

## 2. 功能模块
- MVDR 3D 波束形成（MATLAB）
- 噪声抑制与干扰消除
- 时频特征提取（STFT / Mel / GASF / GADF）
- PINN/PCNN 分类网络（Python + PyTorch）
- t-SNE 可视化、混淆矩阵、物理场可视化

## 3. 项目结构
transformer_acoustic_analysis/
├── matlab_mvdr/
├── python_nn/
├── docs/
├── results/
└── README.md



## 4. 使用流程
1. 在 `matlab_mvdr/` 中运行 MVDR 波束形成与特征提取脚本
2. 导出的 Mel/GASF/GADF 图像放入 `python_nn/datasets/`
3. 运行 Python 模型训练脚本
4. 使用可视化模块生成混淆矩阵与 t-SNE

## 5. 主要依赖
- MATLAB R2024b
- Python 3.9
- PyTorch
- NumPy, Matplotlib, Scikit-learn

## 6. 实验结果
| 方法 | 准确率 | G-Mean |
|------|--------|--------|
| PINN(ResNet18) | 98.4% | 97.9% |
| PCNN+PINN | 99.1% | 98.7% |

## 7. 引用
如果你使用了本项目，请引用下方格式（示例）：
> Author, "Transformer Acoustic Fault Diagnosis," 2025.
