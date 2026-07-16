# PINN-XX
Note: This project are pending for refactoring.

这 4 个脚本都属于同一类方案：不使用 PCNN，而是直接把灰度 Mel 图像送入 CNN 主干，再额外接一个物理场分支做 PINN 约束。它们的共同流程基本一致：数据集按 5 类读取、输入转灰度、提取特征、输出分类结果和物理场 `phys_out`、用交叉熵加拉普拉斯正则训练，并在测试时输出准确率、Precision/Recall/F1、G-Mean、标准差、t-SNE 和混淆矩阵。

## 各自做了什么
- PINN-LeNet.py 用的是 LeNet 风格的轻量卷积结构，适合做小模型基线。
- PINN-AlexNet.py 用的是自定义 AlexNet 风格卷积堆叠，并加了 SE 注意力模块，增强通道选择能力。
- PINN-VGGNet.py 用的是 VGG16-BN 风格的深层网络，分类器更大，结构更重。
- PINN-ResNet.py 用的是 ResNet18 特征提取，结构更现代，残差连接更利于训练。

## 主要区别
- 结构复杂度不同：LeNet 最轻，AlexNet 次之，ResNet 和 VGG 更深，VGG 的全连接头最大。
- 物理场是否参与分类融合不同：这 4 个文件里，LeNet、AlexNet、ResNet-phys 类似的版本会把物理场池化值拼进分类器；而 PINN-ResNet.py 和 PINN-VGGNet.py 这里是先分类、同时输出物理场，物理分支主要通过损失项约束，不是显式融合进分类头。
- 注意力机制不同：只有 PINN-AlexNet.py 明确加了 SE 模块。
- 数据集标签策略相同但实现上都显式固定了类别顺序，这比按目录自动排序更稳。
- 训练和评估细节基本一致，但 PINN-ResNet.py 的测试实现最简洁，没有像 VGG/AlexNet 那样做完整的特征可视化输出细化流程。

## 一句话总结
这几个文件本质上是在做“同一任务的不同骨干网络对比实验”：  
PINN-LeNet.py 是轻量基线，PINN-AlexNet.py 是加入 SE 的增强版，PINN-VGGNet.py 是深层大模型版，PINN-ResNet.py 是残差网络版，目的是比较不同主干在同一 PINN 约束下的分类效果。

# PCNN与PCNN_1
## 各自实现的功能
1. PCNN.py  
- 递归读取目录图片，统一缩放到 `224x224`。  
- 先把图像转为灰度（单通道）再做 PCNN。  
- `forward` 最终返回“最后一次迭代的二值脉冲图”（0/1）。  
- 按原目录结构保存到新目录。  

2. PCNN_1.py  
- 同样递归读取与缩放。  
- 保持 RGB，按 R/G/B 三个通道分别做 PCNN，再拼回 3 通道输出。  
- `forward` 做“多步脉冲累计”，最后除以步数得到归一化连续图（0~1）。  
- 同样按原目录结构保存。  

## 关键区别
1. 输入通道处理方式不同  
- PCNN.py：`RGB -> L` 灰度后处理。  
- PCNN_1.py：RGB 三通道分别处理，保留颜色通道信息。  

2. 输出形式不同  
- PCNN.py：输出最后一步脉冲结果，偏“硬二值”。  
- PCNN_1.py：输出累计脉冲强度，偏“软强度图”。  

3. 参数可调性不同  
- PCNN.py：参数固定在类内（`alpha_F=0.1, beta=0.2, VT=0.7`）。  
- PCNN_1.py：构造函数可传参（默认 `VT=0.8`，更偏抑噪）。  

4. 信息保留与计算代价  
- PCNN.py：更轻量，适合后续单通道网络。  
- PCNN_1.py：信息更丰富，但计算量约为前者 3 倍通道处理。  

5. 默认数据路径不同  
- PCNN.py：主函数里是 `data_mel_gadf_pcnn -> data_pcnn_mel_gasf`。  
- PCNN_1.py：主函数里是 `datasets_mel_gadf -> datasets_mel_gadf_pcnn_8`。  

## 适用范围
后续模型是单通道输入（如改过首层的 ResNet），通常 PCNN.py 更直接；
如果希望尽量保留颜色/纹理差异，可优先用 PCNN_1.py。

# PCNN-PINN-XX

## 共同功能
这五个文件都实现了同一条主线：
1. 读入灰度 Mel 图像数据集（5类）。
2. 先过 PCNNLayer 做脉冲增强。
3. 主干网络提取特征并输出分类结果。
4. 额外输出一个物理场分支 phys_out。
5. 用总损失 = 分类交叉熵 + 拉普拉斯物理正则。
6. 提供训练、验证、测试流程，保存模型与评估结果。

## 各自功能定位
1. PCNN-PINN-LeNet.py  
轻量版本。LeNet 主干 + 物理场反馈融合（把特征池化后与物理场池化结果拼接再分类）。训练和测试都带较完整评估（Accuracy、Precision/Recall/F1、G-Mean、STD、t-SNE、混淆矩阵）。

2. PCNN-PINN-AlexNet.py  
中等复杂度版本。不是标准 torchvision AlexNet，而是自定义 AlexNet 风格卷积堆叠，并加入 SE 注意力模块。也做物理场输出与完整评估（和 LeNet 类似）。

3. PCNN-PINN-VGGNet.py  
较深网络版本。基于 VGG16-BN 特征层，首层改成单通道输入，分类头较大（含 4096、1024 全连接），并把物理场池化值拼接进分类器输入。评估也很完整。

4. PCNN-PINN-ResNet.py  
基础 ResNet 版本。ResNet18 特征提取后直接分类，phys_out 只作为正则监督，不参与分类融合。测试输出较简洁，主要是 Accuracy + 混淆矩阵。这个文件更像“基础对照组”。

5. PCNN-PINN-ResNet-phys.py  
增强 ResNet 版本。与基础 ResNet 的关键差别是把物理场反馈显式融合进分类器（512+1 维），并扩展了完整评估流程（t-SNE、多指标、TensorBoard测试日志等）。

## 核心区别总结
1. 主干网络不同  
LeNet < AlexNet(SE) < VGG16/ResNet18，容量和计算量逐步上升。

2. 物理场是否参与分类决策不同  
- 参与融合：LeNet、VGGNet、ResNet-phys、original。  
- 不参与融合（只做物理正则）：ResNet。

3. 评估深度不同  
- 完整评估（t-SNE + 多分类指标 + G-Mean + STD + 混淆矩阵）：LeNet、AlexNet、VGGNet、ResNet-phys。  
- 简化评估：ResNet。

4. 训练配置有差异  
- Epoch：LeNet/AlexNet/VGG/ResNet-phys 多为 20；ResNet 为 30。  
- 优化器：ResNet 用 Adam；其余多用 SGD(momentum+weight_decay)。  
- 数据加载：LeNet/AlexNet/VGG/ResNet-phys 常设 num_workers=32；ResNet 默认未显式设高并行。

5. 类别映射策略不同  
- 显式 class_names 固定映射：LeNet/AlexNet/VGG/ResNet-phys。  
- 按目录排序自动映射：ResNet。  
这会影响标签编号一致性与跨脚本可复现性。

## 关于 original 文件
`PCNN-PIINN-ResNet-phys(original).py`的模型结构与 ResNet-phys 主思想一致（物理场反馈融合），但在训练参数、评估模块完整度、代码组织上更接近早期版本。可以把它看成 ResNet-phys 的原型稿。

# 模型结构对比表

| 文件 | 是否含 PCNN | 主干网络 | 物理场策略 | 分类头 / 特征融合 | 主要定位 |
|---|---|---|---|---|---|
| PCNN-PINN-LeNet.py | 是 | LeNet | phys_out 参与分类 | feat_avg 与 phys_avg 拼接后分类 | 轻量基线，完整评估 |
| PCNN-PINN-AlexNet.py | 是 | 自定义 AlexNet + SE | phys_out 参与分类 | 物理场与主干特征融合后分类 | 中等复杂度增强版 |
| PCNN-PINN-VGGNet.py | 是 | VGG16-BN | phys_out 参与分类 | 物理场与主干特征融合后分类 | 深层大模型版 |
| PCNN-PINN-ResNet.py | 是 | ResNet18 | phys_out 仅作拉普拉斯正则 | 不融合 phys_out | 基础对照组 |
| PCNN-PINN-ResNet-phys.py | 是 | ResNet18 | phys_out 参与分类 | feat_avg 与 phys_avg 拼接后分类 | ResNet 增强版，完整评估 |
| PINN-LeNet.py | 否 | LeNet | phys_out 参与分类 | feat_avg 与 phys_avg 拼接后分类 | 无 PCNN 的轻量基线 |
| PINN-AlexNet.py | 否 | 自定义 AlexNet + SE | phys_out 参与分类 | 物理场与主干特征融合后分类 | 无 PCNN 的增强版 |
| PINN-VGGNet.py | 否 | VGG16-BN | phys_out 仅作拉普拉斯正则 | 不融合 phys_out | 无 PCNN 的深层对照 |
| PINN-ResNet.py | 否 | ResNet18 | phys_out 参与分类 | feat_avg 与 phys_avg 拼接后分类 | 无 PCNN 的 ResNet 对照 |


# Mel-GAF-X
## 实现的功能
1. Mel-GAF.py  
- 最基础的单样本版本。  
- 输入一个 wav，提取 Mel（R 通道）+ GADF（G 通道）+ 零填充 B 通道，输出一张融合 RGB 图。  
- 适合快速验证思路，不适合直接跑大数据集。

2. Mel-GAF-0.py  
- 基于 torch/torchaudio 的版本。  
- 先做重采样（到 44100），再提 Mel 和 GAF（这里默认是 GASF，summation），拼成 2 通道张量，再补 1 个零通道保存为 RGB。  
- 按类别文件夹批处理（遍历 input_dir 下每个类别目录）。  
- 更像“深度学习特征张量先行”的实现。

3. Mel-GAF-1.py  
- 基于 librosa + cv2 的批处理版本。  
- Mel 做了 dB 限幅和 gamma 增强；GAF 可切换 GASF/GADF。  
- R=Mel，G=GAF，B=0，直接输出 RGB（三通道但不是伪彩）。  
- 使用 os.walk 递归遍历并保持目录结构，工程化程度较高。

4. Mel-GAF-2.py  
- 在“Mel+GAF 融合 RGB”后，再对整图做 colormap 伪彩。  
- 伪彩是对融合图三通道均值再上色，因此会弱化原本 R/G 分通道的语义。  
- 批处理只处理单层目录（os.listdir），不递归子目录。  
- 你注释里“Mel 特征不明显”与这一设计是吻合的。

5. Mel-GAF-3.py  
- 与 2 的主流程接近，但增强了 Mel：dB 裁剪到 [-80,0] + gamma（0.6）。  
- 目标是让 Mel 能量带更明显，再做伪彩。  
- 依然是单层目录批处理，不保留多级目录结构。

6. Mel-GAF-4.py  
- 标注为 paper 版，思路最完整。  
- 不做整图伪彩，保持融合通道可解释性：  
- R=Mel，G=GAF，B=Mel-GAF 差异图（归一化后）。  
- 支持递归批处理并保留目录结构。  
- 在“可解释性 + 批量落地”上是这些版本里最均衡的一版。

---

## 主要不同点
1. 输入处理链路不同  
- Mel-GAF-0.py 用 torchaudio，含显式重采样。  
- 其他多数用 librosa，通常 sr=None 保留原采样率，不主动统一重采样。

2. GAF 类型不同  
- Mel-GAF-0.py 默认 GASF（summation）。  
- Mel-GAF-1.py、Mel-GAF-2.py、Mel-GAF-3.py、Mel-GAF-4.py 支持 GASF/GADF 切换。  
- Mel-GAF.py 固定 GADF。

3. 融合策略不同  
- 简单双特征拼接：R=Mel，G=GAF，B=0（Mel-GAF.py、Mel-GAF-1.py、Mel-GAF-2.py、Mel-GAF-3.py）。  
- 增强差异表达：B=Mel-GAF（Mel-GAF-4.py）。  
- 张量拼接后补零通道：(Mel-GAF-0.py)。

4. 是否做伪彩不同  
- 不伪彩，保留通道语义：1、4、基础版。  
- 伪彩后语义被压缩到单映射：2、3。

5. 批处理能力不同  
- 单文件实验：(Mel-GAF.py)。  
- 单层批处理：(Mel-GAF-2.py、Mel-GAF-3.py)。  
- 递归批处理并保留目录结构：(Mel-GAF-1.py、Mel-GAF-4.py)。  
- 类别子目录遍历但非通用递归：(Mel-GAF-0.py)。

---

## 一句话建议
- 如果你要“论文主线 + 可解释 + 可批处理”，优先用 Mel-GAF-4.py。  
- 如果你要“最直观稳定的双特征融合基线”，用 Mel-GAF-1.py。  
- Mel-GAF-2.py 和 Mel-GAF-3.py 更适合做伪彩对照实验，不建议当主数据生成脚本。
