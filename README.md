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