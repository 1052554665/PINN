# PCNN + PINN (VGG16) 全流程整合版：物理场反馈 + 可视化模块 + 评估指标    paper 3-3 4-3

from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import os
from torchvision import transforms, models
from PIL import Image




# ========== 全局参数 ==========
NUM_CLASSES = 5
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-2
LAMBDA_PHY = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 数据集 ==========

class MelSpectrogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # 显式定义类别名顺序
        self.class_names = ['DCBias', 'Harmonic', 'Loosen', 'Normal', 'PartialDischarge']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        for class_name in self.class_names:
            class_path = os.path.join(root_dir, class_name)
            label = self.class_to_idx[class_name]
            for fname in os.listdir(class_path):
                if fname.lower().endswith((".png", ".jpg")):
                    self.samples.append((os.path.join(class_path, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# ========== PCNN 模块 ==========
class PCNNLayer(nn.Module):
    def __init__(self, num_steps=10):
        super().__init__()
        self.num_steps = num_steps
        self.alpha_F = 0.1
        self.beta = 0.2
        self.VT = 0.8

    def forward(self, x):
        B, C, H, W = x.shape
        Y = torch.zeros_like(x)
        F = x.clone()
        T = self.VT * torch.ones_like(x)
        for _ in range(self.num_steps):
            U = F * (1 + self.beta * Y)
            Y_new = (U > T).float()
            T = T * (1 + self.alpha_F) - Y_new * T * self.alpha_F
            Y = Y_new
        return Y

# ========== PINN VGG16 + PCNN ==========
class PINNVGG16Classifier(nn.Module):
    def __init__(self, num_classes=5, in_channels=1):
        super().__init__()
        self.pcnn = PCNNLayer(num_steps=20)

        vgg16 = models.vgg16_bn(weights=None)
        features = list(vgg16.features)
        features[0] = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.features = nn.Sequential(*features)

        self.phys_out = nn.Conv2d(512, 1, kernel_size=1)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7 + 1, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.pcnn(x)
        feat = self.features(x)
        out_phys = self.phys_out(feat)

        feat_avg = self.avgpool(feat)  # AdaptiveAvgPool2d((7,7))
        phys_avg = F.adaptive_avg_pool2d(out_phys, (1, 1))

        feat_flat = feat_avg.view(x.size(0), -1)  # flatten to (batch_size, 512*7*7)
        phys_flat = phys_avg.view(x.size(0), -1)  # flatten to (batch_size, 1)

        fused = torch.cat([feat_flat, phys_flat], dim=1)  # shape (batch_size, 512*7*7 + 1)

        out_cls = self.classifier(fused)
        return out_cls, out_phys


# ========== 拉普拉斯正则项 ==========
def laplacian_loss(u):
    u_xx = u[:, :, :-2, 1:-1] - 2 * u[:, :, 1:-1, 1:-1] + u[:, :, 2:, 1:-1]
    u_yy = u[:, :, 1:-1, :-2] - 2 * u[:, :, 1:-1, 1:-1] + u[:, :, 1:-1, 2:]
    lap = u_xx + u_yy
    return torch.mean(lap ** 2)

def total_loss(pred_cls, true_cls, phys_field):
    loss_cls = F.cross_entropy(pred_cls, true_cls)
    loss_phy = laplacian_loss(phys_field)
    return loss_cls + LAMBDA_PHY * loss_phy, loss_cls, loss_phy



# ========== 训练函数 ==========
def train_model(train_dir, val_dir, log_dir="runs/pinn_pcnn"):
    train_dataset = MelSpectrogramDataset(train_dir, transform)
    val_dataset = MelSpectrogramDataset(val_dir, transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers= 32)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers= 32)

    model = PINNVGG16Classifier(num_classes=NUM_CLASSES).to(DEVICE)
    # optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)  # 权重衰减（L2正则）

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        momentum = 0.9,  # 加快收敛
        weight_decay = 1e-4  # L2正则
    )
    writer = SummaryWriter(log_dir)

    global_step = 0
    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)

            pred_cls, phys = model(x)
            loss, loss_cls, loss_phy = total_loss(pred_cls, y, phys)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("train/total_loss", loss.item(), global_step)
            writer.add_scalar("train/class_loss", loss_cls.item(), global_step)
            writer.add_scalar("train/phys_loss", loss_phy.item(), global_step)
            global_step += 1

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred_cls, phys = model(x)
                pred = pred_cls.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

            # visualize_phys_field(phys, save_path=f"phys_epoch_{epoch+1}.png")

        acc = correct / total
        writer.add_scalar("val/accuracy", acc, epoch)
        print(f"Epoch {epoch+1}/{EPOCHS} - Val Acc: {acc:.4f}")

    writer.close()
    torch.save(model.state_dict(), "pinn_pcnn_model.pth")

# ========== 测试函数 ==========
def test_model(test_dir="datasets_mel_gadf/test", model_path="pinn_pcnn_model.pth", log_dir="runs/pinn_pcnn"):
    writer = SummaryWriter(log_dir)

    test_dataset = MelSpectrogramDataset(test_dir, transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = PINNVGG16Classifier(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            logits, phys = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    class_names = test_dataset.class_names

    # 把数字标签映射为类别名
    all_label_names = [class_names[label] for label in all_labels]

    # t-SNE 特征提取（用模型倒数第二层特征）
    # ========== 全量特征提取（取消 DataLoader） ==========
    all_features = []
    all_labels = []

    for idx in range(len(test_dataset)):
        x, y = test_dataset[idx]
        x = x.unsqueeze(0).to(DEVICE)  # 添加 batch 维度
        x_pcnn = model.pcnn(x)
        feat = model.features(x_pcnn)
        feat_avg = F.adaptive_avg_pool2d(feat, (1, 1)).view(1, -1)
        all_features.append(feat_avg.detach().cpu().numpy())
        all_labels.append(y)
    all_features = np.concatenate(all_features, axis=0)  # 从 list 合并成 ndarray，才能送入 PCA、TSNE

    # 在进入 PCA 或 TSNE 前清洗特征矩阵
    def check_nan_inf(features):
        nan_exists = np.isnan(features).any()
        inf_exists = np.isinf(features).any()
        print(f"NAN exists: {nan_exists}, INF exists: {inf_exists}")
        if nan_exists or inf_exists:
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            print("NaN/Inf found and replaced with 0.0")
        return features

    # 插入这行在 t-SNE 之前
    all_features = check_nan_inf(all_features)

    # PCA + t-SNE 降维
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    pca = PCA(n_components=min(30, all_features.shape[1]))
    features_pca = pca.fit_transform(all_features)

    tsne = TSNE(n_components=2, perplexity= 30, learning_rate=200, max_iter=1000, random_state=42)
    tsne_result = tsne.fit_transform(features_pca)

    # 转为 pandas Series 方便画图
    labels_series = pd.Series(all_label_names, name="Class")

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=tsne_result[:, 0], y=tsne_result[:, 1],
        hue=labels_series, palette="tab10", s=40, alpha=0.8, edgecolor="k", linewidth=0.3
    )
    plt.title("t-SNE Visualization of Test Features", fontsize=14)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.xlim(-100, 100)
    plt.ylim(-200, 200)
    plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("tsne_plot.png", dpi=600)
    plt.close()

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    # overall metrics
    acc = accuracy_score(all_labels, all_preds)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    # per-class metrics
    precision_all, recall_all, f1_all, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)

    # G-Mean
    gmean = np.prod(recall_all) ** (1.0 / len(recall_all))

    # 标准差
    precision_std = np.std(precision_all)
    recall_std = np.std(recall_all)
    f1_std = np.std(f1_all)

    # 输出
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Precision (weighted): {precision_w:.4f}")
    print(f"Recall (weighted): {recall_w:.4f}")
    print(f"F1-score (weighted): {f1_w:.4f}")
    print(f"G-Mean: {gmean:.4f}")
    print(f"Precision STD: {precision_std:.4f}")
    print(f"Recall STD: {recall_std:.4f}")
    print(f"F1-score STD: {f1_std:.4f}")

    # 写入 TensorBoard 日志
    writer.add_scalar("test/accuracy", acc)
    writer.add_scalar("test/precision_weighted", precision_w)
    writer.add_scalar("test/recall_weighted", recall_w)
    writer.add_scalar("test/f1_weighted", f1_w)
    writer.add_scalar("test/gmean", gmean)
    writer.add_scalar("test/precision_std", precision_std)
    writer.add_scalar("test/recall_std", recall_std)
    writer.add_scalar("test/f1_std", f1_std)

    writer.close()

    # 混淆矩阵绘制
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Test Confusion Matrix")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=600)
    plt.close()


# ========== 主函数 ==========
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    train_model("datasets_mel_gadf/train", "datasets_mel_gadf/val")
    test_model("datasets_mel_gadf/test")
