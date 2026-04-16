# PCNN + PINN (ResNet18) 全流程代码
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torch.utils.tensorboard import SummaryWriter
import os
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights

# ========== 全局参数 ==========
NUM_CLASSES = 5
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-2
LAMBDA_PHY = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 数据集 ==========
class MelSpectrogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        for label, class_dir in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_dir)
            for fname in os.listdir(class_path):
                if fname.endswith((".png", ".jpg")):
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

    # 记录所有脉冲总和，形成一个“激活强度图”，可视性更高
    def forward(self, x):  # x: [B, 1, H, W]
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

# ========== 模型：PCNN + PINN(ResNet18) ==========
class PINNResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.pcnn = PCNNLayer(num_steps=10)

        weights = ResNet18_Weights.DEFAULT
        base = models.resnet18(weights=weights)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential(*list(base.children())[:-2])

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
        self.phys_out = nn.Conv2d(512, 1, kernel_size=1)    # phys_out:输出物理场

    def forward(self, x):
        x = self.pcnn(x)
        feat = self.features(x)
        out_cls = self.classifier(feat)
        out_phys = self.phys_out(feat)
        return out_cls, out_phys

# ========== 损失函数 ==========
# 拉普拉斯平滑正则项
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
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = PINNResNetClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
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
                pred_cls, _ = model(x)
                pred = pred_cls.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total
        writer.add_scalar("val/accuracy", acc, epoch)
        print(f"Epoch {epoch+1}/{EPOCHS} - Val Acc: {acc:.4f}")

    writer.close()
    torch.save(model.state_dict(), "pinn_pcnn_model.pth")

# ========== 测试函数 ==========
def test_model(test_dir, model_path="pinn_pcnn_model.pth"):
    test_dataset = MelSpectrogramDataset(test_dir, transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = PINNResNetClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            logits, _ = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"Test Accuracy: {acc:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Test Confusion Matrix")
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    # plt.show()

# ========== 主函数 ==========
if __name__ == "__main__":
    train_model("datasets/datasets_mel_gadf/train", "datasets/datasets_mel_gadf/val")
    test_model("datasets/datasets_mel_gadf/test")
