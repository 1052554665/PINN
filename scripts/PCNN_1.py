import torch.nn as nn
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils

# 加载 PCNN 模块
class PCNNLayer(nn.Module):
    def __init__(self, num_steps=10, alpha_F = 0.1, beta = 0.2, VT = 0.8):  # VT设置大点，去除噪声更明显
        super().__init__()
        self.num_steps = num_steps
        self.alpha_F = alpha_F
        self.beta = beta
        self.VT = VT

    # 使用连续脉冲累计图
    def forward(self, x):
        B, C, H, W = x.shape
        Y = torch.zeros_like(x)
        F = x.clone()
        T = self.VT * torch.ones_like(x)
        Y_accum = torch.zeros_like(x)  # 用于累计

        for _ in range(self.num_steps):
            U = F * (1 + self.beta * Y)
            Y_new = (U > T).float()
            T = T * (1 + self.alpha_F) - Y_new * T * self.alpha_F
            Y = Y_new
            Y_accum += Y_new  # 累加每轮脉冲

        # 归一化到 0~1 之间便于显示
        Y_norm = Y_accum / self.num_steps
        return Y_norm


# 图像变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()  # 输出：C×H×W，值域[0,1]
])

# RGB 三通道分别 PCNN 处理再拼接（增强信息保留）
def process_and_save_pcnn_images_rgb(input_root, output_root):
    pcnn = PCNNLayer().eval()
    rgb_split = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    for root, _, files in os.walk(input_root):
        for fname in files:
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(root, fname)
                rel_path = os.path.relpath(input_path, input_root)
                output_path = os.path.join(output_root, rel_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                image = Image.open(input_path).convert("RGB")
                tensor = rgb_split(image)  # [3, H, W]
                tensor = tensor.unsqueeze(0)  # [1, 3, H, W]

                with torch.no_grad():
                    # 分通道 PCNN
                    channels = []
                    for i in range(3):
                        ch = tensor[:, i:i+1, :, :]  # [1,1,H,W]
                        ch_out = pcnn(ch)
                        channels.append(ch_out)

                    processed = torch.cat(channels, dim=1)  # [1,3,H,W]

                vutils.save_image(processed.squeeze(0), output_path)
                print(f"Saved: {output_path}")

if __name__ == "__main__":
    # process_and_save_pcnn_images_rgb("mel_gadf", "data_pcnn_mel_gadf")
    process_and_save_pcnn_images_rgb("datasets_mel_gadf", "datasets_mel_gadf_pcnn")

