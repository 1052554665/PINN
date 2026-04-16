# 把一个目录下的图像批量做 PCNN 二值脉冲处理并转灰度，按原目录结构保存到新目录。
import torch
import torch.nn as nn
import os
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils

# 加载 PCNN 模块
class PCNNLayer(nn.Module):
    def __init__(self, num_steps=10):
        super().__init__()
        self.num_steps = num_steps
        self.alpha_F = 0.1
        self.beta = 0.2
        self.VT = 0.7

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

# 图像变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()  # 输出：C×H×W，值域[0,1]
])

# 保存 PCNN 图像
# RGB 转灰度后处理
def process_and_save_pcnn_images_gray(input_root, output_root):
    pcnn = PCNNLayer().eval()
    for root, _, files in os.walk(input_root):
        for fname in files:
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(root, fname)
                rel_path = os.path.relpath(input_path, input_root)
                output_path = os.path.join(output_root, rel_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                image = Image.open(input_path).convert("L")  # 转灰度
                tensor = transform(image).unsqueeze(0)  # [1, 1, H, W]

                with torch.no_grad():
                    processed = pcnn(tensor)

                save_tensor = processed.squeeze(0)  # [1, H, W]
                vutils.save_image(save_tensor, output_path)
                print(f"Saved: {output_path}")

# 示例用法
if __name__ == "__main__":
    process_and_save_pcnn_images_gray("data_mel_gadf_pcnn", "data_pcnn_mel_gasf")
