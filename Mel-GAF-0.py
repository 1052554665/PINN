# 将文件夹中的.wav波形文件进行Mel和GAF图像特征融合
import os
import torch
import torchaudio
import librosa
import numpy as np
from pyts.image import GramianAngularField
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F

# 参数设置
sample_rate = 44100
n_fft=2048  # n_freqs = 1 + n_fft // 2  FFT窗口大小，决定频率分辨率
mel_size = (224, 224)
gaf_size = 224
gaf_transform = GramianAngularField(image_size=gaf_size, method='summation')

# 将 (1, Harmonic, W) 格式的张量调整为目标大小 size
def resize_tensor(tensor, size):
    tensor = tensor.unsqueeze(0)  # (1, 1, Harmonic, W)
    tensor = F.interpolate(tensor, size=size, mode='bilinear', align_corners=False)
    return tensor.squeeze(0)  # (1, Harmonic, W)

to_tensor = transforms.ToTensor()

# 提取融合特征：输出 shape (2, Harmonic, W)
def extract_fused_tensor(wav_path):
    waveform, sr = torchaudio.load(wav_path)
    if sr != sample_rate:   # 重采样
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)
    waveform = waveform.mean(dim=0) # 转换为单通道（平均左右声道）

    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft = n_fft, n_mels = mel_size[0])(waveform)
    mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-6)   # 归一化到 [0, 1] 范围
    mel_spec = mel_spec.unsqueeze(0)  # (1, Harmonic, W)
    mel_tensor = resize_tensor(mel_spec, mel_size)  # 调整尺寸为目标大小

    ts = waveform.numpy()
    ts = librosa.util.fix_length(ts, size=gaf_size)
    gaf_img = gaf_transform.fit_transform(ts[np.newaxis, :])[0]
    gaf_tensor = to_tensor(gaf_img)
    gaf_tensor = resize_tensor(gaf_tensor, mel_size)  # 保证与mel_tensor尺寸一致

    fused = torch.cat([mel_tensor, gaf_tensor], dim=0).type(torch.float32)
    return fused  # shape: (2, Harmonic, W)

# 保存 (2, Harmonic, W) 融合特征为 RGB PNG
def save_fused_tensor_as_rgb(tensor, save_path):
    if tensor.shape[0] == 2:
        c, h, w = tensor.shape
        tensor = torch.cat([tensor, torch.zeros(1, h, w)], dim=0)  # (3, Harmonic, W)
    img = (tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)  # HWC
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 自动创建子目录
    Image.fromarray(img).save(save_path)

# 批量处理数据集
def process_dataset(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    samples = []

    class_names = sorted(os.listdir(input_dir))
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    for cls in class_names:
        cls_path = os.path.join(input_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for fname in tqdm(os.listdir(cls_path), desc=f"Processing '{cls}'"):
            if not fname.endswith(".wav"):
                continue
            wav_path = os.path.join(cls_path, fname)

            # 构造保存路径：保持 input_dir 之后的相对路径结构
            relative_path = os.path.relpath(wav_path, input_dir)
            save_path = os.path.join(output_dir, os.path.splitext(relative_path)[0] + '.png')

            try:
                fused_tensor = extract_fused_tensor(wav_path)
                save_fused_tensor_as_rgb(fused_tensor, save_path)
            except Exception as e:
                print(f"Failed to process {wav_path}: {e}")

process_dataset(
    input_dir="data",                  # 原始语音目录
    output_dir="mel_gasf",           # 保存融合图像的目录
)

