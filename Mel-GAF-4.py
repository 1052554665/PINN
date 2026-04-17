# PAPER
import os
import numpy as np
import librosa
import cv2
from PIL import Image
from pyts.image import GramianAngularField

def generate_mel_spectrogram(y, sr, n_mels=256, hop_length=1024, gamma=1.7):    # gamma越小对比越强
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=4096, hop_length=hop_length, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_db = np.clip(mel_db, a_min=-80, a_max=0)
    mel_norm = (mel_db + 80) / 80  # [-80, 0] → [0, 1]
    mel_norm = mel_norm ** gamma   # 增强强度差异
    return mel_norm


def generate_gaf_feature(data, size=224, method='difference'):
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    gaf = GramianAngularField(image_size=len(data), method=method)
    gaf_image = gaf.fit_transform(data.reshape(1, -1))[0]
    gaf_resized = cv2.resize(gaf_image, (size, size), interpolation=cv2.INTER_CUBIC)
    return gaf_resized

def generate_fusion_image(wav_path, output_path, target_size=(224, 224), use_gasf=False):
    y, sr = librosa.load(wav_path, sr=None)
    # 按长度截取N点（适合固定图像大小）
    y_short = y[:3000]  # 截取前一段用于 GAF 特征

    mel = generate_mel_spectrogram(y, sr)

    # OpenCV 和大多数图像库默认是左上角为原点，y 轴向下为正方向，所以纵轴上的频率被“翻转”了 —— 原本低频在下、高频在上，变成低频在上、高频在下。
    mel_resized = cv2.resize(mel, target_size, interpolation=cv2.INTER_CUBIC)
    mel_resized = np.flipud(mel_resized)  # 上下翻转，恢复正常频率方向

    # === R通道：Mel ===
    r = (mel_resized * 255).astype(np.uint8)

    # === G通道：GAF ===
    if use_gasf:
        gaf = generate_gaf_feature(y_short, size=target_size[0], method='summation')  # GASF
    else:
        gaf = generate_gaf_feature(y_short, size=target_size[0], method='difference')  # GADF

    g = (gaf * 255).astype(np.uint8)

    # === B通道：Mel - GAF（归一化到0~1再转换）===
    mel_resized_01 = mel_resized  # 原本已在 [0,1]
    gaf_resized_01 = gaf  # 原本也在 [0,1]

    diff = mel_resized_01 - gaf_resized_01
    # 为防止负值，先标准化到 [0, 1]
    diff_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-6)
    b = (diff_norm * 255).astype(np.uint8)

    fusion_rgb = np.stack([r, g, b], axis=2)
    fusion_img = Image.fromarray(fusion_rgb)
    fusion_img.save(output_path)

def batch_process_folder(wav_folder, output_folder, use_gasf=False):
    for root, _, files in os.walk(wav_folder):
        for f in files:
            if f.lower().endswith('.wav'):
                rel_path = os.path.relpath(root, wav_folder)
                output_subdir = os.path.join(output_folder, rel_path)
                os.makedirs(output_subdir, exist_ok=True)
                wav_path = os.path.join(root, f)
                img_name = os.path.splitext(f)[0] + ".png"
                output_path = os.path.join(output_subdir, img_name)
                generate_fusion_image(wav_path, output_path, use_gasf=use_gasf)


if __name__ == "__main__":
    input_wav_dir = "sample_mel_gasf"               # 输入音频文件夹
    output_img_dir = "data_mel_gasf_gamma"     # 输出图像保存路径
    batch_process_folder(input_wav_dir, output_img_dir, use_gasf=False)  # 设置 True 表示使用 GASF


