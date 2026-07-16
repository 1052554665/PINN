# 备选    RGB三通道融合后上伪彩    Mel特征不明显
import os
import numpy as np
import librosa
import cv2
from PIL import Image
from pyts.image import GramianAngularField
from tqdm import tqdm
import matplotlib.pyplot as plt

# === 图像生成函数（Mel + GADF + 可选 GASF） ===
def generate_mel_spectrogram(y, sr, n_mels=256, hop_length=2048):
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=4096, hop_length=hop_length, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
    return mel_norm

def generate_gaf_image(data, N=3000, method='difference', resize=(224, 224)):
    if data.ndim > 1:
        data = np.mean(data, axis=1)  # 多通道转单通道
    data = data.astype(np.float32)[:N]
    data = (data - np.min(data)) / (np.max(data) - np.min(data))  # 归一化

    gaf = GramianAngularField(image_size=len(data), method=method)
    gaf_image = gaf.fit_transform(data.reshape(1, -1))[0]

    # 将 gaf_image的像素值线性缩放到 [0, 1] 的范围
    range_val = np.max(gaf_image) - np.min(gaf_image)
    if range_val < 1e-8:
        gaf_norm = np.zeros_like(gaf_image)
    else:
        gaf_norm = (gaf_image - np.min(gaf_image)) / range_val

    return gaf_norm     # 返回 float32 矩阵（范围 0~1）


def generate_fusion_image(wav_path, output_path='', target_size=(224, 224), use_gasf = False, cmap='viridis'):  # viridis  plasma inferno magma  cividis gray

    # Step 1: 加载音频
    y, sr = librosa.load(wav_path, sr=None)  # librosa 自动转为 float32
    N = 5000
    y_short = y[:N]  # 截取片段用于 GAF

    # Step 2: Mel 频谱图（归一化并resize）
    mel = generate_mel_spectrogram(y, sr)
    mel_resized = cv2.resize(mel, target_size, interpolation=cv2.INTER_CUBIC)

    # Step 3: GAF 灰度图（未上色）
    gaf_method = 'summation' if use_gasf else 'difference'
    gaf_gray = generate_gaf_image(y_short, method=gaf_method)
    gaf_resized = cv2.resize(gaf_gray, target_size, interpolation=cv2.INTER_CUBIC)

    # Step 4: 融合为 RGB 图像
    r = (mel_resized * 255).astype(np.uint8)    # 将归一化后的 Mel 频谱图（范围在 0,1）转换成图像格式（0~255 的整数），用于作为 RGB 图像的 R 通道
    g = (gaf_resized * 255).astype(np.uint8)
    b = np.zeros_like(r, dtype=np.uint8)  # B 通道可以是 0 或 GASF 等
    fusion_rgb = np.stack([r, g, b], axis=2)

    # Step 5: 应用 colormap 到整个融合图像
    fusion_float = fusion_rgb.astype(np.float32) / 255.0
    cmap_func = plt.get_cmap(cmap)
    colored = cmap_func(fusion_float.mean(axis=2))[:, :, :3]  # 平均后上伪彩
    colored_uint8 = (colored * 255).astype(np.uint8)

    # Step 6: 保存图像
    fusion_img = Image.fromarray(colored_uint8)
    fusion_img.save(output_path)
    print(f"✅ 融合图像已保存至: {output_path}")



# === 批量处理函数 ===
def batch_process_folder(wav_folder, output_folder, use_gasf=False):
    os.makedirs(output_folder, exist_ok=True)
    wav_files = [f for f in os.listdir(wav_folder) if f.lower().endswith('.wav')]

    for wav_file in tqdm(wav_files, desc="Processing WAV files"):
        wav_path = os.path.join(wav_folder, wav_file)
        img_name = os.path.splitext(wav_file)[0] + ".png"
        output_path = os.path.join(output_folder, img_name)
        generate_fusion_image(wav_path, output_path, use_gasf=use_gasf)

    print(f"\n✅ 所有图像已保存到：{output_folder}")

# === 示例调用 ===
if __name__ == "__main__":
    input_wav_dir = "sample"  # 替换为你的音频文件夹路径
    output_img_dir = "processed_sample_mel_gaf"  # 替换为保存图像的输出路径
    batch_process_folder(input_wav_dir, output_img_dir, use_gasf=False)  # 设置 True 表示使用 GASF