# 处理单张图像

import numpy as np
import librosa
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
from scipy.io import wavfile
import cv2
from PIL import Image
import torchaudio

# n_mels：Mel 滤波器组数量，决定频谱图的高度
# hop_length：帧移，每两个帧之间的采样点间距，影响时间轴长度，影响频谱图的时间分辨率
# n_fft：每帧进行 FFT 的点数，影响频率分辨率。较大值可提高频率精度，但降低时间精度
def generate_mel_spectrogram(y, sr, n_mels=256, hop_length=1024):
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=4096, hop_length=hop_length, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())  # 归一化到 [0, 1]
    return mel_norm

def generate_gaf_feature(data, size=224):
    # 归一化
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    gaf = GramianAngularField(image_size=len(data), method='difference')
    gaf_image = gaf.fit_transform(data.reshape(1, -1))[0]
    gaf_resized = cv2.resize(gaf_image, (size, size), interpolation=cv2.INTER_CUBIC)
    return gaf_resized

def generate_fusion_image(wav_path, output_path='mel_gadf_fusion.png', target_size=(224, 224)):

    # Step 1: 加载音频
    y, sr = librosa.load(wav_path, sr=None)  # librosa 自动转为 float32
    y_short = y[:300*10]  # 截取片段用于 GAF

    # Step 2: Mel 频谱图（归一化并resize）
    mel = generate_mel_spectrogram(y, sr)
    mel_resized = cv2.resize(mel, target_size, interpolation=cv2.INTER_CUBIC)

    # Step 3: GADF 图像
    gaf = generate_gaf_feature(y_short, size=target_size[0])

    # Step 4: 融合为 RGB 图像
    r = (mel_resized * 255).astype(np.uint8)    # 将归一化后的 Mel 频谱图（范围在 0,1）转换成图像格式（0~255 的整数），用于作为 RGB 图像的 R 通道
    g = (gaf * 255).astype(np.uint8)
    b = np.zeros_like(r, dtype=np.uint8)  # B 通道可以是 0 或 GASF 等
    fusion_rgb = np.stack([r, g, b], axis=2)

    # Step 5: 保存融合图像
    fusion_img = Image.fromarray(fusion_rgb)
    fusion_img.save(output_path)
    print(f"融合图像已保存至: {output_path}")

# 示例调用
generate_fusion_image('DCBias2_1.wav')
