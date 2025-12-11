clear; clc;

%% 读取信号
% [x, fs] = audioread("DCBias2_1.wav");
% [x, fs] = audioread("G4_10pSeventhHarmonic_1.wav");
% [x, fs] = audioread("Loosen1_1.wav");
% [x, fs] = audioread("Normal_part1.wav");
[x, fs] = audioread("PartialDischarge1_1.wav");
x = x(:);   % 确保为列向量

%% STFT 参数
win = hamming(1024,"symmetric");   % 窗函数
nfft = 4096;                      % FFT 点数
hop = 512;                        % 帧移

%% 计算 STFT
[S, F, T] = spectrogram(x, win, length(win)-hop, nfft, fs);

%% 绘制幅度谱（dB）
figure;
% imagesc(T, F, 20*log10(abs(S)));
imagesc(T, F, abs(S));

axis xy;
ylim([0 1000])
xlabel("Time (s)");
ylabel("Frequency (Hz)");
title("STFT Spectrogram");
colorbar;
