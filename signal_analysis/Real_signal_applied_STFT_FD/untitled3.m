%% 固定窗 VS STFT-FD

clear; clc;

%% 读取你的真实信号
% [x, fs] = audioread("DCBias2_1.wav");
% [x, fs] = audioread("G4_10pSeventhHarmonic_1.wav");
% [x, fs] = audioread("Loosen1_1.wav");
[x, fs] = audioread("Normal_part1.wav");
% [x, fs] = audioread("PartialDischarge1_1.wav");

%% 设置 STFT-FD 参数
NC = 4;                         % 周期数
x = x(:);                  % 确保列向量
N = length(x);
strat_fre = 10;
end_fre = 1000;
fre_point = 990;    % 频率分辨率的数量（非 STFT 分辨率）Δf=(500-10)/300=1.63Hz
FreqList = linspace(strat_fre, end_fre, fre_point);  

% % 如信号是低频为主，最好的频率列表是对数分布
% FreqList = logspace(log10(5), log10(500), 300);

% % 按照电网频率的整数倍
% f0 = 50; % 电网频率
% FreqList = (1:20) * f0;   % 50, 100, ..., 1000

Tpos = 1000 : 20 : (N-1000);    % 时间中心点（样本）

% %% ================== 4. 固定窗 STFT（固定 256 点） ==================
% N_fixed = 256;              % 固定窗长，可根据需要更改
% w_fixed = hamming(N_fixed, "periodic");
% overlap_fixed = N_fixed/2;
% FFT_len = 4096;
% 
% [S_fixed, F_fixed, T_fixed] = stft(x, fs, ...
%     "Window", w_fixed, ...
%     "OverlapLength", overlap_fixed, ...
%     "FFTLength", FFT_len);

%% ================== 4. 固定窗 STFT ==================
%% STFT 参数

N_fixed = 1024;              % 固定窗长，可根据需要更改
w_fixed = hamming(N_fixed, "symmetric");
overlap_fixed = N_fixed/2;
FFT_len = 4096;

%% 计算 STFT
[S_fixed, F_fixed, T_fixed] = spectrogram(x, w_fixed, ...
    length(w_fixed)-overlap_fixed, FFT_len, fs);

%% 调用函数
[TF, F, T] = stft_fd(x, fs, Tpos, FreqList, NC);

%% 绘图
figure;
% 20*log10(abs(...)+eps) 更便于看弱信号；线性幅值便于看真实幅度
imagesc(Tpos/fs, F, abs(TF));
axis xy;
xlabel("Time (s)");
ylabel("Frequency (Hz)");
title("STFT-FD Time-Frequency Representation");
colorbar;


%% -------（B）固定窗 STFT 结果 -------
figure;
imagesc(T_fixed, F_fixed, abs(S_fixed));
% imagesc(T_fixed, F_fixed, 20*log10(abs(S_fixed)+eps));
axis xy; colormap jet; colorbar;
title("固定窗 STFT（256 点汉明窗）");
ylim([strat_fre end_fre])
xlabel("Time (s)");
ylabel("Frequency (Hz)");

%% -------（C）低频放大对比（同一频率范围） -------
fmax = 500;   % 根据需要调整

figure;
subplot(2,1,1);
imagesc(Tpos/fs, F, 20*log10(abs(TF)+eps));
% imagesc(Tpos/fs, F, abs(TF));
axis xy; ylim([strat_fre fmax]);
colormap jet; colorbar;
title("STFT-FD");
xlabel("Time (s)");

subplot(2,1,2);
imagesc(T_fixed, F_fixed, 20*log10(abs(S_fixed)+eps));
% imagesc(T_fixed, F_fixed, abs(S_fixed));
axis xy; 
ylim([strat_fre fmax]);
colormap jet; colorbar;
title("固定窗 STFT");
xlabel("Time (s)");

