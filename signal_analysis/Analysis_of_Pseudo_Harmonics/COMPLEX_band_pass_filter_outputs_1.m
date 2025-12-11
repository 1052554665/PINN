%% 0. 读取你的真实信号
% [x, fs] = audioread("DCBias2_1.wav");   % 写你的 wav 文件名
% [x, fs] = audioread("G4_10pSeventhHarmonic_1.wav");   % 写你的 wav 文件名
% [x, fs] = audioread("Loosen1_1.wav");
% [x, fs] = audioread("Normal_part1.wav");
[x, fs] = audioread("PartialDischarge1_1.wav");


% 如果是立体声，取一个通道
if size(x,2) > 1
    x = x(:,1);
end

x = x(:);   % 转成列向量
t = (0:length(x)-1)'/fs;


%% 1. STFT 参数
L = 256;                   % 窗长
hop = L/4;                 % 步长
win = hamming(L,"periodic");

%% 2. 做 STFT
[S, F, T] = stft(x, fs, "Window", win, "OverlapLength", L-hop, "FFTLength", L);

%% 3. 提取奇次谐波：1、3、5、7 次（图中的 1,3,5,7）
f0 = 50;                          % 基频
harmonics = [1 3 5 7];
mag_h = zeros(length(harmonics), length(T));

for k = 1:length(harmonics)
    target_freq = harmonics(k)*f0;
    [~, idx] = min(abs(F-target_freq));
    mag_h(k,:) = abs(S(idx,:));   % 复数带通滤波器输出的模值
end

%% 4. 绘图
figure;
subplot(5,1,1)
plot(t, x); title("真实输入电压信号")

for k = 1:length(harmonics)
    subplot(5,1,k+1)
    plot(T, mag_h(k,:));
    title(sprintf("%d 次谐波滤波器输出幅值", harmonics(k)));
end
xlabel("time")

