%% 0. 加载或生成电压信号（示例：60Hz + 电压跌落）
fs = 2560;                       % 采样率（电力信号常用 2–10kHz）
t = (0:1/fs:1.4).';
f0 = 60;                         % 基频
x = (1 - 0.5*(t>0.2 & t<0.8)) ...% 电压跌落（sag）
    .* sin(2*pi*f0*t);           % 电压信号

%% 1. STFT 参数
L = 256;                         % 窗长（论文中参数）
window = hamming(L);
Nfft = 1024;
hop = L/4;                       % 常用 hop

%% 2. 计算 STFT
[S, F, T] = stft(x, fs, ...
                 "Window", window, ...
                 "FFTLength", Nfft, ...
                 "OverlapLength", L-hop);

%% 3. 选择谐波对应的 STFT bin
harmonics = [1 3 5 7];           % 奇次谐波
k = round(harmonics * f0 / (fs/Nfft));  % 对应的频率 bin

%% 4. 求每个谐波的复数带通幅度
figure;
subplot(5,1,1)
plot(t, x); title("Input signal (voltage sag)"); grid on;

for i = 1:length(k)
    subplot(5,1,i+1)
    mag = abs(S(k(i), :));       % STFT 对应频率 bin 的幅值
    plot(T*fs, mag, 'LineWidth', 1.5);
    title(sprintf("%d^{th} harmonic magnitude", harmonics(i)));
    grid on;
end
