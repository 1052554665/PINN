%% 已知麦克风阵列真实坐标，可以直接读取坐标。
%% 设置三组实验：目标正弦信号、目标正弦信号+干扰信号+噪声、MVDR处理后的目标正弦信号+干扰信号+噪声
%% 同时在信号传播中加入室内脉冲响应（RIR）

%% 读取麦克风阵列坐标
data = readmatrix('mic_positions.xlsx');
X = data(:,1);
Y = data(:,2);
Z = zeros(size(X));
mic_pos = [X Y Z];
Nmic = size(mic_pos, 1);

%% 可视化阵列几何结构
fig1 = figure(1);
scatter(X, Y, 40, 'filled');
axis equal;
grid on;
xlabel('X (m)');
ylabel('Y (m)');
title('Microphone Array Geometry');


%% 参数设置
c = 340;
fs = 16000;
t = (0:1/fs:1)';
f0 = 1000;
f1 = 2000;
SNR = 10;
INR = 10;
Nfft = 4096;
f = (0:Nfft-1)*(fs/Nfft);

theta_target = 0;
theta_interf = 60;

deg2rad = @(x) x*pi/180;
d_target = [cos(deg2rad(theta_target)); sin(deg2rad(theta_target)); 0];
d_interf = [cos(deg2rad(theta_interf)); sin(deg2rad(theta_interf)); 0];

%% 房间参数与RIR生成
L = [5 4 6];
beta = 0.4;
nsample = 4096;
mtype = 'hypercardioid' % 麦克风的指向性类型
order = -1;
dim = 3;
orientation = [pi/2 0];
hp_filter = 1;

% 定义信号源和麦克风中心位置
s_target = [2 3.5 2];   % 目标源位置
s_interf = [2.5 3.8 2]; % 干扰源位置

%% 生成目标和干扰信号
x_target = sin(2*pi*f0*t);
x_interf = sin(2*pi*f1*t);

%% 模拟阵列接收信号（含延时与混响）
X_target = zeros(length(t), Nmic);
X_interf = zeros(length(t), Nmic);

for m = 1:Nmic
    %% 对每个麦克风独立生成一个 RIR（因为每个位置不同，房间反射不同）。
    % 生成目标信号的RIR
    h_t = rir_generator(c, fs, mic_pos(m,:), s_target, L, beta, ...
        nsample, mtype, order, dim, orientation, hp_filter);
    % 为干扰源生成RIR
    h_i = rir_generator(c, fs, mic_pos(m,:), s_interf, L, beta, ...
        nsample, mtype, order, dim, orientation, hp_filter);
    
    % 延时信号（平面波入射）
    tau_t = (mic_pos(m,:) * d_target) / c;
    tau_i = (mic_pos(m,:) * d_interf) / c;
    x_t_delayed = interp1(t, x_target, t-tau_t, 'linear', 0);
    x_i_delayed = interp1(t, x_interf, t-tau_i, 'linear', 0);

    % 卷积加混响
    y_t = conv(x_t_delayed, h_t, 'full');
    y_i = conv(x_i_delayed, h_i, 'full');

    % 若卷积结果过短，补零；若过长，截取
    len_t = min(length(y_t), length(t));
    len_i = min(length(y_i), length(t));

    X_target(1:len_t, m) = y_t(1:len_t);
    X_interf(1:len_i, m) = y_i(1:len_i);
end

% 添加高斯噪声
noise = randn(size(X_target));
noise = noise ./ norm(noise) * norm(X_target) / (10^(SNR/20));

% 合成信号
X_noisy = X_target + X_interf / (10^(INR/20)) + noise;

%% MVDR波束形成
a_target = exp(-1j*2*pi*f0*(mic_pos*d_target) / c);
Rxx = (X_noisy.' * X_noisy) / size(X_noisy, 1);
w_mvdr = (Rxx \ a_target) / (a_target' * (Rxx \ a_target));
y_mvdr = real(X_noisy * w_mvdr);

%% 波形对比
fig2 = figure(2);
subplot(3,1,1);
plot(t, X_target(:,1), 'b');
title('原始目标信号（含混响）'); 
xlabel('Time (s)'); 
ylabel('Amplitude'); 
grid on;

subplot(3,1,2);
plot(t, X_noisy(:,1), 'r');
title('目标+干扰+噪声（含混响）'); 
xlabel('Time (s)'); 
ylabel('Amplitude'); 
grid on;

subplot(3,1,3);
plot(t, y_mvdr, 'k');
title('MVDR输出信号'); 
xlabel('Time (s)'); 
ylabel('Amplitude'); 
grid on;

%% 幅频图对比
fig3 = figure(3);
subplot(3,1,1);
plot(f, abs(fft(x_target, Nfft)), 'b'); 
xlim([0 fs/2]);
title('目标信号幅频图'); 
xlabel('Frequency (Hz)'); 
ylabel('Amplitude'); 
grid on;

subplot(3,1,2);
plot(f, abs(fft(X_noisy(:,1), Nfft)), 'r'); 
xlim([0 fs/2]);
title('目标+干扰+噪声幅频图'); 
xlabel('Frequency (Hz)'); 
ylabel('Amplitude'); 
grid on;

subplot(3,1,3);
plot(f, abs(fft(y_mvdr, Nfft)), 'k'); 
xlim([0 fs/2]);
title('MVDR输出幅频图'); 
xlabel('Frequency (Hz)'); 
ylabel('Amplitude'); 
grid on;


%% 频谱图对比（STFT）
fig4 = figure(4);
subplot(3,1,1);
spectrogram(X_target(:,1), hamming(256), 128, 512, fs, 'yaxis');
title('目标信号频谱图（含混响）');
colormap jet; colorbar;

subplot(3,1,2);
spectrogram(X_noisy(:,1), hamming(256), 128, 512, fs, 'yaxis');
title('目标+干扰+噪声频谱图');
colormap jet; colorbar;

subplot(3,1,3);
spectrogram(y_mvdr, hamming(256), 128, 512, fs, 'yaxis');
title('MVDR输出频谱图');
colormap jet; colorbar;

exportgraphics(fig1, 'array_geometry.png', 'Resolution', 600);
exportgraphics(fig2, 'waveforms.png', 'Resolution', 600);
exportgraphics(fig3, 'amplitude_frequency.png', 'Resolution', 600);
exportgraphics(fig4, 'spectrograms.png', 'Resolution', 600);