%% 已知麦克风阵列真实坐标，可以直接读取坐标。
%% 设置三组实验：目标正弦信号、目标正弦信号+干扰信号+噪声、MVDR处理后的目标正弦信号+干扰信号+噪声

% Step 1. 读取麦克风阵列坐标
% Step 2. 设置目标与干扰方向
% Step 3. 生成信号（正弦 + 干扰 + 噪声）
% Step 4. 各阵元接收到的信号叠加（根据延时）
% Step 5. MVDR 波束形成
% Step 6. 对比波形与频谱

%% 读取麦克风阵列坐标
data = readmatrix('mic_positions.xlsx');    % 每一行表示一个麦克风的位置（X, Y 坐标）
X = data(:,1);
Y = data(:,2);
Z = zeros(size(X));
mic_pos = [X Y Z];  % Nmic × 3 矩阵，每行对应一个麦克风的三维坐标
Nmic = size(mic_pos, 1);    % 阵元数量（即麦克风总数）

%% 检查阵列布局
figure(1);
scatter(X,Y,40,'filled');
hold on;
% plot(X,Y,'-k','LineWidth',0.5); % 连线显示阵列形状
axis equal;     % 保持比例
grid on;
xlabel('X (m)'); 
ylabel('Y (m)');
title('Microphone Array Geometry');


%% 参数设置
c = 340;
fs = 16000;
t = (0:1/fs:1)';
f0 = 1000;  % 目标信号频率
f1 = 2000;  % 干扰信号频率
SNR = 10;
INR = 10;   % 干扰比
Nfft = 4096;          % FFT点数 用于频谱计算
f = (0:Nfft-1)*(fs/Nfft);  % 频率坐标

%% 定义信号源方向向量
theta_target = 0;   % 目标信号方位角
theta_interf = 60;  % 干扰方位角
% 将角度转为弧度
deg2rad = @(x) x*pi/180;

% 定义方向单位向量 指示声波入射方向（DOA）
d_target = [cos(deg2rad(theta_target)); sin(deg2rad(theta_target)); 0];
d_interf = [cos(deg2rad(theta_interf)); sin(deg2rad(theta_interf)); 0];

%% 生成目标信号与干扰信号
x_target = sin(2*pi*f0*t);  % 目标信号：1kHz 正弦波
x_interf = sin(2*pi*f1*t);  % 干扰信号：2kHz 正弦波

%% 计算阵列接收信号（延时叠加）
% 对于每个麦克风，根据声源方向和阵列几何，计算传播延时 τ = r·d / c
% 使用 interp1 对信号进行时间偏移（模拟波前到达的延迟）
X_target = zeros(length(t), Nmic);
X_interf = zeros(length(t), Nmic);

for m = 1:Nmic
    % 计算每个麦克风的传播延时
    tau_t = (mic_pos(m,:) * d_target) / c;
    tau_i = (mic_pos(m,:) * d_interf) / c;

    % 通过延时插值模拟波达时间差
    X_target(:,m) = interp1(t, x_target, t-tau_t,'linear',0);
    X_interf(:,m) = interp1(t, x_interf, t-tau_i,'linear',0);
end

% 添加高斯白噪声
noise = randn(size(X_target));
noise = noise ./ norm(noise) * norm(X_target) / (10^(SNR/20));  % 根据 SNR 缩放噪声能量

% 合成接收信号矩阵
X_noisy = X_target + X_interf / (10^(INR/20)) + noise;  % 干扰信号也按 INR 缩放

%% MVDR波束形成
% 计算阵列导向矢量
a_target = exp(-1j*2*pi*f0*(mic_pos*d_target)/c);   % 阵列导向矢量，描述目标方向的相位差

% 接收信号协方差矩阵估计
Rxx = (X_noisy.' * X_noisy) / size(X_noisy,1);

% MVDR权值
w_mvdr = (Rxx \ a_target) / (a_target' * (Rxx \ a_target));

% 波束形成输出
y_mvdr = real(X_noisy * w_mvdr);    % 将所有麦克风信号加权叠加，输出为对目标方向增强、对干扰方向抑制的信号

%% 波形对比
figure(2);
subplot(3,1,1);
% plot(t, X_target(:,1), 'b');    % 只画第一个麦克风的接收信号
plot(t,X_target,'b');
title('原始目标信号');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(3,1,2);
plot(t,X_noisy(:,1), 'r');
title('目标+干扰+噪声');
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
figure(3);
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
title('目标+干扰+噪声信号幅频图');
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

%% 频谱图对比
figure(4);
subplot(3,1,1);
spectrogram(X_target(:,1), hamming(256), 128, 512, fs, 'yaxis');
title('目标信号频谱图');
colormap jet;
colorbar;

subplot(3,1,2);
spectrogram(X_noisy(:,1), hamming(256), 128, 512, fs, 'yaxis');
title('目标+干扰+噪声频谱图');
colormap jet;
colorbar;

subplot(3,1,3);
spectrogram(y_mvdr, hamming(256), 128, 512, fs, 'yaxis');
title('MVDR输出频谱图');
colormap jet;
colorbar;

exportgraphics(figure(1), 'array_geometry.png', 'Resolution', 600);
exportgraphics(figure(2), 'waveforms.png', 'Resolution', 600);
exportgraphics(figure(3), 'amplitude_frequency.png', 'Resolution', 600);
exportgraphics(figure(4), 'spectrogram.png', 'Resolution', 600);