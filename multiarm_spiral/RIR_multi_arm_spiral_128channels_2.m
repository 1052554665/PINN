%% 


%% ====================== 1. 读取麦克风阵列坐标 ======================
data = readmatrix('mic_positions.xlsx');
X = data(:,1);
Y = data(:,2);
Z = zeros(size(X));
mic_pos = [X Y Z];
Nmic = size(mic_pos, 1);


% 麦克风阵列的原点放在RIR

figure(1);
scatter(X, Y, 40, 'filled');
axis equal; grid on;
xlabel('X (m)'); ylabel('Y (m)');
title('Microphone Array Geometry');

%% ====================== 2. 参数设置 ======================
c = 340;
fs = 16000;
t = (0:1/fs:1)';       % 1 秒信号
f0 = 1000; f1 = 2000;  % 目标与干扰频率
SNR = 10; INR = 10;
Nfft = 4096;

theta_target = 0;      % 目标方向
theta_interf = 60;     % 干扰方向
deg2rad = @(x) x*pi/180;
d_target = [cos(deg2rad(theta_target)); sin(deg2rad(theta_target)); 0];
d_interf = [cos(deg2rad(theta_interf)); sin(deg2rad(theta_interf)); 0];

%% ====================== 3. 房间参数与RIR缓存 ======================
L = [5 4 6]; beta = 0.4; nsample = 4096;
mtype = 'hypercardioid';
order = -1; dim = 3; orientation = [pi/2 0]; hp_filter = 1;

s_target = [2 3.5 2];
s_interf = [2.5 3.8 2];

fprintf('Generating representative RIRs...\n');
% 仅计算中心点RIR作为卷积模板，加速
r_center = mean(mic_pos, 1);
h_target = rir_generator(c, fs, r_center, s_target, L, beta, ...
    nsample, mtype, order, dim, orientation, hp_filter);
h_interf = rir_generator(c, fs, r_center, s_interf, L, beta, ...
    nsample, mtype, order, dim, orientation, hp_filter);

fprintf('RIR precomputation complete.\n');

%% ====================== 4. 生成目标与干扰信号 ======================
x_target = sin(2*pi*f0*t);
x_interf = sin(2*pi*f1*t);

% 混响信号（统一使用中心RIR）
x_target_rir = conv(x_target, h_target, 'same');
x_interf_rir = conv(x_interf, h_interf, 'same');

%% ====================== 5. 模拟阵列信号（仅延时，无重复RIR） ======================
X_target = zeros(length(t), Nmic);
X_interf = zeros(length(t), Nmic);

for m = 1:Nmic
    tau_t = (mic_pos(m,:) * d_target) / c;
    tau_i = (mic_pos(m,:) * d_interf) / c;
    X_target(:,m) = interp1(t, x_target_rir, t-tau_t, 'linear', 0);
    X_interf(:,m) = interp1(t, x_interf_rir, t-tau_i, 'linear', 0);
end

% 加噪声
noise = randn(size(X_target));
noise = noise ./ norm(noise) * norm(X_target) / (10^(SNR/20));

X_noisy = X_target + X_interf/(10^(INR/20)) + noise;

%% ====================== 6. 频域MVDR波束形成 ======================
fprintf('Performing frequency-domain MVDR...\n');

% FFT转换
Xf = fft(X_noisy, Nfft);  % Nfft × Nmic
f_bins = (0:Nfft-1)*(fs/Nfft);

% 导向矢量（每个频点）
a_f = zeros(Nmic, Nfft);
for k = 1:Nfft
    freq = f_bins(k);
    a_f(:,k) = exp(-1j*2*pi*freq*(mic_pos*d_target)/c);
end

Yf = zeros(Nfft,1);
for k = 1:Nfft
    Xk = squeeze(Xf(k,:)).';
    Rxx = (Xk*Xk')/Nmic + 1e-6*eye(Nmic); % 正则化防止病态
    w_mvdr = (Rxx \ a_f(:,k)) / (a_f(:,k)' * (Rxx \ a_f(:,k)));
    Yf(k) = w_mvdr' * Xk;
end

y_mvdr = real(ifft(Yf, Nfft));
fprintf('MVDR beamforming done.\n');

%% ====================== 7. 可视化 ======================
figure(2);
subplot(3,1,1);
plot(t, X_target(:,1)); title('目标信号（含混响）'); xlabel('Time'); ylabel('Amplitude'); grid on;
subplot(3,1,2);
plot(t, X_noisy(:,1)); title('目标+干扰+噪声'); xlabel('Time'); ylabel('Amplitude'); grid on;
subplot(3,1,3);
plot(y_mvdr); title('频域MVDR输出'); xlabel('Time'); ylabel('Amplitude'); grid on;

figure(3);
subplot(3,1,1); spectrogram(X_target(:,1), hamming(256), 128, 512, fs, 'yaxis');
title('目标信号频谱图'); colormap jet;
subplot(3,1,2); spectrogram(X_noisy(:,1), hamming(256), 128, 512, fs, 'yaxis');
title('目标+干扰+噪声'); colormap jet;
subplot(3,1,3); spectrogram(y_mvdr, hamming(256), 128, 512, fs, 'yaxis');
title('频域MVDR输出频谱图'); colormap jet;

fprintf('All processing complete.\n');
