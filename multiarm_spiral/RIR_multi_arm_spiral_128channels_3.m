%% 模拟一个麦克风阵列的声场，生成目标信号与干扰（含混响），在阵列上叠加噪声后对每个通道做 STFT，
% 然后按频率做频域 MVDR 波束形成，其中协方差矩阵使用 多帧平均（multi-frame averaging） 来平滑估计，
% 最后用 istft 重构时域输出并可视化结果


%% ====================== 1. 读取麦克风阵列坐标 ======================
data = readmatrix('mic_positions.xlsx');
X = data(:,1);  % 读取表格中第一列为横坐标
Y = data(:,2);  % 第二列为纵坐标
Z = zeros(size(X));
mic_pos = [X Y Z];
Nmic = size(mic_pos, 1);    % 每行是一个麦克风的坐标

figure(1);
scatter(X, Y, 40, 'filled');
axis equal; grid on;
xlabel('X (m)'); ylabel('Y (m)');
title('Microphone Array Geometry');

%% ====================== 2. 参数设置 ======================
c = 340;
fs = 16000;
% t = (0:1/fs:1)，由于包含了0和1/fs，故产生的是1s + 一个样点
t = (0:1/fs:1-1/fs)';       
f0 = 1000; f1 = 2000;  % 目标与干扰频率
SNR = 10; INR = 10;

theta_target = 0;      % 目标方向
theta_interf = 60;     % 干扰方向
deg2rad = @(x) x*pi/180;
d_target = [cos(deg2rad(theta_target)); sin(deg2rad(theta_target)); 0]; % 平面波传播方向的单位向量
d_interf = [cos(deg2rad(theta_interf)); sin(deg2rad(theta_interf)); 0];

%% ====================== 3. 房间参数与RIR缓存 ======================
L = [5 4 6]; beta = 0.4; nsample = 4096;
mtype = 'hypercardioid'; % 单个麦克风形状
order = -1; % reflection order
dim = 3; 
orientation = [pi/2 0]; % microphone orientation(rad)
hp_filter = 1;  % 高通滤波

s_target = [2 3.5 2];
s_interf = [2.5 3.8 2];

fprintf('Generating representative RIRs...\n');
% 为加速，仅对阵列中心点 r_center 计算 RIR，后续对所有通道都用同一 RIR 做卷积
% 相当于假设混响是场内均匀的（或 RIR 对不同麦克风差异可忽略）
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

% 把相同混响信号按照不同麦克风的位置引入相对时延，生成每个麦克风接收到的信号
for m = 1:Nmic
    tau_t = (mic_pos(m,:) * d_target) / c;  % 传播时间
    tau_i = (mic_pos(m,:) * d_interf) / c;
    % 用线性插值实现小数采样点延时，边界处用 0 填充
    X_target(:,m) = interp1(t, x_target_rir, t-tau_t, 'linear', 0); 
    X_interf(:,m) = interp1(t, x_interf_rir, t-tau_i, 'linear', 0);
end

% 加噪声
noise = randn(size(X_target));
noise = noise ./ norm(noise) * norm(X_target) / (10^(SNR/20));
X_noisy = X_target + X_interf/(10^(INR/20)) + noise;

%% ====================== 频域MVDR波束形成（多帧平均） ======================
fprintf('Performing frequency-domain MVDR with multi-frame averaging...\n');

% === 1. 参数 ===
Nfft = 1024;
win = hamming(512);
noverlap = 256;

% === 2. STFT: 对每个麦克风做短时傅里叶变换 ===
for m = 1:Nmic
    [S(:,:,m), F, T] = stft(X_noisy(:,m), fs, 'Window', win, 'OverlapLength', noverlap, 'FFTLength', Nfft);
end
% S: F_bins × Time_frames × Nmic

Fbins = length(F);
Tframes = length(T);

% === 3. 计算导向矢量 ===
a_f = zeros(Nmic, Fbins);
for k = 1:Fbins
    freq = F(k);
    a_f(:,k) = exp(-1j*2*pi*freq*(mic_pos*d_target)/c);
end

% === 4. 初始化输出 ===
Yf = zeros(Fbins, Tframes);

% === 5. 多帧平均协方差矩阵 ===
Mavg = 5;  % 平均的时间帧数（可调 3~10）

for k = 1:Fbins
    % 获取第 k 个频率下的所有帧信号（Nmic × Tframes）
    Xkf = squeeze(S(k,:,:)).';  % Nmic × Tframes
    
    for t_idx = 1:Tframes
        % 当前帧的邻域范围（防越界）
        t1 = max(1, t_idx - floor(Mavg/2));
        t2 = min(Tframes, t_idx + floor(Mavg/2));
        X_local = Xkf(:, t1:t2);
        
        % 协方差矩阵（多帧平均）
        Rxx = (X_local * X_local') / size(X_local,2) + 1e-6*eye(Nmic);
        
        % MVDR 权值
        w_mvdr = (Rxx \ a_f(:,k)) / (a_f(:,k)' * (Rxx \ a_f(:,k)));
        
        % 计算输出
        Yf(k, t_idx) = w_mvdr' * Xkf(:, t_idx);
    end
end

% === 6. 逆STFT重建信号 ===
y_mvdr = istft(Yf, fs, 'Window', win, 'OverlapLength', noverlap, 'FFTLength', Nfft);
y_mvdr = real(y_mvdr);

fprintf('MVDR (multi-frame) beamforming completed.\n');


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
