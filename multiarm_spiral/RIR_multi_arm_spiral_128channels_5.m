%% 仅在阵列中心计算一次RIR，然后假设所有麦克风的混响响应相同

%% ====================== 1. 读取麦克风阵列坐标 ======================
data = readmatrix('mic_positions.xlsx');
X = data(:,1);  % 读取表格中第一列为横坐标
Y = data(:,2);  % 第二列为纵坐标
Z = zeros(size(X));
mic_pos = [X Y Z];

% 查看麦克风位置范围
disp([min(mic_pos(:,1)) max(mic_pos(:,1))])
disp([min(mic_pos(:,2)) max(mic_pos(:,2))])

% 如果Excel中麦克风单位为cm
% mic_pos = mic_pos / 100;   % cm → m
% 如果是mm，用 
mic_pos = mic_pos / 1000;

% === 设置阵列在房间中的中心位置 ===
array_center = [2.5 2 1.5];   % 房间内部的一个点 (x,y,z)，单位：米
% === 平移整个阵列，使阵列中心位于该位置 ===
mic_pos = mic_pos - mean(mic_pos, 1);   % 以阵列中心为参考
mic_pos = mic_pos + array_center;       % 移动到房间中心位置

Nmic = size(mic_pos, 1);    % 每行是一个麦克风的坐标

figure(1);
scatter(X, Y, 40, 'filled');
% scatter3(mic_pos(:,1), mic_pos(:,2), mic_pos(:,3), 40, 'filled'); % 三维阵列图
axis equal; grid on;
xlabel('X (m)'); ylabel('Y (m)');
title('Microphone Array Geometry');

%% ====================== 可视化房间、声源和麦克风阵列位置 ======================
figure(4); clf;
hold on; grid on; axis equal;

% 1. 绘制房间边界（矩形框）
L = [5 4 6];  % 房间长宽高
xv = [0 L(1) L(1) 0 0];
yv = [0 0 L(2) L(2) 0];
z0 = zeros(size(xv));

% 地面和顶面
plot3(xv, yv, z0, 'k--', 'LineWidth', 1.2);
plot3(xv, yv, L(3)*ones(size(z0)), 'k--', 'LineWidth', 1.2);

% 垂直边
for i = 1:4
    plot3([xv(i) xv(i)], [yv(i) yv(i)], [0 L(3)], 'k--');
end

% 2. 绘制麦克风阵列位置
scatter3(mic_pos(:,1), mic_pos(:,2), mic_pos(:,3), 60, 'filled', 'b');
text(mean(mic_pos(:,1)), mean(mic_pos(:,2)), mean(mic_pos(:,3))+0.1, 'Mic Array', ...
     'Color', 'b', 'FontWeight', 'bold');

% 3. 绘制目标声源与干扰源
s_target = [2 3.5 2];
s_interf = [2.5 3.8 2];

scatter3(s_target(1), s_target(2), s_target(3), 100, 'r', 'filled');
text(s_target(1), s_target(2), s_target(3)+0.1, 'Target Source', 'Color', 'r', 'FontWeight', 'bold');

scatter3(s_interf(1), s_interf(2), s_interf(3), 100, 'm', 'filled');
text(s_interf(1), s_interf(2), s_interf(3)+0.1, 'Interference', 'Color', 'm', 'FontWeight', 'bold');

% 4. 可选：连接阵列中心与声源方向
r_center = mean(mic_pos,1);
plot3([r_center(1) s_target(1)], [r_center(2) s_target(2)], [r_center(3) s_target(3)], 'r--', 'LineWidth', 1.2);
plot3([r_center(1) s_interf(1)], [r_center(2) s_interf(2)], [r_center(3) s_interf(3)], 'm--', 'LineWidth', 1.2);

% 5. 设置视角与标签
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('Room, Sources, and Microphone Array Layout');
view(45, 25);
legend({'Room boundary','Microphones','Target Source','Interference'}, 'Location','best');

%% ====================== 2. 参数设置 ======================
c = 340;
fs = 16000; % 采样率
% t = (0:1/fs:1)，由于包含了0和1/fs，故产生的是1s + 一个样点
t = (0:1/fs:1-1/fs)';       
f0 = 1000; f1 = 2000;  % 目标与干扰频率
SNR = 10; 
INR = 10;   % 干扰与目标的比率 dB 或干扰强度相对于目标的 dB，后续按 INR 控制干扰振幅

theta_target = 0;      % 目标方位角
theta_interf = 60;     % 干扰方向
deg2rad = @(x) x*pi/180;    % 用匿名函数实现度到弧度的转换
d_target = [cos(deg2rad(theta_target)); sin(deg2rad(theta_target)); 0]; % 平面波传播方向的单位向量
d_interf = [cos(deg2rad(theta_interf)); sin(deg2rad(theta_interf)); 0];

%% ====================== 3. 房间参数与RIR缓存 ======================
%  ========仅在阵列中心计算一次RIR，然后假设所有麦克风的混响响应相同=========
% beta = 0 或 order = 0 等同于不加混响
L = [5 4 6]; 
beta = 0.4; % 墙面反射系数（0=全吸收，1=全反射）
nsample = 4096;
mtype = 'hypercardioid'; % 单个麦克风形状
order = -1; % reflection order; -1 表示自动确定
dim = 3; 
orientation = [pi/2 0]; % microphone orientation(rad)
hp_filter = 1;  % 高通滤波（去除直流分量）

s_target = [2 3.5 2];   % 目标声源在房间内的位置坐标
s_interf = [2.5 3.8 2]; % 干扰声源在房间内的位置坐标

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
x_target_rir = conv(x_target, h_target, 'same');    % 结果长度返回中心对齐的部分，保留与 t 等长的信号
x_interf_rir = conv(x_interf, h_interf, 'same');

%% ====================== 5. 模拟阵列信号（仅延时，无重复RIR） ======================
X_target = zeros(length(t), Nmic);
X_interf = zeros(length(t), Nmic);

% 把相同混响信号按照不同麦克风的位置引入相对时延，生成每个麦克风接收到的信号
for m = 1:Nmic
    % 对第 m 个麦克风位置 mic_pos(m,:) 与方向向量的内积，除以 c 得到传播时间（秒）。
    % 注意符号/约定：若麦克风坐标越远，mic_pos * d 值越大，tau 越大，意味着该通道信号相对于参考更晚到达。
    % 该表达式隐含参考时间是原点；若期望以阵列中心为参考，需要做 mic_pos(m,:) - r_ref

    r_ref = mean(mic_pos, 1);        % 阵列中心位置
    tau_t = ((mic_pos(m,:) - r_ref) * d_target) / c;
    tau_i = ((mic_pos(m,:) - r_ref) * d_interf) / c;

    % tau_t = (mic_pos(m,:) * d_target) / c;  % 传播时间
    % tau_i = (mic_pos(m,:) * d_interf) / c;

    %% 用线性插值实现小数采样点延时，边界处用 0 填充
    % 对时间轴做插值，把信号整体向右/向左平移实现无整数采样点的时延。
    % 'linear' 做线性插值，最后一个参数 0 指定超出 t 范围的插值值为 0（边界补零）

    % 边界效应：当 t - tau 超出 [t(1), t(end)] 时会被填 0，会在信号起始或末尾造成零填充。
    % 若要避免，可以用循环延长信号或使用频域延时（相位旋转）实现环绕或更平滑边界
    X_target(:,m) = interp1(t, x_target_rir, t-tau_t, 'linear', 0); 
    X_interf(:,m) = interp1(t, x_interf_rir, t-tau_i, 'linear', 0);
end

% 按通道噪声 RMS 缩放（每个麦克风）
sig_rms = sqrt(mean(X_target.^2, 1)); % 1 x Nmic，计算每个通道目标信号的 RMS（均方根），用于确定噪声幅度
desired_noise_rms = sig_rms ./ (10^(SNR/20));   % 将 SNR 从 dB 转换为线性比值
noise = randn(size(X_target));  % 高斯白噪声（0 均值，单位方差）
cur_noise_rms = sqrt(mean(noise.^2, 1));    % 估计当前噪声的 RMS（每列），做逐列缩放
noise = noise .* (desired_noise_rms ./ cur_noise_rms); % each col scaled
% 干扰信号通过 X_interf/(10^(INR/20)) 缩小到设定 INR（dB）。
% 注意这里 INR 的定义看起来是干扰相对于“目标”的 dB，若需要定义不同需统一。
% 最终 X_noisy 包含目标、缩放的干扰与按通道噪声
X_noisy = X_target + X_interf/(10^(INR/20)) + noise;    


%% ====================== 频域MVDR波束形成（多帧平均） ======================
fprintf('Performing frequency-domain MVDR with multi-frame averaging (0–5 kHz)...\n');
% 设置 STFT 参数：FFT 长度、窗、重叠长度。win 长度 512，Nfft 1024 表示零填充以增加频率分辨率
Nfft = 1024;
win = hamming(512);
noverlap = 256;

% === 1. 对每个通道计算短时傅里叶变换（STFT） ===
for m = 1:Nmic
    [S(:,:,m), F, T] = stft(X_noisy(:,m), fs, 'Window', win, 'OverlapLength', noverlap, 'FFTLength', Nfft);
end
Fbins = length(F);
Tframes = length(T);

% === 2. 限制频率范围到 0–5 kHz ===
f_limit = find(F <= 5000, 1, 'last');    % 找到5 kHz对应频点
fprintf('Using frequency bins 1–%d (%.1f Hz–%.1f Hz)\n', f_limit, F(1), F(f_limit));

% === 3. 导向矢量 ===
% 这里没有包含频率相关的幅度衰减（比如距离衰减 1/r）——只包含相位差（平面波近似）。
% 如果需要近场或球面波，需要用 exp(-1j*2*pi*freq*r/c)./r 等带距离项的模型，并以 mic_pos 与参考点求距离 r
a_f = zeros(Nmic, Fbins);
for k = 1:Fbins
    freq = F(k);
    a_f(:,k) = exp(-1j*2*pi*freq*(mic_pos*d_target)/c);
end

% === 4. 初始化输出 ===
Yf = zeros(Fbins, Tframes); % 频域输出矩阵（频 × 帧）
Mavg = 5;    % 进行协方差估计时以5帧做平均（窗口长度）
reg = 1e-6;  % 数值正则化避免协方差奇异或求逆不稳定

% GPU加速
% S = gpuArray(S);
% a_f = gpuArray(a_f);


% === 5. 主循环（仅 0–5 kHz 范围） ===
tic
for k = 1:f_limit
    Xkf = squeeze(S(k,:,:)).';  % Nmic × Tframes
    Rxx_all = zeros(Nmic, Nmic, Tframes, 'like', Xkf);

    % --- (1) 批量构建多帧平均协方差矩阵 ---
    for t_idx = 1:Tframes
        t1 = max(1, t_idx - floor(Mavg/2));
        t2 = min(Tframes, t_idx + floor(Mavg/2));
        X_local = Xkf(:, t1:t2);
        Rxx_all(:,:,t_idx) = (X_local * X_local') / size(X_local,2) + reg*eye(Nmic);
    end

    % --- (2) 批量求解 Rxx \ a_f(:,k) ---
    a_k = repmat(a_f(:,k), 1, 1, Tframes);   % Nmic × 1 × Tframes
    w_tmp = pagemldivide(Rxx_all, a_k);      % Nmic × 1 × Tframes
    w_tmp = squeeze(w_tmp);                  % Nmic × Tframes
    
    denom = sum(conj(a_f(:,k)) .* w_tmp, 1); % 1 × Tframes
    w_tmp = w_tmp ./ denom;                  % Nmic × Tframes
    
    % --- (3) 批量计算输出 ---
    Yf(k,:) = sum(conj(w_tmp) .* Xkf, 1);

end
toc

% GPU加速
% Yf = gather(Yf);


% === 6. 逆STFT ===
y_mvdr = istft(Yf, fs, 'Window', win, 'OverlapLength', noverlap, 'FFTLength', Nfft);

% 能量补偿
win_gain = sum(win.^2) / (length(win) - noverlap);
y_mvdr = y_mvdr / win_gain;

y_mvdr = real(y_mvdr);

fprintf('MVDR (multi-frame, 0–5 kHz) beamforming completed.\n');


%% ====================== 7. 可视化 ======================
figure(2);
subplot(3,1,1);
plot(t, X_target(:,1)); title('目标信号（含混响）'); xlabel('Time'); ylabel('Amplitude'); grid on;
subplot(3,1,2);
plot(t, X_noisy(:,1)); title('目标+干扰+噪声'); xlabel('Time'); ylabel('Amplitude'); grid on;
subplot(3,1,3);
plot(y_mvdr); title('频域MVDR输出波形'); xlabel('Time'); ylabel('Amplitude'); grid on;

figure(3);
subplot(3,1,1); spectrogram(X_target(:,1), hamming(256), 128, 512, fs, 'yaxis');
title('目标信号频谱图'); colormap jet;
subplot(3,1,2); spectrogram(X_noisy(:,1), hamming(256), 128, 512, fs, 'yaxis');
title('目标+干扰+噪声'); colormap jet;
subplot(3,1,3); spectrogram(y_mvdr, hamming(256), 128, 512, fs, 'yaxis');
title('频域MVDR输出频谱图'); colormap jet;

fprintf('All processing complete.\n');




