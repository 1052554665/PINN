%% Underbrink-like 多臂螺旋阵列 + MVDR 仿真 (窄带示例)
% 36 元：中心 6 个，6 臂每臂 5 个

%% ====== 阵列几何 ======
M = 6;  % 螺旋臂数
center_ring_N = 6;  % 中心环6个阵元
arm_elem = 5;   % 每臂5个阵元
N = center_ring_N + M * arm_elem    % 总元数

% 几何参数
r_center = 0.01;   % 中心小环半径 (m)
arm_spacing = 0.02; % 臂上相邻阵元沿半径方向间距 (m)
R_max = r_center + arm_spacing * arm_elem;

pos = zeros(3,N);
idx = 1;

% 1) 中心环（6点） 均匀绕一个小圆
for n = 1:center_ring_N
    ang = 2 * pi * (n-1) / center_ring_N;
    pos(:,idx) = [r_center * cos(ang); r_center * sin(ang); 0];
    idx = idx + 1;
end

% 2) 多臂，每臂 arm_elem 个，交错排列（从内向外）
for m = 0:M-1
    base_ang = 2 * pi * m / M;
    for j = 1:arm_elem
        rj = r_center + j * arm_spacing;    % 往外延伸
        ang = base_ang + 0.15 * j;  % 给点小的螺旋弯曲 (可选)
        pos(:,idx) = [rj*cos(ang); rj*sin(ang);0];
        idx = idx + 1;
    end
end

% 绘图：阵列结构
figure('Name','Array geometry');
scatter(pos(1,:), pos(2,:), 60, 'k', 'filled'); hold on;
% 标记每臂连线（按臂抽取）
for m = 0:M-1
    idxs = (center_ring_N+1 + m*arm_elem) : (center_ring_N + (m+1)*arm_elem);
    plot(pos(1, idxs), pos(2, idxs), '-','Color',[0.6 0.6 0.6]);
end
axis equal; grid on; xlabel('x (m)'); ylabel('y (m)');
title(sprintf('Array geometry: total=%d, center=%d, M=%d, armElem=%d', N, center_ring_N, M, arm_elem));

%% ====== 信号场景（窄带） ======
fs = 16000;          % 采样率
T = 1.0;             % 仿真时长 (s)
t = (0:1/fs:T-1/fs).';
Nsamp = length(t);

% 频率（窄带中心频率）
fc = 2000;           % 中心频率 2 kHz（可改）
c = 343;             % 声速 (m/s)
lambda = c/fc;
k = 2*pi/lambda;

% 源设置：1 个期望源 + 2 个干扰源
src_desired_az = 20;            % 方位角 (deg)，0 为 x 轴正向
src_desired_el = 0;             % 俯仰 (deg)
src_interf_az = [-80, 60];      % 两个干扰角
src_interf_el = [0, 0];


% 合成窄带信号（幅度包络）
sig_env = hann(Nsamp*2); sig_env = sig_env(1:Nsamp);   % 一个平滑的包络
s_desired = real(sig_env .* exp(1j*2*pi*fc*t));       % 复数窄带信号 -> 实部为传感器观测
s_interf1 = 0.6 * real(sig_env .* exp(1j*2*pi*(fc+50)*t)); % 干扰可稍微偏频
s_interf2 = 0.5 * real(sig_env .* exp(1j*2*pi*(fc-80)*t));

% 背景噪声
SNR_db = 20;   % 期望源到噪声比（参考）
sigma_noise = 10^(-SNR_db/20);


% 生成阵列接收：平面波模型 (窄带)
% 方向到单位矢量函数（水平面）
az_rad = deg2rad(src_desired_az);
u_des = [cos(az_rad); sin(az_rad); 0];

% 干扰单位向量
u_interf = zeros(3, length(src_interf_az));
for ii = 1:length(src_interf_az)
    az_i = deg2rad(src_interf_az(ii));
    u_interf(:,ii) = [cos(az_i); sin(az_i); 0];
end

% 构造接收矩阵 X (N x Nsamp)
X = zeros(N, Nsamp);
for n = 1:N
    rn = pos(:, n);  % 3x1
    % 相位项 exp(-j k r·u) (注意物理符号习惯，这里选用 +j*k*...)
    % 用 exp(1j*k*(r_n·u)) 保持与前面阵列因子一致
    phase_des = exp(1j * k * (rn.' * u_des));
    % desired signal across array
    xr_des = real(phase_des .* (s_desired.'));   % 1 x Nsamp
    % interference contributions
    xr_if = zeros(1, Nsamp);
    xr_if = xr_if + real(exp(1j * k * (rn.' * u_interf(:,1))) .* (s_interf1.'));
    xr_if = xr_if + real(exp(1j * k * (rn.' * u_interf(:,2))) .* (s_interf2.'));
    % noise
    noise = sigma_noise * randn(1, Nsamp);
    X(n, :) = xr_des + xr_if + noise;
end

% 选取参考通道（例如阵列中心第 1 个中心环元）
ref_chan = 1;
ref_signal = X(ref_chan, :);

%% ====== 计算样本协方差矩阵并求 MVDR 权重（窄带） ======
% 这里用样本协方差: R = (1/Ts) * X * X^H
R = (X * X') / Nsamp;    % N x N
% 波束指向向量 a(theta) (窄带) -- 注意与生成时相位一致
a_des = zeros(N,1);
for n = 1:N
    a_des(n) = exp(1j * k * (pos(:,n).' * u_des));
end
% 对协方差做对角加载以获得数值稳定性
delta = 1e-6 * trace(R)/N;
Rld = R + delta * eye(N);

% MVDR 权重
w_mvdr = (Rld \ a_des) / (a_des' * (Rld \ a_des));   % N x 1

% 波束形成输出 y(t) = w^H * x(t) (逐时刻)
y_mvdr = (w_mvdr' * X).';   % Nsamp x 1 (注意转置)
% 如果想要保持实数信号，取实部
y_mvdr = real(y_mvdr);

%% ====== 绘图：时域波形对比 ======
t_plot = t(1:round(0.05*fs));  % 取前 50 ms 作示例
figure('Name','Time domain comparison');
plot(t_plot, ref_signal(1:length(t_plot)), 'b', 'DisplayName','Ref channel (raw)'); hold on;
plot(t_plot, y_mvdr(1:length(t_plot)), 'r', 'LineWidth',1.2, 'DisplayName','MVDR output');
xlabel('Time (s)'); ylabel('Amplitude');
legend; grid on;
title('MVDR beamforming: raw reference vs MVDR output (time domain)');

%% ====== 绘图：更长时间的波形和 PSD 对比（可选） ======
% 长时间波形（整段）比较（规范化显示）
figure('Name','Long time series (normalized)');
sig1 = ref_signal / max(abs(ref_signal));
sig2 = y_mvdr.' / max(abs(y_mvdr));
subplot(2,1,1); plot(t, sig1); title('Reference channel (normalized)'); xlim([0 0.2]);
subplot(2,1,2); plot(t, sig2); title('MVDR output (normalized)'); xlim([0 0.2]);

% 频谱对比
nfft = 2048;
S_ref = fftshift(abs(fft(ref_signal, nfft)));
S_out = fftshift(abs(fft(y_mvdr, nfft)));
fvec = linspace(-fs/2, fs/2, nfft);
figure('Name','Spectrum comparison');
plot(fvec, 20*log10(S_ref + eps), 'b', fvec, 20*log10(S_out + eps), 'r');
legend('Ref', 'MVDR'); xlabel('Frequency (Hz)'); ylabel('Magnitude (dB)');
title('Spectrum: reference vs MVDR output (linear array)'); grid on;
xlim([fc-1000, fc+1000]);