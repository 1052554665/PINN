% 基于最小均方误差（MMSE）的自适应波束形成

% 参数设置
M = 18; % 阵元数量
lambda = 10; % 波长
d = lambda / 2; % 阵元间距
L = 100; % 采样点数
thetas = 10; % 目标信号方向
thetai = [-30, 30]; % 干扰信号方向

% 生成目标信号和干扰信号的导向矢量
n = (0:M-1)'; % 阵元索引
vs = exp(-1i * 2 * pi * n * d * sind(thetas) / lambda); % 目标信号导向矢量
vn1 = exp(-1i * 2 * pi * n * d * sind(thetai(1)) / lambda); % 干扰信号1导向矢量
vn2 = exp(-1i * 2 * pi * n * d * sind(thetai(2)) / lambda); % 干扰信号2导向矢量

% 生成信号和噪声
f = 1600; % 信号频率
t = (0:L-1); % 时间序列
di = sin(2 * pi * f * t / (8 * f)); % 目标信号
vn1_singal = sin(2 * pi * 2 * f * t / (8 * f)); % 干扰信号1
vn2_singal = sin(2 * pi * 4 * f * t / (8 * f)); % 干扰信号2
A = [vs, vn1, vn2]; % 信号矩阵
St = [di; vn1_singal; vn2_singal]; % 信号源
Xt = A * St + randn(M, L); % 接收到的信号

% 计算协方差矩阵
R_x = (1 / L) * (Xt * Xt'); % 接收到的信号的协方差矩阵
R_x_inv = inv(R_x); % 协方差矩阵的逆


% 计算最优权重
W_opt = R_x_inv * vs / (vs' * R_x_inv * vs);    % 最优权重向量

% 绘制波束图
sita = -180:1:180; % 扫描方向范围
v = exp(-1i * 2 * pi * n * d * sind(sita) / lambda); % 扫描方向的导向矢量
B = abs(v' * W_opt); % 波束响应

figure;
plot(sita, 20 * log10(B / max(B)), 'k');
title('自适应波束形成');
xlabel('角度/degree');
ylabel('功率/dB');
grid on;
axis([-180, 180, -50, 0]);  % 设置坐标轴范围
