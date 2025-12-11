clc;
clear;
M = 18; % 天线数
lambda = 10;
d = lambda / 2;
L = 100;  % 快拍数
thetas = [10];    % 期望信号入射角度
thetai = [-30 30]; % 干扰入射角度
n = [0:M-1]'; % 天线索引
vs = exp(-1j * 2 * pi * n * d * sind(thetas) / lambda); % 信号方向向量
vn = exp(-1j * 2 * pi * n * d * sind(thetai) / lambda); % 干扰方向向量
f = 1600; % 载波频率
t = [0:L-1];
di = sin(2*pi*f*t/(8*f));    % 期望信号
vn1 = sin(2*pi*2 * f*t/(8*f));  % 干扰信号1 
vn2 = sin(2*pi*4 * f*t/(8*f));  % 干扰信号2 
A = [vs vn];
St = [di;vn1;vn2];
Xt = A*St + randn(M,L);   % 矩阵形式的公式
R_x = 1/L * (Xt * Xt');
R_x_inv = inv(R_x + 1e-6 * eye(size(R_x))); % 添加正则化项
W_opt = R_x_inv * vs / (vs' * R_x_inv * vs);
% 测试此时的方向图
sita = linspace(-180, 180, 3600); % 扩展角度范围
v = exp(-1i*2*pi*n* d*sind(sita)/lambda); % 不同角度的方向矢量
B = abs(W_opt' * v);
plot(sita,20*log10(B/max(B)),'k') % 绘制归一化波束图
title('波束图')
xlabel('角度/degree')
ylabel('波束图/dB')
grid on
