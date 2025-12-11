%matlab实现：分别在笛卡尔坐标系和极坐标系下绘制一个有十个传感器的
%等间距线性阵列的最小二乘法波束形成器的波束方向图，间距d=4cm，
% 角度范围60-120，频率1.5kHz

% 清空工作区、关闭所有图形窗口、清空命令窗口
clear;
close all;
clc;

% 定义参数
c = 1500; % 声速，单位 m/s
f = 1500; % 频率，单位 Hz
lambda = c / f; % 波长，单位 m
d = 0.04; % 阵元间距，单位 m
N = 10; % 阵元数量
theta_range = [60, 120]; % 角度范围，单位 度
theta = linspace(theta_range(1), theta_range(2), 360); % 生成指定角度范围的角度值
theta_rad = deg2rad(theta); % 转换为弧度

% 生成导向矢量矩阵 A
A = zeros(N, length(theta));
for i = 1:length(theta)
    A(:, i) = exp(1j * 2 * pi * (0:N - 1)' * (d / lambda) * sin(theta_rad(i)));
end

% 期望响应矢量
d_0 = ones(length(theta), 1); 

% 最小二乘法求解权重向量 w
% 先计算 A' * A 的伪逆
A_pinv = pinv(A' * A);
% 计算中间结果 A * A_pinv
intermediate_result = A * A_pinv;
% 确保维度匹配进行矩阵乘法
w = intermediate_result * d_0; 

% 计算波束方向图
beam_pattern = zeros(size(theta));
for i = 1:length(theta)
    steering_vector = exp(1j * 2 * pi * (0:N - 1)' * (d / lambda) * sin(theta_rad(i)));
    beam_pattern(i) = abs(w' * steering_vector);
end

% 归一化波束方向图
beam_pattern = beam_pattern / max(beam_pattern);

% 笛卡尔坐标系下绘制波束方向图
figure;
plot(theta, 20 * log10(beam_pattern));
title('最小二乘法波束形成器波束方向图');
xlabel('角度 (度)');
ylabel('幅度 (dB)');
grid on;

% 极坐标系下绘制波束方向图
figure;
polarplot(theta_rad, beam_pattern);
title('最小二乘法波束形成器波束方向图');