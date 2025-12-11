%MATLAB实现：最大信噪比滤波器的波束方向图。具有十个传感器的等间距线性阵列，
%阵元间距d=8cm，噪声信号来自一个单位幅度、频率为2khz的点窄带源，
% 该噪声源位于远场，以60°的入射角传播到阵列

% 清空工作区、关闭所有图形窗口、清空命令窗口
clear;
close all;
clc;

% 定义参数
c = 1500; % 声速，单位 m/s
f = 2000; % 频率，单位 Hz
lambda = c / f; % 波长，单位 m
d = 0.08; % 阵元间距，单位 m
N = 10; % 阵元数量
theta_noise = 60; % 噪声源入射角，单位 度
theta_noise_rad = deg2rad(theta_noise); % 转换为弧度
theta = -90:0.1:90; % 扫描角度范围，单位 度
theta_rad = deg2rad(theta); % 转换为弧度

% 生成噪声导向矢量
a_noise = exp(1j * 2 * pi * (0:N-1)' * (d / lambda) * sin(theta_noise_rad));

% 计算噪声协方差矩阵
R_n = a_noise * a_noise';

% 计算噪声协方差矩阵的逆
R_n_inv = inv(R_n);

% 计算波束方向图
beam_pattern = zeros(size(theta));
for i = 1:length(theta)
    % 生成信号导向矢量
    a_signal = exp(1j * 2 * pi * (0:N-1)' * (d / lambda) * sin(theta_rad(i)));
    
    % 计算最大信噪比滤波器权重
    w = R_n_inv * a_signal / (a_signal' * R_n_inv * a_signal);
    
    % 计算波束响应
    beam_pattern(i) = abs(w' * a_signal);
end

% 归一化波束方向图
beam_pattern = beam_pattern / max(beam_pattern);

% 笛卡尔坐标系下绘制波束方向图
figure;
plot(theta, 20 * log10(beam_pattern));
title('最大信噪比滤波器波束方向图');
xlabel('角度 (度)');
ylabel('幅度 (dB)');
grid on;

% 极坐标系下绘制波束方向图
figure;
polarplot(theta_rad, beam_pattern);
title('最大信噪比滤波器波束方向图');