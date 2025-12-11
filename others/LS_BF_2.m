% 用于矩阵乘法的维度不正确，无法执行！！

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
theta_range = [85, 95]; % 角度范围，单位 度
theta = linspace(theta_range(1), theta_range(2), 360); % 生成指定角度范围的角度值
theta_rad = deg2rad(theta); % 转换为弧度

% 生成导向矢量矩阵 A
A = zeros(N, length(theta));
for i = 1:length(theta)
    A(:, i) = exp(1j * 2 * pi * (0:N - 1)' * (d / lambda) * sin(theta_rad(i)));
end

% 期望响应矢量，通常在主瓣方向设为 1，其余为 0 或其他值
% 在主瓣方向（85度）设置为1，其他方向为0
desired_response = zeros(length(theta), 1); % 确保是一个列向量
[~, idx] = min(abs(theta - 85)); % 找到最接近85度的索引
desired_response(idx) = 1;

% 最小二乘法求解权重向量 w
w = pinv(A) * desired_response; % 使用伪逆求解权重

% 计算波束方向图
beam_pattern = abs(A' * w); % 计算波束方向图

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
