clc; clear; close all;

%% 1. 离散正弦信号（每周期 8 点）
N_per = 8;                 % 每周期 8 点
num_periods = 4;           % 4 个周期
N = N_per * num_periods;   % 总点数 = 32

n = 0:N-1;
x = sin(2*pi*(1/N_per)*n); % 离散正弦

%% 2. Hamming 窗
w = hamming(N).';

%% 3. 加窗信号
xw = x .* w;

%% 4. 平滑插值（用于平滑曲线）
ns = linspace(0, N-1, 400);      % 更密集的横坐标（400个点）
xs  = interp1(n, x,  ns, 'spline');
xws = interp1(n, xw, ns, 'spline');

%% 4. 绘制在同一张图中
figure; hold on; grid on;

plot(n, x, '-o', 'LineWidth', 1.5, 'MarkerSize', 5, ...
     'DisplayName', 'sin 信号');
plot(n, w, 'LineWidth', 1.5, 'MarkerSize', 5, ...
     'DisplayName', 'Hamming 窗');
% plot(n, w, '-s', 'LineWidth', 1.5, 'MarkerSize', 5, ...
%      'DisplayName', 'Hamming 窗');
% plot(n, xw, '-d', 'LineWidth', 1.5, 'MarkerSize', 5, ...
%      'DisplayName', '加窗信号 x[n]w[n]');
plot(ns, xws, 'LineWidth', 1.5, 'MarkerSize', 5, ...
     'DisplayName', '加窗信号 x[n]w[n]');

xlabel('n'); ylabel('幅度');
title('离散正弦信号 / Hamming 窗 / 加窗信号');
legend('show');
