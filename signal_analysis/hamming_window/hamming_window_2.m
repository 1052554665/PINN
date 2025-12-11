clc; clear; close all;

%% 1. 离散正弦信号（每周期 8 点）
N_per = 8;                 % 每周期 8 点
num_periods = 4;           % 4 个周期
N = N_per * num_periods;   % 总点数 = 32

n = 0:N-1;
x = sin(2*pi*(1/N_per)*n);

%% 2. Hamming 窗
w = hamming(N).';

%% 3. 加窗信号
xw = x .* w;

%% 4. 生成平滑插值曲线
ns = linspace(0, N-1, 400);      % 更密集采样点
xs  = interp1(n, x,  ns, 'spline');
ws  = interp1(n, w,  ns, 'spline');
xws = interp1(n, xw, ns, 'spline');

%% 5. 绘图
figure; hold on; grid on;

%% --- 平滑曲线 ---
plot(ns, xs, 'black', 'LineWidth', 2, 'DisplayName', 'sin 平滑曲线');
plot(ns, ws, 'blue', 'LineWidth', 2, 'DisplayName', 'Hamming 窗平滑曲线');
plot(ns, xws,'r', 'LineWidth', 2, 'DisplayName', '加窗信号平滑曲线');

% %% --- 原始离散点 ---
% stem(n, x,  'bo', 'filled', 'DisplayName', 'sin 离散点');
stem(n, x,  'bo', 'filled', 'DisplayName', 'sin 离散点');
% stem(n, w,  'ko', 'filled', 'DisplayName', 'Hamming 窗离散点');
% stem(n, xw, 'go', 'filled', 'DisplayName', '加窗信号离散点');

xlabel('n');
ylabel('幅度');
title('离散正弦、窗函数、加窗信号（平滑曲线 + 离散点）');
legend('show');
