% 清空工作区、关闭所有图形窗口、清空命令窗口
clear;
close all;
clc;

% 假设信号和参数
N = 10; % 阵元数量
d = 0.5; % 阵元间距
M = 5; % 观测数据数量
threshold = 0.1; % 阈值

% 生成信号
signal = randn(N, M); % 随机信号

% 生成测量矩阵
Phi = randn(M, N); % 随机测量矩阵

% 生成观测数据
y = Phi * signal; % 观测数据

% 使用 L1 正则化重建信号
cvx_begin
    variable x(N, M) complex; % 重建信号
    minimize(norm(x, 1)); % L1 正则化
    subject to
        Phi * x == y; % 测量约束
cvx_end

% 重建的信号
reconstructed_signal = x;

% 计算波束方向图
AF = abs(reconstructed_signal); % 取信号的幅度作为阵列因子

% 绘制波束图
figure;
plot(AF);
title('基于压缩感知的波束形成');
xlabel('阵元索引');
ylabel('波束幅度');
grid on;
