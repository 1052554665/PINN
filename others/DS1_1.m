% 分别在笛卡尔坐标系和极坐标系下绘制一个有十个传感器的等间距线性阵列的延迟求和
% 波束形成器的波束方向图，间距d=8cm，入射角90°，频率2kHz

clear;
close all;
clc;

% 参数设置
N = 10; % 传感器数量
d = 0.08; % 传感器间距（单位：米）
f = 2e3; % 频率（单位：赫兹）
c = 343; % 声速（单位：米/秒）
theta_inc = 90; % 入射角（单位：度）

% 波长
lambda = c / f; % 波长 = 声速 / 频率

% 波束方向图计算
theta = linspace(0, 360, 361); % 方向角范围（单位：度）
theta_rad = deg2rad(theta); % 转换为弧度

% 阵列因子计算
AF = zeros(1, length(theta));
for n = 1:N
    AF = AF + exp(-1j * 2 * pi * (n-1) * d * sin(theta_rad) / lambda);
end

% 归一化阵列因子
AF = AF / max(abs(AF));

% 转换为dB
AF_dB = 20 * log10(abs(AF));

% 绘制笛卡尔坐标系下的波束方向图
figure;
plot(theta, AF_dB);
xlabel('角度 (度)');
ylabel('幅度 (dB)');
title('延迟求和波束形成器的波束方向图');
grid on;

% 绘制极坐标系下的波束方向图
figure;
polarplot(theta_rad, abs(AF));
title('延迟求和波束形成器的波束方向图');
