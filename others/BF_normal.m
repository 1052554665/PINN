% 常规波束形成
% 假设有一个包含4个阵元的均匀线阵，阵元间距为半波长
N = 4; % 阵元数量
wavelength = 1; % 波长
d = 0.5 * wavelength; % 阵元间距，设为半波长
theta = -180:1:180; % 方向角范围，从-180°-180°，步长为1°
theta_0 = 0; % 波束指向角，设为0°

% 使用矩阵运算计算阵列因子
% 使用repmat将方向角theta和阵元索引n扩展为矩阵
theta_grid = repmat(theta, N, 1); % 将方向角扩展为矩阵
n = (0:N-1)'; % 阵元索引
n_grid = repmat(n, 1, length(theta)); % 将阵元索引扩展为矩阵
AF = exp(-1i * 2 * pi * d * n_grid .* sind(theta_grid - theta_0) / wavelength);
AF = sum(AF, 1); % 沿阵元方向（矩阵的行方向）求和，得到总的阵列因子

% 绘制波束图
figure;
plot(theta, abs(AF) / max(abs(AF))); % 归一化波束幅度
title('常规波束形成');
xlabel('角度/degree');
ylabel('波束幅度');
grid on;
