% Arcondoulis 螺旋阵列 —— 修正版本（处理 N 不能整除 M 的情况）
clear; clc; close all;

%% 参数
N = 128;        % 阵元总数
M = 6;          % 螺旋臂数
a = 0.01;       % 起始半径 (m)
b = 0.005;      % 阿基米德螺旋的螺距
nTurns = 3;     % 螺旋圈数
fc = 20e3;      % 频率 (Hz)
c = 340;        % 传播速度 (m/s)
lambda = c / fc;
k0 = 2*pi / lambda;   % 波数

%% 将 N 均匀（尽量）分配到 M 臂，处理不能整除的情况
N_base = floor(N / M);
remainder = N - N_base * M;      % 余数需要分配给前 remainder 个臂
N_arm_counts = repmat(N_base, 1, M);
if remainder > 0
    N_arm_counts(1:remainder) = N_arm_counts(1:remainder) + 1;
end
% 验证
totalAssigned = sum(N_arm_counts);
if totalAssigned ~= N
    error('分配错误：总分配(%d) != N(%d)', totalAssigned, N);
end

%% 生成阵元位置（按臂分配）
pos = zeros(3, N);
idx = 1;
for mIdx = 1:M
    nElemThisArm = N_arm_counts(mIdx);
    if nElemThisArm == 1
        % 只有一个点时直接取 theta = 0
        thetas = 0;
    else
        thetas = (0:(nElemThisArm-1)) ./ (nElemThisArm-1) * (2*pi*nTurns);
    end
    for j = 1:nElemThisArm
        theta = thetas(j);
        r = a + b * theta;                 % 阿基米德螺旋
        th = theta + 2*pi*(mIdx-1)/M;      % 每臂相位偏移
        x = r * cos(th);
        y = r * sin(th);
        z = 0;
        pos(:, idx) = [x; y; z];
        idx = idx + 1;
    end
end

%% 方向余弦网格
ux = linspace(-1,1,401);   % 网格加密一些
uy = linspace(-1,1,401);
[UX, UY] = meshgrid(ux, uy);
mask = (UX.^2 + UY.^2) <= 1;

%% 计算阵列因子（注意只对已分配的 N 个元素循环）
AF = zeros(size(UX));
for nIdx = 1:N
    xn = pos(1, nIdx);
    yn = pos(2, nIdx);
    zn = pos(3, nIdx);
    uz = sqrt(max(0, 1 - UX.^2 - UY.^2));   % 取正根（上半球）
    AF = AF + exp(1j * k0 * (xn * UX + yn * UY + zn * uz));
end

%% 归一化并转 dB
AF = AF ./ max(abs(AF(:)));
AFdB = 20 * log10(abs(AF) + eps);
AFdB(~mask) = -60;    % 单位球面外填低值（更低以便观察）

%% 绘制阵列几何（分臂连线）
figure('Name','几何结构'); 
scatter(pos(1,:), pos(2,:), 36, 'r', 'filled'); hold on;
% 每臂连线
startIdx = 1;
for mIdx = 1:M
    nElemThisArm = N_arm_counts(mIdx);
    idx_arm = startIdx : (startIdx + nElemThisArm - 1);
    plot(pos(1, idx_arm), pos(2, idx_arm), '--k', 'LineWidth', 1);
    startIdx = startIdx + nElemThisArm;
end
text(pos(1,:), pos(2,:), arrayfun(@num2str, 1:N, 'UniformOutput', false), ...
    'FontSize',6, 'VerticalAlignment','bottom', 'HorizontalAlignment','right');
axis equal; grid on;
xlabel('x (m)'); ylabel('y (m)');
title(sprintf('Arcondoulis 螺旋阵列 (N=%d, M=%d)', N, M));

%% 二维方向图（ux-uy 平面）
figure('Name','二维方向图');
imagesc(ux, uy, AFdB);
set(gca,'YDir','normal');
axis square;
colorbar;
caxis([-40 0]);
xlabel('u_x'); ylabel('u_y');
title('Arcondoulis 阵列二维方向图 (方向余弦坐标)');

%% 三维方向图
figure('Name','三维方向图');
surf(UX, UY, AFdB, 'EdgeColor', 'none');
axis square; view(30,40);
colorbar; caxis([-40 0]);
xlabel('u_x'); ylabel('u_y'); zlabel('增益 (dB)');
title('Arcondoulis 阵列三维方向图');

