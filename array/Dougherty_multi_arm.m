% Dougherty 多臂对数螺旋阵列 (N=128) - 阵列几何 + 二维/三维方向图
clear; clc; close all;

%% ===== 用户参数 =====
N = 128;                  % 总阵元数
M = 4;                    % 螺旋臂数 (可改为 2, 3, 6 ...)
r_min = 0.015;            % 最小半径 (m)
r_max = 0.100;            % 最大半径 (m)
theta_max = 13*pi/2;      % 最大旋转角 (rad)

% 对数螺旋系数 b (由边界条件确定)
b = log(r_max / r_min) / theta_max;
a0 = r_min;

% 波参数 (默认声学场)
fc = 20e3;                % 频率 Hz
c = 340;                  % 速度 m/s
lambda = c / fc;
k0 = 2*pi / lambda;

%% ===== 生成多臂阵元位置 =====
N_arm = floor(N / M);     % 每臂阵元数
pos = zeros(3, N);        % 保存所有位置
idx = 1;

for m = 0:M-1
    theta = linspace(0, theta_max, N_arm);
    r = a0 * exp(b * theta);
    th = theta + 2*pi*m/M;      % 每臂旋转偏移
    x = r .* cos(th);
    y = r .* sin(th);
    z = zeros(size(x));
    pos(:, idx:idx+N_arm-1) = [x; y; z];
    idx = idx + N_arm;
end

%% ===== 绘制阵列几何结构 =====
figure('Name','阵列几何','NumberTitle','off');
scatter(pos(1,:), pos(2,:), 48, 'filled'); hold on;
for m = 1:M
    idx_arm = (m-1)*N_arm + (1:N_arm);
    plot(pos(1,idx_arm), pos(2,idx_arm), '--k', 'LineWidth', 0.8);
    text(pos(1,idx_arm), pos(2,idx_arm), ...
         arrayfun(@num2str, idx_arm, 'UniformOutput', false), ...
         'FontSize',6,'VerticalAlignment','bottom','HorizontalAlignment','right');
end
axis equal; grid on;
xlabel('x (m)'); ylabel('y (m)');
title(sprintf('Dougherty 对数螺旋阵列 (M=%d, N=%d)', M, N));

%% ===== 方向余弦网格 =====
ux = linspace(-1, 1, 401);
uy = linspace(-1, 1, 401);
[UX, UY] = meshgrid(ux, uy);
mask = (UX.^2 + UY.^2) <= 1;
UZ = sqrt(max(0, 1 - UX.^2 - UY.^2));

%% ===== 阵列因子 =====
AF = zeros(size(UX));
for n = 1:N
    xn = pos(1,n); yn = pos(2,n); zn = pos(3,n);
    AF = AF + exp(1j * k0 * (xn*UX + yn*UY + zn*UZ));
end
AF = AF ./ max(abs(AF(:)));
AFdB = 20*log10(abs(AF)+eps);
AFdB(~mask) = -80;

%% ===== 二维方向图 =====
figure('Name','二维方向图','NumberTitle','off');
imagesc(ux, uy, AFdB);
set(gca,'YDir','normal'); axis square;
colorbar; caxis([-60 0]);
xlabel('u_x'); ylabel('u_y');
title('二维方向图 (u_x,u_y)');
hold on;
t = linspace(0,2*pi,400);
plot(cos(t), sin(t), 'w--','LineWidth',1);

%% ===== 三维方向图 =====
figure('Name','三维方向图','NumberTitle','off');
surf(UX, UY, AFdB, 'EdgeColor','none');
axis square; view(30,40); shading interp;
colorbar; caxis([-60 0]);
xlabel('u_x'); ylabel('u_y'); zlabel('增益 (dB)');
title('三维方向图 (u_x,u_y,dB)');
